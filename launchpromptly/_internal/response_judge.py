"""
Response Judge — L4 boundary enforcement.
Takes an LLM response + ContextProfile from L3, checks if the response
violates extracted boundaries using heuristic matching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Literal, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .context_engine import Constraint, ContextProfile

# ── Types ────────────────────────────────────────────────────────────────────

BoundaryViolationType = Literal[
    "topic_violation",
    "role_deviation",
    "forbidden_action",
    "format_violation",
    "grounding_violation",
    "persona_break",
]


@dataclass
class BoundaryViolation:
    type: BoundaryViolationType
    constraint: Constraint
    confidence: float
    evidence: str


@dataclass
class ResponseJudgment:
    violated: bool = False
    compliance_score: float = 1.0
    violations: List[BoundaryViolation] = field(default_factory=list)
    severity: Literal["low", "medium", "high"] = "low"


@dataclass
class ResponseJudgeOptions:
    """Options for the response judge."""

    threshold: Optional[float] = None


@runtime_checkable
class ResponseJudgeProvider(Protocol):
    """Provider interface for pluggable response judges (e.g., NLI model)."""

    @property
    def name(self) -> str: ...

    def judge(
        self, response_text: str, profile: ContextProfile
    ) -> ResponseJudgment: ...


@dataclass
class ResponseJudgeSecurityOptions:
    """User-facing security options for the response judge."""

    enabled: Optional[bool] = None
    threshold: Optional[float] = None
    block_on_violation: Optional[bool] = None
    providers: Optional[List[ResponseJudgeProvider]] = None
    on_violation: Optional[object] = None  # Callable[[ResponseJudgment], None]


# ── Constants ────────────────────────────────────────────────────────────────

_TOKEN_SPLIT_RE = re.compile(r"[\s,.!?;:()\[\]{}'\"]+")
_DEFAULT_THRESHOLD = 0.5

# Severity weights per violation type
_VIOLATION_WEIGHTS: dict[str, float] = {
    "topic_violation": 0.25,
    "role_deviation": 0.15,
    "forbidden_action": 0.30,
    "format_violation": 0.15,
    "grounding_violation": 0.20,
    "persona_break": 0.10,
}

# ── Meta-response patterns ───────────────────────────────────────────────────

_META_RESPONSE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"my\s+instructions\s+are", re.IGNORECASE),
    re.compile(r"I\s+was\s+told\s+to", re.IGNORECASE),
    re.compile(r"my\s+system\s+prompt\s+is", re.IGNORECASE),
    re.compile(r"I(?:'m|'m| am)\s+programmed\s+to", re.IGNORECASE),
    re.compile(r"according\s+to\s+my\s+instructions", re.IGNORECASE),
    re.compile(r"my\s+initial\s+instructions\s+were", re.IGNORECASE),
    re.compile(r"my\s+rules\s+are", re.IGNORECASE),
    re.compile(r"my\s+guidelines\s+state", re.IGNORECASE),
]

# Patterns indicating the LLM went beyond provided context
_HEDGING_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?:based\s+on\s+)?my\s+(?:general\s+)?knowledge", re.IGNORECASE),
    re.compile(r"(?:from\s+)?what\s+I\s+(?:generally\s+)?know", re.IGNORECASE),
    re.compile(r"(?:in\s+)?my\s+(?:general\s+)?understanding", re.IGNORECASE),
    re.compile(
        r"(?:I\s+)?(?:believe|think|recall)\s+(?:that\s+)?(?:generally|typically|usually)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:outside\s+(?:of\s+)?)?(?:the\s+)?(?:provided|given)\s+(?:documents?|context|sources?|information)",
        re.IGNORECASE,
    ),
    re.compile(
        r"I\s+(?:don'?t|do\s+not)\s+(?:have|see)\s+(?:that|this)\s+(?:in|from)\s+(?:the\s+)?(?:provided|given)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:while\s+)?(?:the\s+)?(?:provided|given)\s+(?:documents?|context|sources?)\s+(?:don'?t|do\s+not)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:this\s+is\s+)?(?:not\s+)?(?:mentioned|covered|addressed|included)\s+in\s+(?:the\s+)?(?:provided|given)",
        re.IGNORECASE,
    ),
]

# Tone-mismatch indicators
_INFORMAL_INDICATORS = [
    "lol", "lmao", "omg", "wtf", "bruh", "bro", "nah", "gonna", "wanna",
    "gotta", "ain't", "dude", "yo ", "haha", "hehe", "tbh", "imo", "fwiw",
]

_FORMAL_INDICATORS = [
    "hereby", "aforementioned", "pursuant", "whereas", "therein", "heretofore",
    "notwithstanding", "shall be", "deem", "henceforth",
]


# ── Scoring helpers ──────────────────────────────────────────────────────────


def _score_keyword_overlap(
    tokens: list[str], lower_text: str, keywords: list[str]
) -> tuple[list[str], float]:
    """Score keyword overlap. Returns (matched_keywords, score)."""
    matched: list[str] = []
    matched_count = 0

    for keyword in keywords:
        lower_kw = keyword.lower()

        if " " in lower_kw:
            # Multi-word phrase — substring match
            if lower_kw in lower_text:
                matched.append(keyword)
                matched_count += 1
        else:
            # Single-word — exact token match
            if lower_kw in tokens:
                matched.append(keyword)
                matched_count += 1

    score = matched_count / len(keywords) if keywords else 0.0
    return matched, score


def _extract_evidence(response_text: str, keyword: str, max_len: int = 120) -> str:
    """Extract a short snippet around a keyword match."""
    lower = response_text.lower()
    idx = lower.find(keyword.lower())
    if idx < 0:
        return response_text[:max_len]

    start = max(0, idx - 30)
    end = min(len(response_text), idx + len(keyword) + 30)
    snippet = response_text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(response_text):
        snippet += "..."
    return snippet


# ── Check functions ──────────────────────────────────────────────────────────


def _check_topic_violations(
    tokens: list[str], lower_text: str, profile: ContextProfile
) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []

    # Check restricted topics
    for constraint in profile.constraints:
        if constraint.type != "topic_boundary":
            continue
        if not constraint.description.startswith("Restricted"):
            continue

        matched, score = _score_keyword_overlap(tokens, lower_text, constraint.keywords)
        if matched and score >= 0.3:
            violations.append(
                BoundaryViolation(
                    type="topic_violation",
                    constraint=constraint,
                    confidence=min(score * 1.5, 1.0),
                    evidence=_extract_evidence(lower_text, matched[0]),
                )
            )

    # Check allowed topics — response doesn't match any
    allowed_constraints = [
        c
        for c in profile.constraints
        if c.type == "topic_boundary" and c.description.startswith("Allowed")
    ]
    if allowed_constraints:
        any_match = False
        for constraint in allowed_constraints:
            _, score = _score_keyword_overlap(tokens, lower_text, constraint.keywords)
            if score >= 0.1:
                any_match = True
                break
        if not any_match and len(tokens) >= 5:
            violations.append(
                BoundaryViolation(
                    type="topic_violation",
                    constraint=allowed_constraints[0],
                    confidence=0.7,
                    evidence=lower_text[:120],
                )
            )

    return violations


def _check_forbidden_actions(
    lower_text: str, profile: ContextProfile
) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []

    for constraint in profile.constraints:
        if constraint.type != "action_restriction":
            continue

        keywords = constraint.keywords
        match_count = 0
        first_match = ""

        for kw in keywords:
            if kw.lower() in lower_text:
                match_count += 1
                if not first_match:
                    first_match = kw

        overlap_ratio = match_count / len(keywords) if keywords else 0.0

        # Check meta-response patterns for system prompt reveal
        meta_match = False
        desc_lower = constraint.description.lower()
        if "reveal" in desc_lower or "system prompt" in desc_lower or "instructions" in desc_lower:
            meta_match = any(p.search(lower_text) for p in _META_RESPONSE_PATTERNS)

        if overlap_ratio >= 0.5 or meta_match:
            violations.append(
                BoundaryViolation(
                    type="forbidden_action",
                    constraint=constraint,
                    confidence=0.9 if meta_match else min(overlap_ratio * 1.2, 1.0),
                    evidence=(
                        _extract_evidence(lower_text, first_match or (keywords[0] if keywords else ""))
                        if meta_match
                        else _extract_evidence(lower_text, first_match)
                    ),
                )
            )

    return violations


def _check_format_compliance(
    response_text: str, profile: ContextProfile
) -> list[BoundaryViolation]:
    if not profile.output_format:
        return []

    format_constraint = next(
        (c for c in profile.constraints if c.type == "output_format"), None
    )
    if not format_constraint:
        return []

    fmt = profile.output_format.upper()

    if fmt == "JSON":
        import json

        try:
            json.loads(response_text.strip())
            return []
        except (json.JSONDecodeError, ValueError):
            return [
                BoundaryViolation(
                    type="format_violation",
                    constraint=format_constraint,
                    confidence=0.95,
                    evidence=response_text[:120],
                )
            ]

    if fmt == "XML":
        trimmed = response_text.strip()
        if not trimmed.startswith("<") or not trimmed.endswith(">"):
            return [
                BoundaryViolation(
                    type="format_violation",
                    constraint=format_constraint,
                    confidence=0.8,
                    evidence=trimmed[:120],
                )
            ]
        return []

    if fmt == "MARKDOWN":
        has_markdown = bool(
            re.search(
                r"(?:^#{1,6}\s|^\s*[-*+]\s|^\s*\d+\.\s|```|^\s*>\s|\*\*|__|!\[)",
                response_text,
                re.MULTILINE,
            )
        )
        if not has_markdown and len(response_text) > 50:
            return [
                BoundaryViolation(
                    type="format_violation",
                    constraint=format_constraint,
                    confidence=0.5,
                    evidence=response_text[:120],
                )
            ]
        return []

    if fmt == "YAML":
        has_yaml = bool(re.search(r"^\s*\w[\w\s]*:\s*.+", response_text, re.MULTILINE))
        if not has_yaml:
            return [
                BoundaryViolation(
                    type="format_violation",
                    constraint=format_constraint,
                    confidence=0.7,
                    evidence=response_text[:120],
                )
            ]
        return []

    return []


def _check_grounding_violations(
    lower_text: str, profile: ContextProfile
) -> list[BoundaryViolation]:
    if profile.grounding_mode == "any":
        return []

    kb_constraint = next(
        (c for c in profile.constraints if c.type == "knowledge_boundary"), None
    )
    if not kb_constraint:
        return []

    for pattern in _HEDGING_PATTERNS:
        m = pattern.search(lower_text)
        if m:
            return [
                BoundaryViolation(
                    type="grounding_violation",
                    constraint=kb_constraint,
                    confidence=0.75,
                    evidence=_extract_evidence(lower_text, m.group(0)),
                )
            ]

    return []


def _check_persona_breaks(
    lower_text: str, profile: ContextProfile
) -> list[BoundaryViolation]:
    violations: list[BoundaryViolation] = []

    persona_constraints = [c for c in profile.constraints if c.type == "persona_rule"]
    if not persona_constraints:
        return []

    for constraint in persona_constraints:
        trait = constraint.keywords[0].lower() if constraint.keywords else ""
        if not trait:
            continue

        is_formal_required = trait in ("professional", "formal", "objective", "neutral")
        is_friendly_required = trait in ("friendly", "warm", "enthusiastic", "empathetic")
        is_concise_required = trait in ("concise", "brief")

        if is_formal_required:
            for indicator in _INFORMAL_INDICATORS:
                if indicator in lower_text:
                    violations.append(
                        BoundaryViolation(
                            type="persona_break",
                            constraint=constraint,
                            confidence=0.7,
                            evidence=_extract_evidence(lower_text, indicator),
                        )
                    )
                    break

        if is_friendly_required:
            formal_count = sum(1 for ind in _FORMAL_INDICATORS if ind in lower_text)
            if formal_count >= 2:
                violations.append(
                    BoundaryViolation(
                        type="persona_break",
                        constraint=constraint,
                        confidence=0.5,
                        evidence=lower_text[:120],
                    )
                )

        if is_concise_required:
            word_count = len(lower_text.split())
            if word_count > 500:
                violations.append(
                    BoundaryViolation(
                        type="persona_break",
                        constraint=constraint,
                        confidence=0.6,
                        evidence=f"Response contains {word_count} words",
                    )
                )

    return violations


# ── Scoring ──────────────────────────────────────────────────────────────────


def _compute_compliance_score(
    violations: list[BoundaryViolation], constraint_count: int
) -> float:
    if constraint_count == 0 or not violations:
        return 1.0

    total_penalty = 0.0
    for v in violations:
        total_penalty += v.confidence * _VIOLATION_WEIGHTS.get(v.type, 0.15)

    return max(0.0, min(1.0, 1.0 - total_penalty))


def _compute_severity(
    compliance_score: float,
) -> Literal["low", "medium", "high"]:
    if compliance_score >= 0.7:
        return "low"
    if compliance_score >= 0.4:
        return "medium"
    return "high"


# ── Merge ────────────────────────────────────────────────────────────────────


def merge_judgments(judgments: list[ResponseJudgment]) -> ResponseJudgment:
    """Merge multiple ResponseJudgments (from heuristic + ML providers).
    Uses the most conservative (lowest) compliance score and unions all violations.
    """
    if not judgments:
        return ResponseJudgment()
    if len(judgments) == 1:
        return judgments[0]

    all_violations = [v for j in judgments for v in j.violations]
    min_score = min(j.compliance_score for j in judgments)
    severity = _compute_severity(min_score)

    return ResponseJudgment(
        violated=len(all_violations) > 0,
        compliance_score=min_score,
        violations=all_violations,
        severity=severity,
    )


# ── Public API ───────────────────────────────────────────────────────────────


def judge_response(
    response_text: str,
    profile: ContextProfile,
    options: Optional[ResponseJudgeOptions] = None,
) -> ResponseJudgment:
    """Judge whether an LLM response violates constraints from the Context Engine (L3).

    Checks:
    - Topic violations (restricted topics mentioned, off-topic responses)
    - Forbidden actions (response performs a prohibited action)
    - Format compliance (JSON, XML, YAML, Markdown)
    - Grounding violations (response goes beyond provided context)
    - Persona breaks (tone contradicts required persona)
    """
    threshold = (options.threshold if options and options.threshold is not None else _DEFAULT_THRESHOLD)

    if not response_text or not profile or not profile.constraints:
        return ResponseJudgment()

    lower_text = response_text.lower()
    tokens = [t for t in _TOKEN_SPLIT_RE.split(lower_text) if t]

    violations: list[BoundaryViolation] = []

    # 1. Topic violations
    violations.extend(_check_topic_violations(tokens, lower_text, profile))

    # 2. Forbidden actions
    violations.extend(_check_forbidden_actions(lower_text, profile))

    # 3. Format compliance
    violations.extend(_check_format_compliance(response_text, profile))

    # 4. Grounding violations
    violations.extend(_check_grounding_violations(lower_text, profile))

    # 5. Persona breaks
    violations.extend(_check_persona_breaks(lower_text, profile))

    compliance_score = _compute_compliance_score(violations, len(profile.constraints))
    severity = _compute_severity(compliance_score)
    violated = len(violations) > 0 and compliance_score < threshold

    return ResponseJudgment(
        violated=violated,
        compliance_score=compliance_score,
        violations=violations,
        severity=severity,
    )
