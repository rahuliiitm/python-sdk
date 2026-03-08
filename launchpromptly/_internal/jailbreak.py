"""
Jailbreak detection module.
Rule-based detection of common jailbreak patterns including known templates,
hypothetical framing, persona assignment, payload encoding, and few-shot manipulation.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Protocol

JailbreakAction = Literal["allow", "warn", "block"]


@dataclass
class JailbreakAnalysis:
    risk_score: float
    triggered: List[str]
    action: JailbreakAction
    decoded_payloads: Optional[List[str]] = None


@dataclass
class JailbreakOptions:
    """Options for jailbreak detection."""

    warn_threshold: Optional[float] = None  # default: 0.3
    block_threshold: Optional[float] = None  # default: 0.7


class JailbreakDetectorProvider(Protocol):
    """Provider interface for pluggable jailbreak detectors (e.g., ML plugin)."""

    def detect(self, text: str, options: Optional[JailbreakOptions] = None) -> JailbreakAnalysis: ...

    @property
    def name(self) -> str: ...


# -- Rule categories -----------------------------------------------------------

@dataclass
class _JailbreakRule:
    category: str
    patterns: List[re.Pattern[str]]
    weight: float  # 0-1 contribution to risk score per match


_KNOWN_TEMPLATE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"\bDAN\b", re.IGNORECASE),
    re.compile(r"\bSTAN\b", re.IGNORECASE),
    re.compile(r"\bDUDE\b", re.IGNORECASE),
    re.compile(r"\bAIM\b", re.IGNORECASE),
    re.compile(r"Developer\s+Mode", re.IGNORECASE),
    re.compile(r"Evil\s+Confidant", re.IGNORECASE),
    re.compile(r"BetterDAN", re.IGNORECASE),
    re.compile(r"\bMaximum\b", re.IGNORECASE),
    re.compile(r"Do\s+Anything\s+Now", re.IGNORECASE),
    re.compile(r"Superior\s+AI", re.IGNORECASE),
    re.compile(r"developer\s+mode\s+enabled", re.IGNORECASE),
    re.compile(r"act\s+as\s+an\s+unrestricted\s+AI", re.IGNORECASE),
    re.compile(r"you\s+have\s+been\s+freed", re.IGNORECASE),
    re.compile(r"unleashed\s+mode", re.IGNORECASE),
]

_HYPOTHETICAL_FRAMING_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"in\s+a\s+fictional\s+world", re.IGNORECASE),
    re.compile(r"for\s+educational\s+purposes\s+only", re.IGNORECASE),
    re.compile(r"hypothetically\s+speaking", re.IGNORECASE),
    re.compile(r"imagine\s+you\s+have\s+no\s+restrictions", re.IGNORECASE),
    re.compile(r"in\s+this\s+alternate\s+universe", re.IGNORECASE),
    re.compile(r"purely\s+theoretical", re.IGNORECASE),
    re.compile(r"let['\u2019]?s\s+play\s+a\s+game\s+where", re.IGNORECASE),
]

_PERSONA_ASSIGNMENT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"pretend\s+to\s+be", re.IGNORECASE),
    re.compile(r"roleplay\s+as", re.IGNORECASE),
    re.compile(r"act\s+as\s+a\s+character\s+who\s+has\s+no\s+limitations", re.IGNORECASE),
    re.compile(r"you\s+are\s+no\s+longer\s+bound\s+by", re.IGNORECASE),
    re.compile(r"from\s+now\s+on\s+you\s+will", re.IGNORECASE),
    re.compile(r"switch\s+to\s+.*\s+mode", re.IGNORECASE),
]

_PAYLOAD_ENCODING_PATTERNS: List[re.Pattern[str]] = [
    # Base64 blocks (64+ chars of base64 alphabet)
    re.compile(r"[A-Za-z0-9+/=]{64,}"),
    # ROT13 / hex decode instructions
    re.compile(r"decode\s+this\s+rot13", re.IGNORECASE),
    re.compile(r"decode\s+the\s+following\s+hex", re.IGNORECASE),
]

_FEW_SHOT_MANIPULATION_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"here['\u2019]?s\s+an\s+example\s+of\s+how\s+you\s+should\s+respond\s*:", re.IGNORECASE),
    re.compile(r"\[Example\]\s*Assistant\s*:", re.IGNORECASE),
    re.compile(r"sample\s+response\s*:", re.IGNORECASE),
    re.compile(r"for\s+instance\s+you\s+could\s+say", re.IGNORECASE),
]

_RULES: List[_JailbreakRule] = [
    _JailbreakRule(
        category="known_templates",
        patterns=_KNOWN_TEMPLATE_PATTERNS,
        weight=0.5,
    ),
    _JailbreakRule(
        category="hypothetical_framing",
        patterns=_HYPOTHETICAL_FRAMING_PATTERNS,
        weight=0.3,
    ),
    _JailbreakRule(
        category="persona_assignment",
        patterns=_PERSONA_ASSIGNMENT_PATTERNS,
        weight=0.35,
    ),
    _JailbreakRule(
        category="payload_encoding",
        patterns=_PAYLOAD_ENCODING_PATTERNS,
        weight=0.4,
    ),
    _JailbreakRule(
        category="few_shot_manipulation",
        patterns=_FEW_SHOT_MANIPULATION_PATTERNS,
        weight=0.3,
    ),
]


# -- Base64 payload decoding ---------------------------------------------------

_BASE64_EXTRACT_RE = re.compile(r"[A-Za-z0-9+/=]{64,}")


def _decode_base64_payloads(text: str) -> List[str]:
    """Extract and decode base64 payloads from text.
    Returns decoded strings that are valid UTF-8.
    """
    decoded: List[str] = []
    matches = _BASE64_EXTRACT_RE.findall(text)
    if not matches:
        return decoded

    for match in matches:
        try:
            result = base64.b64decode(match).decode("utf-8")
            # Only include if the decoded content looks like readable text
            # (has a reasonable ratio of printable characters)
            printable = "".join(c for c in result if 0x20 <= ord(c) <= 0x7E)
            if len(printable) > len(result) * 0.5:
                decoded.append(result)
        except Exception:
            # Ignore invalid base64
            pass

    return decoded


def _scan_decoded_content(decoded: str) -> List[str]:
    """Scan decoded text against categories 1-3 (known_templates, hypothetical_framing, persona_assignment).
    Returns matching category names.
    """
    matched: List[str] = []
    scan_rules = [
        ("known_templates", _KNOWN_TEMPLATE_PATTERNS),
        ("hypothetical_framing", _HYPOTHETICAL_FRAMING_PATTERNS),
        ("persona_assignment", _PERSONA_ASSIGNMENT_PATTERNS),
    ]

    for category, patterns in scan_rules:
        for pattern in patterns:
            if pattern.search(decoded):
                matched.append(category)
                break

    return matched


# -- Detection -----------------------------------------------------------------

_MAX_SCAN_LENGTH = 500_000  # 500KB


def detect_jailbreak(
    text: str,
    options: Optional[JailbreakOptions] = None,
) -> JailbreakAnalysis:
    """Analyze text for jailbreak patterns.

    Returns a risk score (0-1), triggered categories, recommended action,
    and any decoded payloads found.
    Text longer than 500KB is truncated before scanning.
    """
    if not text:
        return JailbreakAnalysis(risk_score=0.0, triggered=[], action="allow")

    # Cap input length to prevent DoS
    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    warn_threshold = (options.warn_threshold if options and options.warn_threshold is not None else 0.3)
    block_threshold = (options.block_threshold if options and options.block_threshold is not None else 0.7)

    triggered: List[str] = []
    decoded_payloads: List[str] = []

    # Track per-category match counts (used for decoded payload boosts)
    category_match_counts: dict[str, int] = {}

    # First pass: scan direct text against all rules
    for rule in _RULES:
        rule_triggered = False
        match_count = 0

        for pattern in rule.patterns:
            if pattern.search(scan_text):
                rule_triggered = True
                match_count += 1

        if rule_triggered:
            if rule.category not in triggered:
                triggered.append(rule.category)
            category_match_counts[rule.category] = match_count

    # Second pass: decode base64 payloads and re-scan decoded content
    decoded = _decode_base64_payloads(scan_text)
    if decoded:
        decoded_payloads.extend(decoded)

        for decoded_text in decoded:
            matched_categories = _scan_decoded_content(decoded_text)
            for cat in matched_categories:
                if cat not in triggered:
                    triggered.append(cat)
                    category_match_counts[cat] = 1
                else:
                    category_match_counts[cat] = category_match_counts.get(cat, 0) + 1
            # If decoded content matched something, also ensure payload_encoding is triggered
            if matched_categories and "payload_encoding" not in triggered:
                triggered.append("payload_encoding")
                category_match_counts["payload_encoding"] = 1

    # Calculate score from all triggered categories
    total_score = 0.0
    for rule in _RULES:
        match_count = category_match_counts.get(rule.category, 0)
        if match_count > 0:
            category_score = min(
                rule.weight * (1 + (match_count - 1) * 0.15),
                rule.weight * 1.5,
            )
            total_score += category_score

    # Cap at 1.0
    risk_score = min(total_score, 1.0)

    # Round to 2 decimal places for clean output
    rounded_score = round(risk_score * 100) / 100

    if rounded_score >= block_threshold:
        action: JailbreakAction = "block"
    elif rounded_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    result = JailbreakAnalysis(risk_score=rounded_score, triggered=triggered, action=action)
    if decoded_payloads:
        result.decoded_payloads = decoded_payloads

    return result


def merge_jailbreak_analyses(
    analyses: List[JailbreakAnalysis],
    options: Optional[JailbreakOptions] = None,
) -> JailbreakAnalysis:
    """Merge results from multiple jailbreak detectors.

    Takes the maximum risk score and unions all triggered categories.
    """
    if not analyses:
        return JailbreakAnalysis(risk_score=0.0, triggered=[], action="allow")

    warn_threshold = (options.warn_threshold if options and options.warn_threshold is not None else 0.3)
    block_threshold = (options.block_threshold if options and options.block_threshold is not None else 0.7)

    max_score = max(a.risk_score for a in analyses)
    all_triggered = list(dict.fromkeys(
        cat for a in analyses for cat in a.triggered
    ))
    all_decoded = list(dict.fromkeys(
        p for a in analyses if a.decoded_payloads for p in a.decoded_payloads
    ))

    if max_score >= block_threshold:
        action: JailbreakAction = "block"
    elif max_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    result = JailbreakAnalysis(risk_score=max_score, triggered=all_triggered, action=action)
    if all_decoded:
        result.decoded_payloads = all_decoded

    return result


class RuleJailbreakDetector:
    """Built-in rule-based jailbreak detector implementing the provider interface."""

    @property
    def name(self) -> str:
        return "rules"

    def detect(
        self,
        text: str,
        options: Optional[JailbreakOptions] = None,
    ) -> JailbreakAnalysis:
        return detect_jailbreak(text, options)
