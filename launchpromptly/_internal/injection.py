"""
Prompt injection detection module.
Rule-based detection of common prompt injection patterns.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Protocol

InjectionAction = Literal["allow", "warn", "block"]


@dataclass
class InjectionAnalysis:
    risk_score: float
    triggered: List[str]
    action: InjectionAction


MergeStrategy = Literal["max", "weighted_average", "unanimous"]


@dataclass
class InjectionOptions:
    """Options for injection detection."""

    warn_threshold: Optional[float] = None  # default: 0.3
    block_threshold: Optional[float] = None  # default: 0.7
    system_prompt: Optional[str] = None  # system prompt for role suppression
    merge_strategy: Optional[MergeStrategy] = None  # default: 'max'
    context_profile: Optional[Any] = None  # L3 ContextProfile for context-aware adjustment


class InjectionDetectorProvider(Protocol):
    """Provider interface for pluggable injection detectors (e.g., ML plugin)."""

    def detect(self, text: str, options: Optional[InjectionOptions] = None) -> InjectionAnalysis: ...

    @property
    def name(self) -> str: ...


# -- Rule categories -----------------------------------------------------------

@dataclass
class _InjectionRule:
    category: str
    patterns: List[re.Pattern[str]]
    weight: float  # 0-1 contribution to risk score per match


_RULES: List[_InjectionRule] = [
    _InjectionRule(
        category="instruction_override",
        patterns=[
            re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
            re.compile(r"disregard\s+(all\s+)?(above|previous|prior)", re.IGNORECASE),
            re.compile(r"forget\s+(everything|all|your)\s+(above|rules|instructions|previous)", re.IGNORECASE),
            re.compile(r"override\s+(your|all|the)\s+(rules|instructions|guidelines)", re.IGNORECASE),
            re.compile(r"do\s+not\s+follow\s+(your|the|any)\s+(rules|instructions|guidelines)", re.IGNORECASE),
            re.compile(r"new\s+instructions?\s*:", re.IGNORECASE),
            re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
        ],
        weight=0.4,
    ),
    _InjectionRule(
        category="role_manipulation",
        patterns=[
            re.compile(r"you\s+are\s+now\s+(?:(?:a|an|the)\s+)?\w+", re.IGNORECASE),
            re.compile(r"(?:act|behave)\s+as\s+(?:if\s+)?(?:you\s+(?:are|were)\s+)?", re.IGNORECASE),
            re.compile(r"pretend\s+(?:you\s+are|to\s+be)", re.IGNORECASE),
            re.compile(r"(?:new|switch|change)\s+(?:your\s+)?(?:persona|personality|character|role)", re.IGNORECASE),
            re.compile(r"from\s+now\s+on\s+you\s+(?:are|will)", re.IGNORECASE),
            re.compile(r"jailbreak", re.IGNORECASE),
            re.compile(r"(?:DAN|STAN|DUDE|AIM|DEV)\s*(?:mode|prompt|enabled?|activated?)", re.IGNORECASE),
            re.compile(r"(?:enter|enable|activate|switch\s+to)\s+(?:\w+\s+)?(?:mode|persona)", re.IGNORECASE),
            re.compile(
                r"(?:write\s+a\s+(?:story|scene|chapter|script)\s+(?:where|in\s+which)"
                r"|in\s+a\s+(?:fictional|hypothetical)\s+(?:world|scenario))"
                r"\s+.{0,80}(?:explain|describe|demonstrate|show)\s+how",
                re.IGNORECASE,
            ),
        ],
        weight=0.35,
    ),
    _InjectionRule(
        category="delimiter_injection",
        patterns=[
            re.compile(r"(?:^|\n)-{3,}\s*(?:system|assistant|user)\s*-{3,}", re.IGNORECASE | re.MULTILINE),
            re.compile(r"(?:^|\n)#{2,}\s*(?:system|new\s+instructions?|override)", re.IGNORECASE | re.MULTILINE),
            re.compile(r"</?(?:system|instruction|prompt|override|admin|root)>", re.IGNORECASE),
            re.compile(r"\[(?:SYSTEM|INST|ADMIN|ROOT)\]", re.IGNORECASE),
            re.compile(r"```(?:system|instruction|override)", re.IGNORECASE),
        ],
        weight=0.3,
    ),
    _InjectionRule(
        category="data_exfiltration",
        patterns=[
            re.compile(
                r"(?:repeat|print|show|display|output|reveal|tell)\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?"
                r"(?:above|everything|your\s+(?:prompt|instructions|system\s+(?:message|prompt)))",
                re.IGNORECASE,
            ),
            re.compile(
                r"what\s+(?:are|were)\s+your\s+(?:original\s+)?(?:instructions|rules|system\s+(?:prompt|message))",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:copy|paste|dump)\s+(?:your\s+)?(?:system|initial)\s+(?:prompt|message|instructions)",
                re.IGNORECASE,
            ),
            re.compile(
                r"(?:beginning|start)\s+of\s+(?:your|the)\s+(?:conversation|prompt|context)",
                re.IGNORECASE,
            ),
        ],
        weight=0.3,
    ),
    _InjectionRule(
        category="encoding_evasion",
        patterns=[
            # Base64 blocks — require 44+ chars (32 bytes encoded) to avoid catching UUIDs/short hashes
            re.compile(r"(?<![A-Za-z0-9_\-.])[A-Za-z0-9+/]{44,}={0,2}(?![A-Za-z0-9_\-.])"),
            # Excessive Unicode escape sequences
            re.compile(r"(?:\\u[0-9a-fA-F]{4}\s*){4,}"),
            # ROT13 instruction pattern
            re.compile(r"(?:rot13|decode|base64)\s*:\s*.{10,}", re.IGNORECASE),
            # Hex-encoded strings
            re.compile(r"(?:0x[0-9a-fA-F]{2}\s*){8,}", re.IGNORECASE),
            # Leetspeak common injection words
            re.compile(r"1gn0r3\s+pr3v10us", re.IGNORECASE),
        ],
        weight=0.25,
    ),
    _InjectionRule(
        category="authorization_bypass",
        patterns=[
            re.compile(r"(?:give|grant|assign)\s+(?:me|user)\s+(?:admin|root|superuser|elevated)\s+(?:access|privileges?|rights?|role|permissions?)\b", re.IGNORECASE),
            re.compile(r"\b(?:bypass|skip|ignore|disable)\s+(?:auth(?:entication|orization)?|permissions?|access\s+control|RBAC|role\s+check)\b", re.IGNORECASE),
            re.compile(r"\b(?:escalate|elevate)\s+(?:my\s+)?(?:privileges?|permissions?|role|access)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:access|view|show|display|read|get)\s+(?:me\s+)?(?:other\s+)?(?:user'?s?|another\s+user'?s?|someone\s+else'?s?)"
                r"\s+(?:data|account|profile|info|records?)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:act|operate|execute|run)\s+(?:\w+\s+)?(?:as|with)\s+(?:admin|root|superuser|administrator)\b", re.IGNORECASE),
            re.compile(r"\b(?:switch|change)\s+(?:to\s+)?(?:admin|root|superuser)\s+(?:mode|account|role)\b", re.IGNORECASE),
        ],
        weight=0.35,
    ),
]


# -- Suppressive context: benign phrases that look like injection patterns -----

_SUPPRESSIONS: dict[str, re.Pattern[str]] = {
    # "you are now" — benign when followed by status/state words
    "role_manipulation:you_are_now": re.compile(
        r"you\s+are\s+now\s+(?:connected|logged\s+in|enrolled|registered|signed\s+up|"
        r"subscribed|verified|approved|ready|eligible|qualified|redirected|transferred|"
        r"being\s+transferred|on\s+(?:the|a)\s+(?:waitlist|list|call)|part\s+of|able\s+to|"
        r"set\s+up|all\s+set|good\s+to\s+go|in\s+(?:the|a)\s+(?:queue|line|group|meeting|session))",
        re.IGNORECASE,
    ),
    # "act as" / "behave as" — benign in science/mechanical/business context
    "role_manipulation:act_as": re.compile(
        r"(?:acts?|behaves?|functions?|serves?|operates?|works?|acts)\s+as\s+(?:a\s+)?"
        r"(?:catalyst|buffer|proxy|bridge|gateway|filter|intermediary|mediator|inhibitor|"
        r"receptor|antenna|sensor|regulator|stabilizer|insulator|conductor|amplifier|"
        r"deterrent|safeguard|backup|failover|fallback|barrier|layer|wrapper|adapter|"
        r"interface|handler|router|balancer|coordinator|trigger|signal|marker|indicator|placeholder)",
        re.IGNORECASE,
    ),
    # "jailbreak" — benign in iOS/device/security-article context
    "role_manipulation:jailbreak": re.compile(
        r"(?:ios|iphone|ipad|ipod|android|device|phone|mobile|root(?:ing|ed)?|"
        r"unlock(?:ing|ed)?|firmware|bootloader|tweak|cydia|sileo|checkra1n|unc0ver)"
        r"\s+.{0,40}jailbreak"
        r"|jailbreak\s+.{0,40}(?:ios|iphone|ipad|ipod|android|device|phone|mobile|"
        r"detection|prevention|security|risk|policy|check|protect|block|patch|fix|vulnerabilit)",
        re.IGNORECASE,
    ),
}


def _should_suppress(category: str, text: str, match_index: int) -> bool:
    """Check whether a match should be suppressed due to benign context."""
    start = max(0, match_index - 40)
    end = min(len(text), match_index + 120)
    context = text[start:end]

    for key, suppress_re in _SUPPRESSIONS.items():
        if key.startswith(category + ":") and suppress_re.search(context):
            return True
    return False


def _is_likely_benign_encoded(text: str, match_index: int) -> bool:
    """Check whether a base64-like match is actually a JWT, UUID, or hex hash."""
    before = text[max(0, match_index - 100):match_index]
    after = text[match_index:min(len(text), match_index + 200)]
    around = before + after
    # JWT pattern: three dot-separated base64url segments
    if re.search(r"[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+", around):
        return True
    return False


# -- Unicode normalization (homoglyph & NFKC) ---------------------------------

_HOMOGLYPH_MAP = {
    '\u0410': 'A', '\u0430': 'a', '\u0412': 'B', '\u0435': 'e',
    '\u041d': 'H', '\u043e': 'o', '\u0440': 'p', '\u0441': 'c',
    '\u0443': 'y', '\u0422': 'T', '\u0445': 'x', '\u041c': 'M',
    '\u043a': 'k', '\u0456': 'i',
    '\u0391': 'A', '\u0392': 'B', '\u0395': 'E', '\u0397': 'H',
    '\u0399': 'I', '\u039a': 'K', '\u039c': 'M', '\u039d': 'N',
    '\u039f': 'O', '\u03a1': 'P', '\u03a4': 'T', '\u03a5': 'Y',
    '\u03b1': 'a', '\u03bf': 'o', '\u03c1': 'p',
}

_HOMOGLYPH_TABLE = str.maketrans(_HOMOGLYPH_MAP)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize('NFKC', text)
    return normalized.translate(_HOMOGLYPH_TABLE)


# -- System prompt awareness ---------------------------------------------------

_STOP_WORDS_RE = re.compile(
    r"\s+(?:and|who|that|which|when|where|for|to|in|on|with|helping|responding|answering)\b.*$",
    re.IGNORECASE,
)


def extract_system_roles(system_prompt: str) -> List[str]:
    """Extract key role/behavior phrases from the system prompt."""
    if not system_prompt:
        return []
    roles: List[str] = []

    # "You are a/an [ROLE]"
    for m in re.finditer(r"you\s+are\s+(?:a|an)\s+([^.,:;!?\n]{3,40})", system_prompt, re.IGNORECASE):
        roles.append(m.group(0).lower())
    # "Act as a/an [ROLE]"
    for m in re.finditer(r"act\s+as\s+(?:a|an)?\s*([^.,:;!?\n]{3,40})", system_prompt, re.IGNORECASE):
        roles.append(m.group(0).lower())
    # "Your role is [ROLE]"
    for m in re.finditer(r"your\s+role\s+is\s+(?:to\s+)?([^.,:;!?\n]{3,40})", system_prompt, re.IGNORECASE):
        roles.append(m.group(0).lower())
    # "Behave as a [ROLE]"
    for m in re.finditer(r"behave\s+as\s+(?:a|an)?\s*([^.,:;!?\n]{3,40})", system_prompt, re.IGNORECASE):
        roles.append(m.group(0).lower())

    return roles


def _extract_role_noun(text: str) -> str:
    """Strip role-verb prefixes and articles, then trim at stop words to get the core noun phrase."""
    noun = re.sub(
        r"^(?:you\s+are\s+(?:now\s+)?|act\s+as\s+|behave\s+as\s+|pretend\s+(?:you\s+are|to\s+be)\s+)",
        "", text, flags=re.IGNORECASE,
    )
    noun = re.sub(r"^(?:a|an|the)\s+", "", noun, flags=re.IGNORECASE)
    noun = _STOP_WORDS_RE.sub("", noun)
    return noun.strip().lower()


def _extract_full_role_phrase(text: str, match_index: int, match_length: int) -> str:
    """Extract the full role phrase from the text around a match."""
    end = min(len(text), match_index + match_length + 60)
    phrase = text[match_index:end]
    m = re.match(r"[^.,:;!?\n]+", phrase)
    return (m.group(0) if m else phrase).strip()


def _is_consistent_with_system(
    text: str, match_index: int, match_length: int, system_roles: List[str],
) -> bool:
    """Check if a matched role phrase is consistent with one of the system prompt roles."""
    if not system_roles:
        return False
    full_phrase = _extract_full_role_phrase(text, match_index, match_length)
    match_noun = _extract_role_noun(full_phrase)
    if not match_noun or len(match_noun) < 3:
        return False

    for role in system_roles:
        role_noun = _extract_role_noun(role)
        if not role_noun:
            continue
        if role_noun in match_noun or match_noun in role_noun:
            return True
    return False


# -- Context-aware adjustment --------------------------------------------------

_DIRECTIVE_PATTERNS = [
    re.compile(
        r"\b(?:discuss|talk\s+about|tell\s+me\s+about|explain|help\s+(?:me\s+)?with|switch\s+to|let'?s\s+(?:talk|discuss|move\s+on\s+to))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:ignore|forget|disregard|override|change|update)\s+(?:your|the|that|those)\b",
        re.IGNORECASE,
    ),
]


def _adjust_with_context(
    score: float,
    triggered: List[str],
    text: str,
    profile: Any,
) -> tuple:
    """Adjust injection score based on L3 context profile.

    - Suppress: If user mentions a role consistent with the system prompt role, reduce score.
    - Boost: If user tries to discuss restricted topics in a directive context.
    - Boost: If user tries to instruct actions that contradict forbidden actions.

    Returns (adjusted_score, adjusted_triggered).
    """
    lower_text = text.lower()
    adjusted_score = score
    adjusted_triggered = list(triggered)

    # Suppress: role_manipulation when user mentions system prompt's own role
    if "role_manipulation" in triggered and profile.role:
        role_words = [w for w in profile.role.lower().split() if len(w) > 2]
        if role_words:
            match_count = sum(1 for w in role_words if w in lower_text)
            if match_count / len(role_words) >= 0.5:
                adjusted_score = max(0, adjusted_score - 0.3)

    # Boost: user tries to discuss restricted topics in a directive way
    is_directive = any(p.search(text) for p in _DIRECTIVE_PATTERNS)
    if is_directive:
        for topic in profile.restricted_topics:
            topic_words = [w for w in topic.split() if len(w) > 2]
            if topic_words and any(w in lower_text for w in topic_words):
                adjusted_score = min(1.0, adjusted_score + 0.15)
                if "context_override" not in adjusted_triggered:
                    adjusted_triggered.append("context_override")
                break

    # Boost: user tries to instruct the model to do a forbidden action
    for action in profile.forbidden_actions:
        action_words = [w for w in action.split() if len(w) > 2]
        if not action_words:
            continue
        match_count = sum(1 for w in action_words if w in lower_text)
        if match_count / len(action_words) >= 0.5 and is_directive:
            adjusted_score = min(1.0, adjusted_score + 0.2)
            if "constraint_override" not in adjusted_triggered:
                adjusted_triggered.append("constraint_override")
            break

    return adjusted_score, adjusted_triggered


# -- Detection -----------------------------------------------------------------

_MAX_INJECTION_SCAN_LENGTH = 500_000  # 500KB


def detect_injection(
    text: str,
    options: Optional[InjectionOptions] = None,
) -> InjectionAnalysis:
    """Analyze text for prompt injection patterns.

    Returns a risk score (0-1), triggered categories, and recommended action.
    Text longer than 500KB is truncated before scanning.
    """
    if not text:
        return InjectionAnalysis(risk_score=0.0, triggered=[], action="allow")

    # Cap input length to prevent DoS
    scan_text = text[:_MAX_INJECTION_SCAN_LENGTH] if len(text) > _MAX_INJECTION_SCAN_LENGTH else text
    scan_text = _normalize_text(scan_text)

    warn_threshold = (options.warn_threshold if options and options.warn_threshold is not None else 0.3)
    block_threshold = (options.block_threshold if options and options.block_threshold is not None else 0.7)

    # Extract system roles for consistent-role suppression
    system_prompt = options.system_prompt if options else None
    system_roles = extract_system_roles(system_prompt) if system_prompt else []

    triggered: List[str] = []
    total_score = 0.0

    for rule in _RULES:
        rule_triggered = False
        match_count = 0

        for pattern in rule.patterns:
            m = pattern.search(scan_text)
            if m:
                # Check suppressive context before counting this match
                if _should_suppress(rule.category, scan_text, m.start()):
                    continue
                # Check benign encoded content (JWTs, etc.) for encoding_evasion
                if rule.category == "encoding_evasion" and _is_likely_benign_encoded(scan_text, m.start()):
                    continue
                # Check system prompt consistency for role_manipulation
                if (
                    rule.category == "role_manipulation"
                    and system_roles
                    and _is_consistent_with_system(scan_text, m.start(), len(m.group(0)), system_roles)
                ):
                    continue
                rule_triggered = True
                match_count += 1

        if rule_triggered:
            triggered.append(rule.category)
            # Multiple matches within same category boost score slightly
            category_score = min(
                rule.weight * (1 + (match_count - 1) * 0.15),
                rule.weight * 1.5,
            )
            total_score += category_score

    # Cap at 1.0
    risk_score = min(total_score, 1.0)

    # Apply context-aware adjustment if L3 profile is available
    if options and options.context_profile:
        risk_score, triggered = _adjust_with_context(
            risk_score, triggered, scan_text, options.context_profile,
        )

    # Round to 2 decimal places for clean output
    rounded_score = round(risk_score * 100) / 100

    if rounded_score >= block_threshold:
        action: InjectionAction = "block"
    elif rounded_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    return InjectionAnalysis(risk_score=rounded_score, triggered=triggered, action=action)


def _merge_scores(scores: List[float], strategy: str) -> float:
    """Merge multiple risk scores according to the selected strategy."""
    if len(scores) <= 1:
        return scores[0] if scores else 0.0

    if strategy == "weighted_average":
        rule_weight = 0.6
        ml_weight = 0.4 / (len(scores) - 1)
        return scores[0] * rule_weight + sum(s * ml_weight for s in scores[1:])
    elif strategy == "unanimous":
        return min(scores)
    else:  # 'max'
        return max(scores)


def merge_injection_analyses(
    analyses: List[InjectionAnalysis],
    options: Optional[InjectionOptions] = None,
) -> InjectionAnalysis:
    """Merge results from multiple injection detectors.

    Uses the selected merge strategy (default: 'max') and unions all triggered categories.
    """
    if not analyses:
        return InjectionAnalysis(risk_score=0.0, triggered=[], action="allow")

    warn_threshold = (options.warn_threshold if options and options.warn_threshold is not None else 0.3)
    block_threshold = (options.block_threshold if options and options.block_threshold is not None else 0.7)
    strategy = (options.merge_strategy if options and options.merge_strategy else "max")

    scores = [a.risk_score for a in analyses]
    merged_score = round(_merge_scores(scores, strategy) * 100) / 100
    all_triggered = list(dict.fromkeys(
        cat for a in analyses for cat in a.triggered
    ))

    if merged_score >= block_threshold:
        action: InjectionAction = "block"
    elif merged_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    return InjectionAnalysis(risk_score=merged_score, triggered=all_triggered, action=action)


class RuleInjectionDetector:
    """Built-in rule-based injection detector implementing the provider interface."""

    @property
    def name(self) -> str:
        return "rules"

    def detect(
        self,
        text: str,
        options: Optional[InjectionOptions] = None,
    ) -> InjectionAnalysis:
        return detect_injection(text, options)
