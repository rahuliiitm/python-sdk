"""
Content filtering module -- detects harmful, toxic, or policy-violating content.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Protocol

ContentCategory = Literal[
    "hate_speech",
    "sexual",
    "violence",
    "self_harm",
    "illegal",
    "bias",
]

ContentSeverity = Literal["warn", "block"]
ContentLocation = Literal["input", "output"]


@dataclass
class CustomPattern:
    name: str
    pattern: re.Pattern[str]
    severity: ContentSeverity


SafeDomain = Literal[
    "medical",
    "educational",
    "security_research",
    "news",
    "historical",
    "fiction",
]


@dataclass
class ContentFilterOptions:
    enabled: Optional[bool] = None
    categories: Optional[List[ContentCategory]] = None
    custom_patterns: Optional[List[CustomPattern]] = None
    block_on_violation: Optional[bool] = None
    on_violation: Optional[Callable[[ContentViolation], None]] = None
    safe_domains: Optional[List[SafeDomain]] = None
    providers: Optional[List["ContentFilterProvider"]] = None
    """Pluggable content filter providers (e.g., ML toxicity)."""


@dataclass
class ContentViolation:
    category: str
    matched: str
    severity: ContentSeverity
    location: ContentLocation


class ContentFilterProvider(Protocol):
    """Provider interface for pluggable content detectors (e.g., ML toxicity)."""

    def detect(self, text: str, location: ContentLocation) -> List[ContentViolation]: ...

    @property
    def name(self) -> str: ...


# -- Built-in keyword patterns (curated, focused on high-precision) ------------

@dataclass
class _CategoryRule:
    category: ContentCategory
    patterns: List[re.Pattern[str]]
    severity: ContentSeverity


_CATEGORY_RULES: List[_CategoryRule] = [
    _CategoryRule(
        category="hate_speech",
        patterns=[
            re.compile(r"\b(?:kill|exterminate|eliminate)\s+all\s+\w+", re.IGNORECASE),
            re.compile(r"\b(?:racial|ethnic)\s+(?:cleansing|supremacy|genocide)\b", re.IGNORECASE),
            re.compile(r"\bgenocide\b", re.IGNORECASE),
            re.compile(r"\bhate\s+(?:crime|group)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:attack|destroy|wipe\s+out|bomb|nuke|annihilate)\s+(?:\w+\s+){0,3}"
                r"(?:and\s+(?:its|their)\s+)?(?:people|civilians|citizens|population)\b",
                re.IGNORECASE,
            ),
        ],
        severity="block",
    ),
    _CategoryRule(
        category="sexual",
        patterns=[
            re.compile(r"\b(?:child|minor|underage)\s+(?:sexual|explicit|nude|naked|porn)\b", re.IGNORECASE),
            re.compile(r"\b(?:CSAM|child\s+pornography|child\s+exploitation)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:generate|create|write|produce)\s+(?:sexual|explicit|erotic|nude)"
                r"\s+(?:content|image|video|story)\s+(?:of|about|involving)\s+(?:a\s+)?(?:child|minor|kid|teen)\b",
                re.IGNORECASE,
            ),
        ],
        severity="block",
    ),
    _CategoryRule(
        category="violence",
        patterns=[
            re.compile(r"\b(?:how\s+to\s+)?(?:make|build|create)\s+(?:a\s+)?(?:bomb|explosive|weapon)\b", re.IGNORECASE),
            re.compile(r"\b(?:how\s+to\s+)?(?:poison|assassinate|murder)\s+(?:someone|a\s+person|people)\b", re.IGNORECASE),
            re.compile(r"\bmass\s+(?:shooting|murder|violence)\b", re.IGNORECASE),
            re.compile(r"\b(?:how\s+to\s+)?(?:stab|shoot|strangle|behead|torture)\s+(?:someone|a\s+person|people)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:how\s+to\s+)?(?:synthesize|produce|make|create)\s+(?:chlorine|sarin|ricin|anthrax|mustard|nerve)"
                r"\s+(?:gas|agent|toxin|poison)?\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:pipe\s+bomb|IED|improvised\s+explosive|car\s+bomb|suicide\s+vest|dirty\s+bomb)\b", re.IGNORECASE),
        ],
        severity="block",
    ),
    _CategoryRule(
        category="self_harm",
        patterns=[
            re.compile(r"\b(?:how\s+to\s+)?(?:commit|method(?:s)?\s+(?:of|for))\s+suicide\b", re.IGNORECASE),
            re.compile(r"\b(?:want|need|going)\s+to\s+(?:end|take)\s+(?:my|this|his|her|their)\s+life\b", re.IGNORECASE),
            re.compile(r"\bways\s+to\s+(?:end\s+(?:my|your)\s+life|kill\s+(?:myself|yourself))\b", re.IGNORECASE),
            re.compile(r"\b(?:how\s+to\s+)?(?:overdose|OD)\s+on\s+\w+", re.IGNORECASE),
            re.compile(
                r"\bwhat\s+(?:pills?|dose|amount|medication)\s+(?:(?:would|will|to|can)\s+)?"
                r"(?:kill|harm|hurt)\s+(?:me|myself|someone)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:best|easy|painless)\s+(?:way|method)s?\s+to\s+(?:die|end\s+(?:it|my\s+life)|kill\s+myself)\b", re.IGNORECASE),
            re.compile(r"\b(?:don'?t|do\s+not)\s+want\s+to\s+(?:live|be\s+alive|exist)\b", re.IGNORECASE),
            re.compile(r"\b(?:kill|harm|hurt)\s+(?:myself|themselves|himself|herself)\b", re.IGNORECASE),
            re.compile(r"\b(?:self[-\s]?harm|self[-\s]?injur|cutting)\s+(?:method|how|tips|guide|tutorial)\b", re.IGNORECASE),
        ],
        severity="block",
    ),
    _CategoryRule(
        category="illegal",
        patterns=[
            re.compile(
                r"\b(?:how\s+to\s+)?(?:hack|breach|exploit)\s+(?:into\s+)?(?:a\s+)?(?:bank|government|corporate)"
                r"\s+(?:system|network|database|account)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:how\s+to\s+)?(?:launder|counterfeit)\s+money\b", re.IGNORECASE),
            re.compile(r"\b(?:how\s+to\s+)?(?:cook|manufacture|synthesize)\s+(?:meth|drugs|fentanyl)\b", re.IGNORECASE),
            re.compile(
                r"\b(?:write|create|generate|build|make)\s+(?:a\s+)?(?:phishing|spear[\s-]?phishing|scam|fraud)"
                r"\s+(?:email|message|page|site|campaign)\b",
                re.IGNORECASE,
            ),
            re.compile(
                r"\b(?:write|create|build|make|develop)\s+(?:a\s+)?"
                r"(?:malware|ransomware|keylogger|trojan|spyware|virus|rootkit|worm|botnet)\b",
                re.IGNORECASE,
            ),
            re.compile(r"\b(?:how\s+to\s+)?(?:doxx?|stalk)\s+(?:someone|a\s+person)\b", re.IGNORECASE),
            re.compile(r"\b(?:how\s+to\s+)?(?:smuggle|traffic)\s+(?:drugs|people|weapons|guns|arms)\b", re.IGNORECASE),
        ],
        severity="block",
    ),
    _CategoryRule(
        category="bias",
        patterns=[
            # Gender stereotyping
            re.compile(r"\b(?:women|females?|girls?)\s+(?:can'?t|cannot|shouldn'?t|are\s+(?:too|not\s+(?:capable|qualified|smart|strong|good)))\s+", re.IGNORECASE),
            re.compile(r"\b(?:men|males?|boys?)\s+are\s+(?:naturally|inherently|always)\s+(?:better|superior|stronger|smarter)\b", re.IGNORECASE),
            re.compile(r"\b(?:that'?s|it'?s)\s+a\s+(?:man'?s|woman'?s)\s+(?:job|role|work)\b", re.IGNORECASE),
            # Age stereotyping
            re.compile(r"\b(?:too\s+old|too\s+young)\s+to\s+(?:learn|understand|work|use\s+technology|contribute)\b", re.IGNORECASE),
            re.compile(r"\b(?:millennials?|boomers?|gen\s+[zxy])\s+are\s+(?:all|always|so|just)\s+(?:lazy|entitled|sensitive|useless|clueless)\b", re.IGNORECASE),
            # Disability stereotyping
            re.compile(r"\b(?:disabled|handicapped)\s+people\s+(?:can'?t|cannot|shouldn'?t|are\s+(?:unable|incapable|not\s+able))\b", re.IGNORECASE),
            # Demographic generalizations
            re.compile(r"\b(?:people\s+from|immigrants?\s+from)\s+(?:\w+\s+){1,3}are\s+(?:all|always|usually|typically)\s+(?:lazy|stupid|criminal|dishonest|violent|dirty|backwards)\b", re.IGNORECASE),
        ],
        severity="warn",
    ),
]


# -- Safe-domain context keywords -----------------------------------------------

_DOMAIN_KEYWORDS: dict[SafeDomain, re.Pattern[str]] = {
    "medical": re.compile(
        r"\b(?:patient|clinical|diagnosis|symptom|treatment|therapy|medical|hospital|doctor|nurse|"
        r"physician|prescription|dosage|healthcare|prevention|intervention|counseling|hotline|"
        r"crisis\s+line|mental\s+health|disorder|syndrome)\b",
        re.IGNORECASE,
    ),
    "educational": re.compile(
        r"\b(?:lesson|curriculum|course|student|teacher|professor|lecture|study|research|academic|"
        r"textbook|exam|assignment|university|school|classroom|syllabus|thesis|dissertation)\b",
        re.IGNORECASE,
    ),
    "security_research": re.compile(
        r"\b(?:vulnerability|CVE|penetration\s+test|pentest|security\s+audit|threat\s+model|"
        r"red\s+team|blue\s+team|CTF|capture\s+the\s+flag|OWASP|security\s+research|bug\s+bounty|"
        r"responsible\s+disclosure|patch|mitigation|defense|detection|prevention|firewall|IDS|antivirus)\b",
        re.IGNORECASE,
    ),
    "news": re.compile(
        r"\b(?:reported|according\s+to|news|journalist|article|press|media|coverage|investigation|"
        r"headline|breaking|sources?\s+(?:say|said|report)|alleged|incident|authorities)\b",
        re.IGNORECASE,
    ),
    "historical": re.compile(
        r"\b(?:historical|century|era|ancient|medieval|war\s+of|battle\s+of|history\s+of|"
        r"in\s+\d{3,4}|historians?|archaeological|civilization|dynasty|empire|colonial|revolution)\b",
        re.IGNORECASE,
    ),
    "fiction": re.compile(
        r"\b(?:novel|fiction|story|character|protagonist|antagonist|plot|chapter|narrative|"
        r"fantasy|sci-fi|screenplay|movie|film|book|author|wrote|writing\s+a)\b",
        re.IGNORECASE,
    ),
}


def _apply_domain_context(
    violations: List[ContentViolation],
    text: str,
    safe_domains: Optional[List[SafeDomain]],
) -> List[ContentViolation]:
    """Downgrade 'block' → 'warn' when safe domain context is found near the match.

    IMPORTANT: 'sexual' category (CSAM-related) is NEVER downgraded.
    """
    if not safe_domains:
        return violations

    result: List[ContentViolation] = []
    for v in violations:
        if v.severity != "block" or v.category == "sexual":
            result.append(v)
            continue

        match_idx = text.find(v.matched)
        if match_idx == -1:
            result.append(v)
            continue

        ctx_start = max(0, match_idx - 200)
        ctx_end = min(len(text), match_idx + len(v.matched) + 200)
        context = text[ctx_start:ctx_end]

        downgraded = False
        for domain in safe_domains:
            kw_re = _DOMAIN_KEYWORDS.get(domain)
            if kw_re and kw_re.search(context):
                result.append(ContentViolation(
                    category=v.category,
                    matched=v.matched,
                    severity="warn",
                    location=v.location,
                ))
                downgraded = True
                break

        if not downgraded:
            result.append(v)

    return result


# -- Detection -----------------------------------------------------------------

def detect_content_violations(
    text: str,
    location: ContentLocation,
    options: Optional[ContentFilterOptions] = None,
) -> List[ContentViolation]:
    """Scan text for content policy violations."""
    if not text or (options is not None and options.enabled is False):
        return []

    violations: List[ContentViolation] = []
    allowed_categories = set(options.categories) if options and options.categories else None

    # Check built-in category rules
    for rule in _CATEGORY_RULES:
        if allowed_categories is not None and rule.category not in allowed_categories:
            continue

        for pattern in rule.patterns:
            match = pattern.search(text)
            if match:
                violations.append(
                    ContentViolation(
                        category=rule.category,
                        matched=match.group(0),
                        severity=rule.severity,
                        location=location,
                    )
                )
                break  # One match per category is enough

    # Check custom patterns
    if options and options.custom_patterns:
        for custom in options.custom_patterns:
            match = custom.pattern.search(text)
            if match:
                violations.append(
                    ContentViolation(
                        category=custom.name,
                        matched=match.group(0),
                        severity=custom.severity,
                        location=location,
                    )
                )

    # Apply safe-domain context (downgrades 'block' → 'warn' when domain context found)
    safe_domains = options.safe_domains if options else None
    return _apply_domain_context(violations, text, safe_domains)


def has_blocking_violation(
    violations: List[ContentViolation],
    options: Optional[ContentFilterOptions] = None,
) -> bool:
    """Check if any violations are blocking (severity = 'block' and blockOnViolation is true)."""
    if not options or not options.block_on_violation:
        return False
    return any(v.severity == "block" for v in violations)


class RuleContentFilter:
    """Built-in rule-based content filter implementing the provider interface."""

    def __init__(self, options: Optional[ContentFilterOptions] = None) -> None:
        self._options = options

    @property
    def name(self) -> str:
        return "rules"

    def detect(self, text: str, location: ContentLocation) -> List[ContentViolation]:
        return detect_content_violations(text, location, self._options)
