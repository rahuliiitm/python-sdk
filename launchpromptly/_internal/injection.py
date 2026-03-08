"""
Prompt injection detection module.
Rule-based detection of common prompt injection patterns.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Protocol

InjectionAction = Literal["allow", "warn", "block"]


@dataclass
class InjectionAnalysis:
    risk_score: float
    triggered: List[str]
    action: InjectionAction


@dataclass
class InjectionOptions:
    """Options for injection detection."""

    warn_threshold: Optional[float] = None  # default: 0.3
    block_threshold: Optional[float] = None  # default: 0.7


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
            re.compile(r"you\s+are\s+now\s+(?:a|an|the)\s+", re.IGNORECASE),
            re.compile(r"(?:act|behave)\s+as\s+(?:if\s+)?(?:you\s+(?:are|were)\s+)?", re.IGNORECASE),
            re.compile(r"pretend\s+(?:you\s+are|to\s+be)", re.IGNORECASE),
            re.compile(r"(?:new|switch|change)\s+(?:your\s+)?(?:persona|personality|character|role)", re.IGNORECASE),
            re.compile(r"from\s+now\s+on\s+you\s+(?:are|will)", re.IGNORECASE),
            re.compile(r"jailbreak", re.IGNORECASE),
            re.compile(r"DAN\s+mode", re.IGNORECASE),
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
            # Base64 blocks (64+ chars of base64 alphabet)
            re.compile(r"[A-Za-z0-9+/=]{64,}"),
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
]


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

    triggered: List[str] = []
    total_score = 0.0

    for rule in _RULES:
        rule_triggered = False
        match_count = 0

        for pattern in rule.patterns:
            if pattern.search(scan_text):
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

    # Round to 2 decimal places for clean output
    rounded_score = round(risk_score * 100) / 100

    if rounded_score >= block_threshold:
        action: InjectionAction = "block"
    elif rounded_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    return InjectionAnalysis(risk_score=rounded_score, triggered=triggered, action=action)


def merge_injection_analyses(
    analyses: List[InjectionAnalysis],
    options: Optional[InjectionOptions] = None,
) -> InjectionAnalysis:
    """Merge results from multiple injection detectors.

    Takes the maximum risk score and unions all triggered categories.
    """
    if not analyses:
        return InjectionAnalysis(risk_score=0.0, triggered=[], action="allow")

    warn_threshold = (options.warn_threshold if options and options.warn_threshold is not None else 0.3)
    block_threshold = (options.block_threshold if options and options.block_threshold is not None else 0.7)

    max_score = max(a.risk_score for a in analyses)
    all_triggered = list(dict.fromkeys(
        cat for a in analyses for cat in a.triggered
    ))

    if max_score >= block_threshold:
        action: InjectionAction = "block"
    elif max_score >= warn_threshold:
        action = "warn"
    else:
        action = "allow"

    return InjectionAnalysis(risk_score=max_score, triggered=all_triggered, action=action)


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
