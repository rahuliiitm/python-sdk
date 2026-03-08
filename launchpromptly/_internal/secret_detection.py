"""
Secret / credential detection module.
Detects API keys, tokens, private keys, connection strings, and other
credentials in text. Separate from PII (which covers personal data).
Zero-dependency, regex-based scanner.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

MAX_SCAN_LENGTH = 1024 * 1024  # 1 MB


@dataclass
class SecretDetectionOptions:
    """Options for secret detection."""

    built_in_patterns: Optional[bool] = None  # default: True
    custom_patterns: Optional[List[CustomSecretPattern]] = None


@dataclass
class CustomSecretPattern:
    """A user-supplied secret pattern."""

    name: str
    pattern: str  # Regex source string (compiled internally)
    confidence: Optional[float] = None  # default: 0.9


@dataclass
class SecretDetection:
    """A detected secret."""

    type: str  # Pattern name, or 'custom:<name>' for user-supplied patterns
    value: str  # The matched text
    start: int  # Start index in the (possibly truncated) input
    end: int  # End index in the (possibly truncated) input
    confidence: float  # Confidence score 0-1


# -- Built-in patterns ---------------------------------------------------------

@dataclass
class _PatternEntry:
    name: str
    pattern: re.Pattern[str]
    confidence: float


_PATTERNS: List[_PatternEntry] = [
    _PatternEntry("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b"), 0.95),
    _PatternEntry("github_pat", re.compile(r"\bghp_[A-Za-z0-9]{36}\b"), 0.95),
    _PatternEntry("github_oauth", re.compile(r"\bgho_[A-Za-z0-9]{36}\b"), 0.95),
    _PatternEntry("gitlab_pat", re.compile(r"\bglpat-[A-Za-z0-9\-]{20,}\b"), 0.95),
    _PatternEntry("slack_token", re.compile(r"\bxox[bpas]-[A-Za-z0-9\-]+\b"), 0.90),
    _PatternEntry("stripe_secret", re.compile(r"\bsk_live_[A-Za-z0-9]{24,}\b"), 0.95),
    _PatternEntry("stripe_publishable", re.compile(r"\bpk_live_[A-Za-z0-9]{24,}\b"), 0.85),
    _PatternEntry("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b"), 0.90),
    _PatternEntry("private_key", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"), 0.99),
    _PatternEntry("connection_string", re.compile(r"(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp)://[^\s\"']+"), 0.90),
    _PatternEntry("webhook_url", re.compile(r"https?://hooks\.(?:slack|discord)\.com/[^\s]+"), 0.85),
    _PatternEntry("generic_high_entropy", re.compile(r"(?:secret|key|token|password|api_key|apikey)[\s:=\"']*[A-Za-z0-9/+=]{32,}", re.IGNORECASE), 0.70),
]


# -- Detection -----------------------------------------------------------------

def detect_secrets(
    text: str,
    options: Optional[SecretDetectionOptions] = None,
) -> List[SecretDetection]:
    """Detect secrets, credentials, and tokens in text using regex patterns.
    Returns a list of detections sorted by start position.
    Text longer than 1 MB is truncated before scanning.
    """
    if not text:
        return []

    # Cap input length to prevent DoS via regex scanning
    scan_text = text[:MAX_SCAN_LENGTH] if len(text) > MAX_SCAN_LENGTH else text

    use_built_in = True
    if options and options.built_in_patterns is not None:
        use_built_in = options.built_in_patterns

    detections: List[SecretDetection] = []

    # -- Built-in patterns -----------------------------------------------------
    if use_built_in:
        for entry in _PATTERNS:
            for match in entry.pattern.finditer(scan_text):
                detections.append(SecretDetection(
                    type=entry.name,
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=entry.confidence,
                ))

    # -- Custom patterns -------------------------------------------------------
    if options and options.custom_patterns:
        for custom in options.custom_patterns:
            try:
                regex = re.compile(custom.pattern)
            except re.error:
                # Skip invalid regex strings silently
                continue

            for match in regex.finditer(scan_text):
                detections.append(SecretDetection(
                    type=f"custom:{custom.name}",
                    value=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=custom.confidence if custom.confidence is not None else 0.9,
                ))

    # Sort by start position, then by confidence descending for overlaps
    detections.sort(key=lambda d: (d.start, -d.confidence))

    # Remove overlapping detections (keep highest confidence)
    return _deduplicate_detections(detections)


# -- Deduplication -------------------------------------------------------------

def _deduplicate_detections(sorted_detections: List[SecretDetection]) -> List[SecretDetection]:
    if not sorted_detections:
        return sorted_detections

    result: List[SecretDetection] = [sorted_detections[0]]

    for i in range(1, len(sorted_detections)):
        prev = result[-1]
        curr = sorted_detections[i]

        # If current overlaps with previous, keep the one with higher confidence
        if curr.start < prev.end:
            if curr.confidence > prev.confidence:
                result[-1] = curr
            # else skip current (lower or equal confidence)
        else:
            result.append(curr)

    return result
