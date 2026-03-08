"""
PII (Personally Identifiable Information) detection module.
Zero-dependency, regex-based scanner for common PII patterns.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Protocol, Sequence

PIIType = Literal[
    "email",
    "phone",
    "ssn",
    "credit_card",
    "ip_address",
    "api_key",
    "date_of_birth",
    "us_address",
    "iban",
    "nhs_number",
    "uk_nino",
    "passport",
    "aadhaar",
    "eu_phone",
    "medicare",
    "drivers_license",
]


@dataclass
class PIIDetection:
    type: PIIType
    value: str
    start: int
    end: int
    confidence: float


@dataclass
class PIIDetectOptions:
    types: Optional[List[PIIType]] = None


class PIIDetectorProvider(Protocol):
    def detect(self, text: str, options: Optional[PIIDetectOptions] = None) -> List[PIIDetection]: ...

    @property
    def name(self) -> str: ...

    @property
    def supported_types(self) -> List[PIIType]: ...


# -- Regex patterns -----------------------------------------------------------

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

_PHONE_US_RE = re.compile(
    r"\b(?:\+1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
)

# Python re requires fixed-width lookbehinds; use (?:^|(?<=[\s(])) instead
_PHONE_INTL_RE = re.compile(
    r"(?:(?<=\s)|(?<=\()|(?<=^))\+\d{1,3}[-.\s]?\d{4,14}(?:[-.\s]\d{1,6})*\b",
    re.MULTILINE,
)

_SSN_RE = re.compile(r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b")

_CREDIT_CARD_RE = re.compile(r"\b\d(?:[\s\-]?\d){12,18}\b")

_IP_V4_RE = re.compile(
    r"\b(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\."
    r"(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b"
)

# Common API key / secret patterns
_API_KEY_RE = re.compile(
    r"\b(?:"
    r"sk-[a-zA-Z0-9]{20,}"
    r"|sk-proj-[a-zA-Z0-9\-_]{20,}"
    r"|AKIA[0-9A-Z]{16}"
    r"|ghp_[a-zA-Z0-9]{36}"
    r"|gho_[a-zA-Z0-9]{36}"
    r"|glpat-[a-zA-Z0-9\-_]{20,}"
    r"|xox[bsapr]-[a-zA-Z0-9\-]{10,}"
    r")\b"
)

_DATE_OF_BIRTH_RE = re.compile(
    r"\b(?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12]\d|3[01])[\/\-](?:19|20)\d{2}\b"
)

_US_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z][A-Za-z\s]{1,30}\s+"
    r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Ln|Lane|Rd|Road"
    r"|Way|Ct|Court|Pl(?:ace)?|Cir(?:cle)?|Pkwy|Parkway)\b",
    re.IGNORECASE,
)

# -- International PII patterns ------------------------------------------------

_IBAN_RE = re.compile(r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b')

_NHS_NUMBER_RE = re.compile(r'\b\d{3}\s?\d{3}\s?\d{4}\b')

_UK_NINO_RE = re.compile(
    r'\b[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z]\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b'
)

_PASSPORT_RE = re.compile(r'\b[A-Z]{1,2}\d{6,9}\b')

_AADHAAR_RE = re.compile(r'(?<!\+)\b\d{4}\s?\d{4}\s?\d{4}\b')

_EU_PHONE_RE = re.compile(
    r'(?:(?<=\s)|(?<=^))\+(?:33|49|34|39|31|32|43|41|46|47|48|45|358|351|353|30|36)\s?\d[\d\s]{7,12}\b',
    re.MULTILINE,
)

_MEDICARE_AU_RE = re.compile(r'\b\d{4}\s?\d{5}\s?\d{1}\b')

_DRIVERS_LICENSE_US_RE = re.compile(r'\b[A-Z]\d{3}-\d{4}-\d{4}\b')


# -- Luhn check for credit cards ----------------------------------------------

def _luhn_check(digits: str) -> bool:
    nums = re.sub(r"[\s\-]", "", digits)
    if not re.fullmatch(r"\d{13,19}", nums):
        return False

    total = 0
    alternate = False
    for i in range(len(nums) - 1, -1, -1):
        n = int(nums[i])
        if alternate:
            n *= 2
            if n > 9:
                n -= 9
        total += n
        alternate = not alternate
    return total % 10 == 0


# -- NHS number validation (must be exactly 10 digits) ------------------------

def _nhs_check(value: str) -> bool:
    digits = re.sub(r"\s", "", value)
    return len(digits) == 10 and digits.isdigit()


# -- Aadhaar validation (must be exactly 12 digits) ---------------------------

def _aadhaar_check(value: str) -> bool:
    digits = re.sub(r"\s", "", value)
    return len(digits) == 12 and digits.isdigit()


# -- SSN validation (area/group/serial checks) --------------------------------

def _ssn_check(value: str) -> bool:
    digits = re.sub(r"[-\s]", "", value)
    if len(digits) != 9:
        return False
    area, group, serial = int(digits[:3]), int(digits[3:5]), int(digits[5:9])
    if area == 0 or area == 666 or area >= 900:
        return False
    if group == 0 or serial == 0:
        return False
    return True


# -- Well-known non-PII IP addresses ------------------------------------------

_WELL_KNOWN_IPS = frozenset({
    "0.0.0.0",
    "127.0.0.1",
    "255.255.255.255",
    "255.255.255.0",
    "192.168.0.1",
    "192.168.1.1",
    "10.0.0.1",
})


# -- Pattern registry ---------------------------------------------------------

@dataclass
class _PatternEntry:
    type: PIIType
    regex: re.Pattern[str]
    confidence: float
    validate: Optional[Callable[[str], bool]] = None


_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(type="email", regex=_EMAIL_RE, confidence=0.95),
    _PatternEntry(type="phone", regex=_PHONE_US_RE, confidence=0.85),
    _PatternEntry(type="phone", regex=_PHONE_INTL_RE, confidence=0.8),
    _PatternEntry(type="ssn", regex=_SSN_RE, confidence=0.95, validate=_ssn_check),
    _PatternEntry(
        type="credit_card",
        regex=_CREDIT_CARD_RE,
        confidence=0.9,
        validate=_luhn_check,
    ),
    _PatternEntry(
        type="ip_address",
        regex=_IP_V4_RE,
        confidence=0.8,
        validate=lambda ip: ip not in _WELL_KNOWN_IPS,
    ),
    _PatternEntry(type="api_key", regex=_API_KEY_RE, confidence=0.95),
    _PatternEntry(type="date_of_birth", regex=_DATE_OF_BIRTH_RE, confidence=0.7),
    _PatternEntry(type="us_address", regex=_US_ADDRESS_RE, confidence=0.7),
    # International PII patterns
    _PatternEntry(type="iban", regex=_IBAN_RE, confidence=0.9),
    _PatternEntry(
        type="nhs_number",
        regex=_NHS_NUMBER_RE,
        confidence=0.8,
        validate=_nhs_check,
    ),
    _PatternEntry(type="uk_nino", regex=_UK_NINO_RE, confidence=0.9),
    _PatternEntry(type="passport", regex=_PASSPORT_RE, confidence=0.7),
    _PatternEntry(
        type="aadhaar",
        regex=_AADHAAR_RE,
        confidence=0.85,
        validate=_aadhaar_check,
    ),
    _PatternEntry(type="eu_phone", regex=_EU_PHONE_RE, confidence=0.8),
    _PatternEntry(type="medicare", regex=_MEDICARE_AU_RE, confidence=0.75),
    _PatternEntry(type="drivers_license", regex=_DRIVERS_LICENSE_US_RE, confidence=0.75),
]


# -- Detection ----------------------------------------------------------------

def _deduplicate_detections(sorted_dets: List[PIIDetection]) -> List[PIIDetection]:
    if not sorted_dets:
        return sorted_dets

    result: List[PIIDetection] = [sorted_dets[0]]

    for i in range(1, len(sorted_dets)):
        prev = result[-1]
        curr = sorted_dets[i]

        # If current overlaps with previous, keep the one with higher confidence
        if curr.start < prev.end:
            if curr.confidence > prev.confidence:
                result[-1] = curr
            # else skip current (lower or equal confidence)
        else:
            result.append(curr)

    return result


_MAX_SCAN_LENGTH = 1_000_000  # 1MB — cap input to prevent DoS via regex scanning


def detect_pii(
    text: str,
    options: Optional[PIIDetectOptions] = None,
) -> List[PIIDetection]:
    """Detect PII entities in text using regex patterns.

    Returns a list of detections sorted by start position.
    Text longer than 1MB is truncated before scanning.
    """
    if not text:
        return []

    # Cap input length to prevent DoS
    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    allowed_types = set(options.types) if options and options.types else None
    detections: List[PIIDetection] = []

    for pattern in _PATTERNS:
        if allowed_types and pattern.type not in allowed_types:
            continue

        for match in pattern.regex.finditer(scan_text):
            value = match.group(0)

            # Run optional validation (e.g., Luhn for credit cards)
            if pattern.validate and not pattern.validate(value):
                continue

            detections.append(
                PIIDetection(
                    type=pattern.type,
                    value=value,
                    start=match.start(),
                    end=match.end(),
                    confidence=pattern.confidence,
                )
            )

    # Sort by start position, then by confidence descending for overlaps
    detections.sort(key=lambda d: (d.start, -d.confidence))

    # Remove overlapping detections (keep highest confidence)
    return _deduplicate_detections(detections)


def merge_detections(*detection_arrays: List[PIIDetection]) -> List[PIIDetection]:
    """Merge detections from multiple providers, deduplicating overlapping spans."""
    all_dets: List[PIIDetection] = []
    for arr in detection_arrays:
        all_dets.extend(arr)
    all_dets.sort(key=lambda d: (d.start, -d.confidence))
    return _deduplicate_detections(all_dets)


class RegexPIIDetector:
    """Built-in regex PII detector implementing the provider interface."""

    @property
    def name(self) -> str:
        return "regex"

    @property
    def supported_types(self) -> List[PIIType]:
        return [
            "email",
            "phone",
            "ssn",
            "credit_card",
            "ip_address",
            "api_key",
            "date_of_birth",
            "us_address",
            "iban",
            "nhs_number",
            "uk_nino",
            "passport",
            "aadhaar",
            "eu_phone",
            "medicare",
            "drivers_license",
        ]

    def detect(
        self,
        text: str,
        options: Optional[PIIDetectOptions] = None,
    ) -> List[PIIDetection]:
        return detect_pii(text, options)


# -- Custom PII pattern registration ------------------------------------------

@dataclass
class CustomPIIPattern:
    """Definition for a user-supplied PII pattern."""

    name: str
    type: str
    pattern: str  # regex string
    confidence: float = 0.8


class _CustomPIIDetector:
    """PII detector built from user-supplied patterns."""

    def __init__(self, patterns: List[CustomPIIPattern]) -> None:
        self._patterns = patterns
        self._compiled = [
            (p, re.compile(p.pattern)) for p in patterns
        ]

    @property
    def name(self) -> str:
        return "custom"

    @property
    def supported_types(self) -> List[str]:
        return list({p.type for p in self._patterns})

    def detect(
        self,
        text: str,
        options: Optional[PIIDetectOptions] = None,
    ) -> List[PIIDetection]:
        if not text:
            return []

        allowed_types = set(options.types) if options and options.types else None
        detections: List[PIIDetection] = []

        for pattern, compiled in self._compiled:
            if allowed_types and pattern.type not in allowed_types:
                continue

            for match in compiled.finditer(text):
                detections.append(
                    PIIDetection(
                        type=pattern.type,  # type: ignore[arg-type]
                        value=match.group(0),
                        start=match.start(),
                        end=match.end(),
                        confidence=pattern.confidence,
                    )
                )

        detections.sort(key=lambda d: (d.start, -d.confidence))
        return _deduplicate_detections(detections)


def create_custom_detector(patterns: List[CustomPIIPattern]) -> _CustomPIIDetector:
    """Create a PII detector from a list of custom regex patterns.

    The returned object implements the PIIDetectorProvider protocol and can be
    passed to ``PIISecurityOptions.providers`` or ``RedactionOptions.providers``.
    """
    return _CustomPIIDetector(patterns)
