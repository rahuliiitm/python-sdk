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
    r"(?:(?<=\s)|(?<=\()|(?<=^))\+\d{1,3}[-.\s]?(?:[-.\s]?\d{2,6}){2,5}\b",
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

_IP_V6_RE = re.compile(
    r"\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b"
    r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"
    r"|::(?:[0-9a-fA-F]{1,4}:){0,5}[0-9a-fA-F]{1,4}\b"
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
    r"\b(?:(?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12]\d|3[01])"
    r"|(?:0[1-9]|[12]\d|3[01])[\/\-](?:0[1-9]|1[0-2]))[\/\-](?:19|20)\d{2}\b"
)

_DATE_OF_BIRTH_ISO_RE = re.compile(
    r"\b(?:19|20)\d{2}[\/\-](?:0[1-9]|1[0-2])[\/\-](?:0[1-9]|[12]\d|3[01])\b"
)

_US_ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+[A-Za-z][A-Za-z\s]{1,30}\s+"
    r"(?:St(?:reet)?|Ave(?:nue)?|Blvd|Boulevard|Dr(?:ive)?|Ln|Lane|Rd|Road"
    r"|Way|Ct|Court|Pl(?:ace)?|Cir(?:cle)?|Pkwy|Parkway)\b",
    re.IGNORECASE,
)

# -- International PII patterns ------------------------------------------------

_IBAN_RE = re.compile(r'\b[A-Z]{2}\d{2}\s?[A-Z0-9]{4}\s?\d{4}\s?\d{4}\s?[\dA-Z\s]{0,20}\b')

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

def _luhn_check(digits: str, _full_text: Optional[str] = None, _match_index: Optional[int] = None) -> bool:
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
    if len(digits) != 12 or not digits.isdigit():
        return False
    # Aadhaar numbers start with 2-9 (never 0 or 1)
    if digits[0] in ('0', '1'):
        return False
    return True


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


def _is_private_or_reserved_ip(ip: str) -> bool:
    """Check if an IP is in a private/reserved range (not PII)."""
    if ip in _WELL_KNOWN_IPS:
        return True
    parts = ip.split(".")
    if len(parts) != 4:
        return False
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return False
    a, b, c = nums[0], nums[1], nums[2]
    # Private ranges: 10.x.x.x, 172.16-31.x.x, 192.168.x.x
    if a == 10:
        return True
    if a == 172 and 16 <= b <= 31:
        return True
    if a == 192 and b == 168:
        return True
    # Link-local: 169.254.x.x
    if a == 169 and b == 254:
        return True
    # Documentation ranges (RFC 5737)
    if a == 192 and b == 0 and c == 2:
        return True
    if a == 198 and b == 51 and c == 100:
        return True
    if a == 203 and b == 0 and c == 113:
        return True
    return False


# -- Positive-context checks (reduce false positives) -------------------------
#
# Instead of maintaining an infinite blocklist of non-PII contexts, we require
# POSITIVE context for ambiguous matches. Formatted matches (with dashes/dots/
# parens) pass through. Bare digit/alphanumeric sequences need type-specific
# keywords.

_PHONE_CONTEXT_RE = re.compile(
    r"\b(?:call|phone|tel(?:ephone)?|mobile|cell(?:ular)?|fax|contact|reach"
    r"|dial|text|sms|whatsapp|ring|landline|ph)\b.{0,15}$",
    re.IGNORECASE,
)

_SSN_CONTEXT_RE = re.compile(
    r"\b(?:ssn|social\s+security|social\s+sec|ss#|soc\s*sec)\b.{0,15}$",
    re.IGNORECASE,
)

_NHS_CONTEXT_RE = re.compile(
    r"\b(?:nhs|national\s+health|health\s+service)\b.{0,15}$",
    re.IGNORECASE,
)

_AADHAAR_CONTEXT_RE = re.compile(
    r"\b(?:aadhaar|aadhar|uid|uidai|unique\s+id)\b.{0,15}$",
    re.IGNORECASE,
)

_MEDICARE_CONTEXT_RE = re.compile(
    r"\b(?:medicare|health\s+insurance|irn)\b.{0,15}$",
    re.IGNORECASE,
)

_DOB_CONTEXT_RE = re.compile(
    r"\b(?:born|birth(?:day)?|dob|date\s+of\s+birth|d\.o\.b|age)\b.{0,15}$",
    re.IGNORECASE,
)

_PASSPORT_CONTEXT_RE = re.compile(
    r"\b(?:passport)\b.{0,15}$",
    re.IGNORECASE,
)

_VERSION_CONTEXT_RE = re.compile(
    r"(?:\bv(?:ersion)?\s*[:#]?\s*$|@\s*$|\bv\d+\.\d+\.\s*$)",
    re.IGNORECASE,
)

_VERSION_SUFFIX_RE = re.compile(
    r"^[-.]?(?:alpha|beta|rc|dev|pre|snapshot)\b",
    re.IGNORECASE,
)


def _context_check(
    match: str,
    full_text: Optional[str],
    match_index: Optional[int],
    context_re: "re.Pattern[str]",
    require_always: bool,
) -> bool:
    """Generic context check.

    - require_always: if True, context is required even for formatted matches
      (DOB, passport). Otherwise, formatted matches always pass.
    """
    if full_text is None or match_index is None:
        return True

    if not require_always:
        digits_only = re.sub(r"\D", "", match)
        if match != digits_only:
            return True  # has formatting -> keep

    preceding_start = max(0, match_index - 60)
    preceding = full_text[preceding_start:match_index]
    return bool(context_re.search(preceding))


def _phone_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    return _context_check(match, full_text, match_index, _PHONE_CONTEXT_RE, False)


def _ssn_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    if not _ssn_check(match):
        return False
    return _context_check(match, full_text, match_index, _SSN_CONTEXT_RE, False)


def _nhs_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    if not _nhs_check(match):
        return False
    return _context_check(match, full_text, match_index, _NHS_CONTEXT_RE, False)


def _aadhaar_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    if not _aadhaar_check(match):
        return False
    return _context_check(match, full_text, match_index, _AADHAAR_CONTEXT_RE, False)


def _medicare_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    return _context_check(match, full_text, match_index, _MEDICARE_CONTEXT_RE, False)


def _dob_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    return _context_check(match, full_text, match_index, _DOB_CONTEXT_RE, True)


def _passport_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    return _context_check(match, full_text, match_index, _PASSPORT_CONTEXT_RE, True)


def _ip_context_check(
    match: str, full_text: Optional[str] = None, match_index: Optional[int] = None,
) -> bool:
    if _is_private_or_reserved_ip(match):
        return False
    if full_text is None or match_index is None:
        return True
    preceding = full_text[max(0, match_index - 30):match_index]
    if _VERSION_CONTEXT_RE.search(preceding):
        return False
    following = full_text[match_index + len(match):match_index + len(match) + 10]
    if _VERSION_SUFFIX_RE.search(following):
        return False
    return True


# -- Pattern registry ---------------------------------------------------------

@dataclass
class _PatternEntry:
    type: PIIType
    regex: re.Pattern[str]
    confidence: float
    validate: Optional[Callable[..., bool]] = None


_PATTERNS: List[_PatternEntry] = [
    _PatternEntry(type="email", regex=_EMAIL_RE, confidence=0.95),
    _PatternEntry(type="phone", regex=_PHONE_US_RE, confidence=0.85, validate=_phone_context_check),
    _PatternEntry(type="phone", regex=_PHONE_INTL_RE, confidence=0.8),
    _PatternEntry(type="ssn", regex=_SSN_RE, confidence=0.95, validate=_ssn_context_check),
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
        validate=_ip_context_check,
    ),
    _PatternEntry(type="api_key", regex=_API_KEY_RE, confidence=0.95),
    _PatternEntry(type="date_of_birth", regex=_DATE_OF_BIRTH_RE, confidence=0.7, validate=_dob_context_check),
    _PatternEntry(type="us_address", regex=_US_ADDRESS_RE, confidence=0.7),
    # International PII patterns
    _PatternEntry(type="iban", regex=_IBAN_RE, confidence=0.9),
    _PatternEntry(
        type="nhs_number",
        regex=_NHS_NUMBER_RE,
        confidence=0.8,
        validate=_nhs_context_check,
    ),
    _PatternEntry(type="uk_nino", regex=_UK_NINO_RE, confidence=0.9),
    _PatternEntry(type="passport", regex=_PASSPORT_RE, confidence=0.7, validate=_passport_context_check),
    _PatternEntry(
        type="aadhaar",
        regex=_AADHAAR_RE,
        confidence=0.85,
        validate=_aadhaar_context_check,
    ),
    _PatternEntry(type="eu_phone", regex=_EU_PHONE_RE, confidence=0.8),
    _PatternEntry(type="medicare", regex=_MEDICARE_AU_RE, confidence=0.75, validate=_medicare_context_check),
    _PatternEntry(type="drivers_license", regex=_DRIVERS_LICENSE_US_RE, confidence=0.75),
    _PatternEntry(
        type="ip_address",
        regex=_IP_V6_RE,
        confidence=0.8,
    ),
    _PatternEntry(type="date_of_birth", regex=_DATE_OF_BIRTH_ISO_RE, confidence=0.7, validate=_dob_context_check),
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

            # Run optional validation (e.g., Luhn for credit cards, context checks)
            if pattern.validate and not pattern.validate(value, scan_text, match.start()):
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


# ── Suppressive context: business terms near PII matches reduce confidence ──

_SUPPRESSIVE_CONTEXT: dict[str, re.Pattern[str]] = {
    "phone": re.compile(
        r"\b(?:order|sku|tracking|invoice|reference|ref|ticket|case|account|id|code|pin|ext(?:ension)?|fax|zip|postal)\b",
        re.IGNORECASE,
    ),
    "ssn": re.compile(
        r"\b(?:order|tracking|invoice|reference|serial|model|sku|part|item|product|account|routing)\b",
        re.IGNORECASE,
    ),
    "credit_card": re.compile(
        r"\b(?:order|tracking|serial|model|sku|part|item|product|barcode)\b",
        re.IGNORECASE,
    ),
}


def apply_suppressive_context(detection: PIIDetection, full_text: str) -> PIIDetection:
    """If business terms appear near a PII match, reduce confidence by 0.3 (floor 0.1)."""
    suppress_re = _SUPPRESSIVE_CONTEXT.get(detection.type)
    if not suppress_re:
        return detection
    preceding = full_text[max(0, detection.start - 60):detection.start]
    if suppress_re.search(preceding):
        return PIIDetection(
            type=detection.type,
            value=detection.value,
            start=detection.start,
            end=detection.end,
            confidence=max(detection.confidence - 0.3, 0.1),
        )
    return detection


def filter_allow_list(detections: List[PIIDetection], allow_list: List[str]) -> List[PIIDetection]:
    """Filter detections against an allow list (normalized: strip formatting)."""
    if not allow_list:
        return detections
    normalize = lambda s: re.sub(r"[\s\-().]", "", s)
    allow_set = {normalize(v) for v in allow_list}
    return [d for d in detections if normalize(d.value) not in allow_set]


def filter_by_confidence(
    detections: List[PIIDetection],
    thresholds: dict[str, float],
) -> List[PIIDetection]:
    """Filter detections by per-type confidence thresholds."""
    return [
        d for d in detections
        if thresholds.get(d.type) is None or d.confidence >= thresholds[d.type]
    ]


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
