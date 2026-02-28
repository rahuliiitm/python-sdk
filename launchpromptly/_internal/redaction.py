"""
PII redaction module -- replaces detected PII with safe substitutes.
Supports three strategies: placeholder, synthetic, and hash.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

from .pii import (
    PIIDetection,
    PIIDetectOptions,
    PIIDetectorProvider,
    PIIType,
    detect_pii,
    merge_detections,
)

RedactionStrategy = Literal["placeholder", "synthetic", "hash", "mask", "none"]


@dataclass
class MaskingOptions:
    char: str = "*"
    visible_suffix: int = 4
    visible_prefix: int = 0


@dataclass
class RedactionOptions:
    strategy: Optional[RedactionStrategy] = None
    types: Optional[List[PIIType]] = None
    providers: Optional[List[PIIDetectorProvider]] = None
    masking: Optional[MaskingOptions] = None


@dataclass
class RedactionResult:
    redacted_text: str
    detections: List[PIIDetection]
    mapping: Dict[str, str] = field(default_factory=dict)


# -- Synthetic data pools ------------------------------------------------------

_SYNTHETIC_EMAILS = [
    "alex@example.net",
    "sam@example.org",
    "pat@example.com",
    "jordan@example.net",
    "taylor@example.org",
    "morgan@example.com",
    "casey@example.net",
    "drew@example.org",
]

_SYNTHETIC_PHONES = [
    "(555) 100-0001",
    "(555) 100-0002",
    "(555) 100-0003",
    "(555) 100-0004",
    "(555) 100-0005",
]

_SYNTHETIC_SSNS = [
    "000-00-0001",
    "000-00-0002",
    "000-00-0003",
    "000-00-0004",
]

_SYNTHETIC_CARDS = [
    "4000-0000-0000-0001",
    "4000-0000-0000-0002",
    "4000-0000-0000-0003",
]

_SYNTHETIC_IPS = [
    "198.51.100.1",
    "198.51.100.2",
    "198.51.100.3",
    "198.51.100.4",
]

_SYNTHETIC_MAP: Dict[str, List[str]] = {
    "email": _SYNTHETIC_EMAILS,
    "phone": _SYNTHETIC_PHONES,
    "ssn": _SYNTHETIC_SSNS,
    "credit_card": _SYNTHETIC_CARDS,
    "ip_address": _SYNTHETIC_IPS,
    "api_key": ["sk-REDACTED-0001", "sk-REDACTED-0002", "sk-REDACTED-0003"],
    "date_of_birth": ["01/01/2000", "06/15/1985", "12/25/1970"],
    "us_address": ["100 Example St", "200 Sample Ave", "300 Test Blvd"],
}


# -- Counters per type (for placeholder indexing) ------------------------------

def _next_index(counters: Dict[str, int], pii_type: str) -> int:
    counters[pii_type] = counters.get(pii_type, 0) + 1
    return counters[pii_type]


# -- Replacement generators ---------------------------------------------------

def _placeholder_replacement(pii_type: str, counters: Dict[str, int]) -> str:
    idx = _next_index(counters, pii_type)
    return f"[{pii_type.upper()}_{idx}]"


def _synthetic_replacement(pii_type: str, counters: Dict[str, int]) -> str:
    idx = _next_index(counters, pii_type)
    pool = _SYNTHETIC_MAP.get(pii_type)
    if not pool:
        return _placeholder_replacement(pii_type, counters)
    return pool[(idx - 1) % len(pool)]


def _hash_replacement(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]


def _mask_replacement(value: str, pii_type: str, masking: MaskingOptions) -> str:
    """Generate a masked version of the PII value based on its type.

    Applies type-specific masking logic:
    - credit_card: ****-****-****-1234 (show last 4)
    - email: j***@acme.com (mask local part except first char, keep domain)
    - phone: ***-***-4567 (show last 4)
    - ssn: ***-**-6789 (show last 4)
    - default: replace all but last N chars with mask char
    """
    char = masking.char

    if pii_type == "credit_card":
        # Show last 4 digits, mask the rest preserving dash format
        stripped = value.replace("-", "").replace(" ", "")
        last4 = stripped[-4:]
        return f"{char * 4}-{char * 4}-{char * 4}-{last4}"

    if pii_type == "email":
        # Keep first char of local part, mask rest, preserve domain
        at_idx = value.find("@")
        if at_idx <= 0:
            # fallback for malformed emails
            return char * len(value)
        local = value[:at_idx]
        domain = value[at_idx:]  # includes the @
        first_char = local[0]
        masked_local = first_char + char * (len(local) - 1)
        return masked_local + domain

    if pii_type == "phone":
        # Show last 4 digits, mask the rest
        digits = ""
        positions: List[int] = []
        for i, c in enumerate(value):
            if c.isdigit():
                digits += c
                positions.append(i)
        if len(digits) <= 4:
            return value
        # Build result: mask all digits except last 4
        result_chars = list(value)
        keep_start = len(digits) - 4
        digit_idx = 0
        for i, c in enumerate(value):
            if c.isdigit():
                if digit_idx < keep_start:
                    result_chars[i] = char
                digit_idx += 1
        return "".join(result_chars)

    if pii_type == "ssn":
        # SSN format: ***-**-6789 (show last 4)
        parts = value.split("-")
        if len(parts) == 3:
            return f"{char * 3}-{char * 2}-{parts[2]}"
        # fallback
        return char * max(0, len(value) - 4) + value[-4:]

    # Default: replace all but visible_suffix / visible_prefix with mask char
    prefix_len = masking.visible_prefix
    suffix_len = masking.visible_suffix
    total = len(value)

    if prefix_len + suffix_len >= total:
        return value  # nothing to mask

    prefix = value[:prefix_len]
    suffix = value[total - suffix_len:] if suffix_len > 0 else ""
    mask_len = total - prefix_len - suffix_len
    return prefix + char * mask_len + suffix


# -- Public API ----------------------------------------------------------------

def redact_pii(
    text: str,
    options: Optional[RedactionOptions] = None,
) -> RedactionResult:
    """Detect and redact PII in text.

    Returns the redacted text, a list of detections, and a mapping for de-redaction.
    """
    if not text:
        return RedactionResult(redacted_text="", detections=[], mapping={})

    strategy: RedactionStrategy = (options.strategy if options and options.strategy else "placeholder")

    # Build PIIDetectOptions from redaction options
    detect_opts = PIIDetectOptions(types=options.types if options else None)

    # Run built-in regex detection
    detections = detect_pii(text, detect_opts)

    # Run additional providers and merge
    if options and options.providers:
        provider_detections: List[List[PIIDetection]] = []
        for provider in options.providers:
            try:
                provider_detections.append(provider.detect(text, detect_opts))
            except Exception:
                # Plugin isolation: failures don't crash core
                provider_detections.append([])
        detections = merge_detections(detections, *provider_detections)

    if not detections:
        return RedactionResult(redacted_text=text, detections=[], mapping={})

    mapping: Dict[str, str] = {}
    counters: Dict[str, int] = {}

    # Build redacted text by replacing from end to start (preserves positions)
    redacted = text
    # Process in reverse order so replacements don't shift positions
    reversed_dets = list(reversed(detections))

    masking = (options.masking if options else None) or MaskingOptions()

    for det in reversed_dets:
        if strategy == "synthetic":
            replacement = _synthetic_replacement(det.type, counters)
        elif strategy == "hash":
            replacement = _hash_replacement(det.value)
        elif strategy == "mask":
            replacement = _mask_replacement(det.value, det.type, masking)
        else:
            # placeholder (default)
            replacement = _placeholder_replacement(det.type, counters)

        redacted = redacted[: det.start] + replacement + redacted[det.end :]
        # mapping: replacement -> original (for de-redaction)
        mapping[replacement] = det.value

    return RedactionResult(redacted_text=redacted, detections=detections, mapping=mapping)


def de_redact(text: str, mapping: Dict[str, str]) -> str:
    """Restore original values in text using the redaction mapping.

    Only works with placeholder and synthetic strategies (hash is one-way).
    """
    result = text
    for replacement, original in mapping.items():
        # Replace all occurrences of the replacement
        result = result.replace(replacement, original)
    return result
