"""Mexico PII patterns: RFC, CURP, phone."""

from __future__ import annotations

import re
from typing import Optional

from . import LocalePIIPattern

_RFC_CONTEXT = re.compile(r"\b(?:rfc|registro\s*federal|contribuyente)\b", re.I)
_CURP_CONTEXT = re.compile(r"\b(?:curp|clave\s*(?:unica|de\s*registro)|registro\s*de\s*poblaci[oó]n)\b", re.I)

_VALID_STATES = {
    "AS", "BC", "BS", "CC", "CL", "CM", "CS", "CH", "DF", "DG",
    "GT", "GR", "HG", "JC", "MC", "MN", "MS", "NT", "NL", "OC",
    "PL", "QT", "QR", "SP", "SL", "SR", "TC", "TS", "TL", "VZ",
    "YN", "ZS", "NE",
}


def _rfc_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    clean = re.sub(r"[\s-]", "", match)
    if len(clean) not in (12, 13):
        return False
    date_start = 4 if len(clean) == 13 else 3
    mm = int(clean[date_start + 2:date_start + 4])
    dd = int(clean[date_start + 4:date_start + 6])
    if mm < 1 or mm > 12:
        return False
    if dd < 1 or dd > 31:
        return False
    if full_text and match_index is not None:
        preceding = full_text[max(0, match_index - 60):match_index]
        if _RFC_CONTEXT.search(preceding):
            return True
    return True


def _curp_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    if len(match) != 18:
        return False
    mm = int(match[6:8])
    dd = int(match[8:10])
    if mm < 1 or mm > 12:
        return False
    if dd < 1 or dd > 31:
        return False
    gender = match[10]
    if gender not in ("H", "M"):
        return False
    state = match[11:13]
    if state not in _VALID_STATES:
        return False
    if full_text and match_index is not None:
        preceding = full_text[max(0, match_index - 60):match_index]
        if _CURP_CONTEXT.search(preceding):
            return True
    return True


MX_PII_PATTERNS = [
    LocalePIIPattern(
        type="mx_rfc",
        label="Mexican RFC",
        regex=re.compile(r"\b[A-ZÑ&]{3,4}\d{6}[A-Z0-9]{3}\b"),
        confidence=0.8,
        validate=_rfc_validate,
    ),
    LocalePIIPattern(
        type="mx_curp",
        label="Mexican CURP",
        regex=re.compile(r"\b[A-Z]{4}\d{6}[HM][A-Z]{2}[B-DF-HJ-NP-TV-Z]{3}[A-Z0-9]\d\b"),
        confidence=0.9,
        validate=_curp_validate,
    ),
    LocalePIIPattern(
        type="mx_phone",
        label="Mexican Phone",
        regex=re.compile(r"(?:\+52\s?)?(?:\(?[1-9]\d{1,2}\)?\s?)\d{3,4}[\s-]?\d{4}\b"),
        confidence=0.8,
    ),
]
