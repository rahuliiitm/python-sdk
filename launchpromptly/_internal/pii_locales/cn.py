"""China PII patterns: National ID (18-digit), phone."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional

from . import LocalePIIPattern

_ID_WEIGHTS = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
_CHECK_CHARS = "10X98765432"


def _cn_id_check_digit(digits: str) -> bool:
    id_str = re.sub(r"\s", "", digits).upper()
    if len(id_str) != 18:
        return False
    region = int(id_str[:2])
    if region < 11 or region > 65:
        return False
    year = int(id_str[6:10])
    month = int(id_str[10:12])
    day = int(id_str[12:14])
    if year < 1900 or year > datetime.now().year:
        return False
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False
    total = 0
    for i in range(17):
        d = int(id_str[i]) if id_str[i].isdigit() else -1
        if d < 0:
            return False
        total += d * _ID_WEIGHTS[i]
    expected = _CHECK_CHARS[total % 11]
    return id_str[17] == expected


_CN_ID_CONTEXT = re.compile(r"(?:身份证|id\s*card|national\s*id|citizen\s*id)", re.I)


def _cn_id_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    if not _cn_id_check_digit(match):
        return False
    if full_text:
        preceding = full_text[max(0, (match_index or 0) - 60):(match_index or 0)]
        if _CN_ID_CONTEXT.search(preceding):
            return True
        if re.search(r"[\u4e00-\u9fff]", preceding):
            return True
    return True


CN_PII_PATTERNS = [
    LocalePIIPattern(
        type="cn_national_id",
        label="Chinese National ID",
        regex=re.compile(r"\b[1-6]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b"),
        confidence=0.9,
        validate=_cn_id_validate,
    ),
    LocalePIIPattern(
        type="cn_phone",
        label="Chinese Phone",
        regex=re.compile(r"(?:\+86\s?)?1[3-9]\d[\s-]?\d{4}[\s-]?\d{4}\b"),
        confidence=0.8,
    ),
]
