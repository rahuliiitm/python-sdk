"""South Korea PII patterns: Resident Registration Number (RRN), phone."""

from __future__ import annotations

import re
from typing import Optional

from . import LocalePIIPattern


def _rrn_check_digit(digits: str) -> bool:
    nums = re.sub(r"[\s-]", "", digits)
    if len(nums) != 13 or not nums.isdigit():
        return False
    month = int(nums[2:4])
    day = int(nums[4:6])
    if month < 1 or month > 12:
        return False
    if day < 1 or day > 31:
        return False
    gender = int(nums[6])
    if gender < 1 or gender > 8:
        return False
    weights = [2, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5]
    total = sum(int(nums[i]) * weights[i] for i in range(12))
    expected = (11 - (total % 11)) % 10
    return expected == int(nums[12])


_RRN_CONTEXT = re.compile(r"(?:주민등록|resident\s*registration|rrn)", re.I)


def _rrn_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    if not _rrn_check_digit(match):
        return False
    if re.search(r"\d{6}-\d{7}", match):
        return True
    if full_text:
        preceding = full_text[max(0, (match_index or 0) - 60):(match_index or 0)]
        if _RRN_CONTEXT.search(preceding):
            return True
        if re.search(r"[\uac00-\ud7af]", preceding):
            return True
    return False


KR_PII_PATTERNS = [
    LocalePIIPattern(
        type="kr_rrn",
        label="South Korean RRN",
        regex=re.compile(r"\b\d{6}-?\d{7}\b"),
        confidence=0.85,
        validate=_rrn_validate,
    ),
    LocalePIIPattern(
        type="kr_phone",
        label="South Korean Phone",
        regex=re.compile(r"(?:\+82\s?)?0?1[016-9][\s-]?\d{3,4}[\s-]?\d{4}\b"),
        confidence=0.8,
    ),
]
