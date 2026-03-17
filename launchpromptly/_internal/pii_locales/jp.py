"""Japan PII patterns: My Number (12-digit), phone."""

from __future__ import annotations

import re
from typing import Optional

from . import LocalePIIPattern


def _my_number_check_digit(digits: str) -> bool:
    nums = re.sub(r"\s", "", digits)
    if len(nums) != 12 or not nums.isdigit():
        return False
    q = [6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2]
    total = sum(int(nums[i]) * q[i] for i in range(11))
    remainder = total % 11
    expected = 0 if remainder <= 1 else 11 - remainder
    return expected == int(nums[11])


_MY_NUMBER_CONTEXT = re.compile(r"(?:マイナンバー|my\s*number|個人番号)", re.I)


def _my_number_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    clean = re.sub(r"[\s-]", "", match)
    if not _my_number_check_digit(clean):
        return False
    if full_text:
        preceding = full_text[max(0, (match_index or 0) - 60):(match_index or 0)]
        if _MY_NUMBER_CONTEXT.search(preceding):
            return True
        if re.search(r"[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]", preceding):
            return True
    return False


JP_PII_PATTERNS = [
    LocalePIIPattern(
        type="jp_my_number",
        label="Japanese My Number",
        regex=re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),
        confidence=0.85,
        validate=_my_number_validate,
    ),
    LocalePIIPattern(
        type="jp_phone",
        label="Japanese Phone",
        regex=re.compile(r"(?:\+81\s?|0)[789]0[\s-]?\d{4}[\s-]?\d{4}\b"),
        confidence=0.8,
    ),
]
