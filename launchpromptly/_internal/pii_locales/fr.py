"""France PII patterns: NIR (INSEE number, 15 digits)."""

from __future__ import annotations

import re
from typing import Optional

from . import LocalePIIPattern

_NIR_CONTEXT = re.compile(
    r"(?:nir|insee|s[eé]curit[eé]\s*sociale|num[eé]ro\s*de\s*s[eé]curit[eé]|social\s*security)", re.I
)


def _nir_check_digit(digits: str) -> bool:
    nums = re.sub(r"[\s-]", "", digits)
    if len(nums) != 15 or not nums.isdigit():
        return False
    gender = int(nums[0])
    if gender not in (1, 2):
        return False
    month = int(nums[3:5])
    if month < 1 or (month > 12 and month != 20):
        return False
    base = int(nums[:13])
    key = int(nums[13:15])
    return key == 97 - (base % 97)


def _nir_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    if not _nir_check_digit(match):
        return False
    if full_text and match_index is not None:
        preceding = full_text[max(0, match_index - 80):match_index]
        if _NIR_CONTEXT.search(preceding):
            return True
    return True


FR_PII_PATTERNS = [
    LocalePIIPattern(
        type="fr_nir",
        label="French NIR/INSEE",
        regex=re.compile(r"\b[12]\s?\d{2}\s?\d{2}\s?\d{5}\s?\d{3}\s?\d{2}\b"),
        confidence=0.85,
        validate=_nir_validate,
    ),
]
