"""Germany PII patterns: Steueridentifikationsnummer (Tax ID, 11 digits)."""

from __future__ import annotations

import re
from typing import Optional

from . import LocalePIIPattern


def _tax_id_check(digits: str) -> bool:
    nums = re.sub(r"[\s-]", "", digits)
    if len(nums) != 11 or not nums.isdigit():
        return False
    if nums[0] == "0":
        return False
    counts = [0] * 10
    for ch in nums[:10]:
        counts[int(ch)] += 1
    doubles = sum(1 for c in counts if c in (2, 3))
    if doubles != 1:
        return False
    if any(c > 3 for c in counts):
        return False
    return True


_DE_TAX_CONTEXT = re.compile(
    r"(?:steuer(?:identifikationsnummer|nummer|id|ident)|tin|tax\s*id|steuernummer)", re.I
)


def _tax_id_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    if not _tax_id_check(match):
        return False
    if not full_text or match_index is None:
        return False
    preceding = full_text[max(0, match_index - 80):match_index]
    return bool(_DE_TAX_CONTEXT.search(preceding))


DE_PII_PATTERNS = [
    LocalePIIPattern(
        type="de_tax_id",
        label="German Tax ID",
        regex=re.compile(r"\b[1-9]\d{10}\b"),
        confidence=0.8,
        validate=_tax_id_validate,
    ),
]
