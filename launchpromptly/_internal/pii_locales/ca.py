"""Canada PII patterns: SIN (Social Insurance Number)."""

from __future__ import annotations

import re
from typing import Optional

from . import LocalePIIPattern


def _luhn_check(digits: str) -> bool:
    nums = re.sub(r"[\s-]", "", digits)
    if len(nums) != 9 or not nums.isdigit():
        return False
    total = 0
    for i, ch in enumerate(nums):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


_SIN_CONTEXT = re.compile(r"\b(?:sin|social\s*insurance|social\s*ins)\b", re.I)


def _sin_context_check(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    if re.search(r"\d{3}[\s-]\d{3}[\s-]\d{3}", match):
        return _luhn_check(re.sub(r"[\s-]", "", match))
    if not full_text or match_index is None:
        return False
    preceding = full_text[max(0, match_index - 60):match_index]
    if not _SIN_CONTEXT.search(preceding):
        return False
    return _luhn_check(match)


CA_PII_PATTERNS = [
    LocalePIIPattern(
        type="ca_sin",
        label="Canadian SIN",
        regex=re.compile(r"\b(\d{3})[\s-]?(\d{3})[\s-]?(\d{3})\b"),
        confidence=0.85,
        validate=_sin_context_check,
    ),
]
