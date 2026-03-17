"""Brazil PII patterns: CPF, CNPJ, phone."""

from __future__ import annotations

import re
from typing import Optional

from . import LocalePIIPattern


def _cpf_check_digits(digits: str) -> bool:
    nums = re.sub(r"\D", "", digits)
    if len(nums) != 11:
        return False
    if len(set(nums)) == 1:
        return False
    # First check digit
    s = sum(int(nums[i]) * (10 - i) for i in range(9))
    c = 11 - (s % 11)
    if c >= 10:
        c = 0
    if c != int(nums[9]):
        return False
    # Second check digit
    s = sum(int(nums[i]) * (11 - i) for i in range(10))
    c = 11 - (s % 11)
    if c >= 10:
        c = 0
    return c == int(nums[10])


def _cnpj_check_digits(digits: str) -> bool:
    nums = re.sub(r"\D", "", digits)
    if len(nums) != 14:
        return False
    w1 = [5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    s = sum(int(nums[i]) * w1[i] for i in range(12))
    c = 11 - (s % 11)
    if c >= 10:
        c = 0
    if c != int(nums[12]):
        return False
    w2 = [6, 5, 4, 3, 2, 9, 8, 7, 6, 5, 4, 3, 2]
    s = sum(int(nums[i]) * w2[i] for i in range(13))
    c = 11 - (s % 11)
    if c >= 10:
        c = 0
    return c == int(nums[13])


_CPF_CONTEXT = re.compile(r"\b(?:cpf|cadastro\s*de\s*pessoa)\b", re.I)
_CNPJ_CONTEXT = re.compile(r"\b(?:cnpj|cadastro\s*nacional)\b", re.I)


def _cpf_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    digits = re.sub(r"\D", "", match)
    if not _cpf_check_digits(digits):
        return False
    if re.search(r"\d{3}\.\d{3}\.\d{3}-\d{2}", match):
        return True
    if not full_text or match_index is None:
        return False
    preceding = full_text[max(0, match_index - 60):match_index]
    return bool(_CPF_CONTEXT.search(preceding))


def _cnpj_validate(match: str, full_text: Optional[str] = None, match_index: Optional[int] = None) -> bool:
    digits = re.sub(r"\D", "", match)
    if not _cnpj_check_digits(digits):
        return False
    if re.search(r"\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}", match):
        return True
    if not full_text or match_index is None:
        return False
    preceding = full_text[max(0, match_index - 60):match_index]
    return bool(_CNPJ_CONTEXT.search(preceding))


BR_PII_PATTERNS = [
    LocalePIIPattern(
        type="br_cpf",
        label="Brazilian CPF",
        regex=re.compile(r"\b(\d{3})\.?(\d{3})\.?(\d{3})-?(\d{2})\b"),
        confidence=0.9,
        validate=_cpf_validate,
    ),
    LocalePIIPattern(
        type="br_cnpj",
        label="Brazilian CNPJ",
        regex=re.compile(r"\b(\d{2})\.?(\d{3})\.?(\d{3})/?(\d{4})-?(\d{2})\b"),
        confidence=0.9,
        validate=_cnpj_validate,
    ),
    LocalePIIPattern(
        type="br_phone",
        label="Brazilian Phone",
        regex=re.compile(r"(?:\+55\s?)?(?:\(?[1-9]\d\)?\s?)(?:9\s?\d{4})[\s-]?\d{4}\b"),
        confidence=0.8,
    ),
]
