"""PII locale registry — lazy loading of country-specific patterns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Literal, Optional, Union
import re

PIILocale = Literal["ca", "br", "cn", "jp", "kr", "de", "mx", "fr"]

ALL_LOCALES: List[PIILocale] = ["ca", "br", "cn", "jp", "kr", "de", "mx", "fr"]


@dataclass
class LocalePIIPattern:
    type: str
    label: str
    regex: re.Pattern
    confidence: float
    validate: Optional[Callable] = None


_cache: dict[str, List[LocalePIIPattern]] = {}


def _load_locale(locale: PIILocale) -> List[LocalePIIPattern]:
    if locale in _cache:
        return _cache[locale]

    if locale == "ca":
        from .ca import CA_PII_PATTERNS as patterns
    elif locale == "br":
        from .br import BR_PII_PATTERNS as patterns
    elif locale == "cn":
        from .cn import CN_PII_PATTERNS as patterns
    elif locale == "jp":
        from .jp import JP_PII_PATTERNS as patterns
    elif locale == "kr":
        from .kr import KR_PII_PATTERNS as patterns
    elif locale == "de":
        from .de import DE_PII_PATTERNS as patterns
    elif locale == "mx":
        from .mx import MX_PII_PATTERNS as patterns
    elif locale == "fr":
        from .fr import FR_PII_PATTERNS as patterns
    else:
        patterns = []

    _cache[locale] = patterns
    return patterns


def get_locale_patterns(locales: Union[List[PIILocale], str]) -> List[LocalePIIPattern]:
    """Get PII patterns for specified locales."""
    codes: List[PIILocale] = ALL_LOCALES if locales == "all" else locales  # type: ignore[assignment]
    result: List[LocalePIIPattern] = []
    for code in codes:
        result.extend(_load_locale(code))
    return result
