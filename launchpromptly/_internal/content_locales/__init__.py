"""Content filter locale support — language detection + locale patterns."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

ContentLocale = Literal["es", "pt", "zh", "ja", "ko", "de", "fr", "ar", "hi", "ru"]

ALL_CONTENT_LOCALES: List[ContentLocale] = [
    "es", "pt", "zh", "ja", "ko", "de", "fr", "ar", "hi", "ru"
]


@dataclass
class LocaleContentPattern:
    category: str
    patterns: List[re.Pattern]
    severity: Literal["warn", "block"]


@dataclass
class LanguageDetection:
    language: str
    confidence: float


# ── Script-based detection ───────────────────────────────────────────────────

_SCRIPT_RANGES = [
    (re.compile(r"[\u3040-\u309f]"), "ja"),   # Hiragana
    (re.compile(r"[\u30a0-\u30ff]"), "ja"),   # Katakana
    (re.compile(r"[\uac00-\ud7af]"), "ko"),   # Hangul
    (re.compile(r"[\u0600-\u06ff]"), "ar"),   # Arabic
    (re.compile(r"[\u0900-\u097f]"), "hi"),   # Devanagari
    (re.compile(r"[\u0400-\u04ff]"), "ru"),   # Cyrillic
    (re.compile(r"[\u4e00-\u9fff]"), "zh"),   # CJK
]

_STOP_WORDS: Dict[str, set] = {
    "en": {"the", "is", "at", "which", "on", "and", "or", "but", "this", "that", "with", "for", "not", "you", "all", "can", "had", "her", "was", "one"},
    "es": {"el", "la", "los", "las", "un", "una", "es", "en", "por", "con", "del", "que", "para", "como", "pero", "más", "este", "esta", "son", "no"},
    "pt": {"o", "a", "os", "as", "um", "uma", "em", "de", "do", "da", "que", "com", "para", "por", "não", "mais", "como", "seu", "sua", "dos"},
    "de": {"der", "die", "das", "ein", "eine", "ist", "und", "oder", "aber", "mit", "für", "auf", "von", "den", "dem", "des", "sich", "nicht", "aus", "auch"},
    "fr": {"le", "la", "les", "un", "une", "est", "et", "ou", "mais", "avec", "pour", "dans", "sur", "par", "pas", "que", "qui", "son", "ses", "des"},
}


def detect_language(text: str) -> LanguageDetection:
    """Detect the dominant language of a text."""
    if not text or len(text) < 10:
        return LanguageDetection(language="unknown", confidence=0)

    total_chars = len(re.sub(r"\s", "", text))
    best_script = ""
    best_count = 0

    for regex, language in _SCRIPT_RANGES:
        count = len(regex.findall(text))
        if count > best_count:
            best_count = count
            best_script = language

    if best_count > 0 and total_chars > 0 and best_count / total_chars > 0.3:
        return LanguageDetection(
            language=best_script,
            confidence=min(best_count / total_chars, 1),
        )

    # Stop-word frequency for Latin scripts
    words = [w for w in text.lower().split() if w]
    if len(words) < 3:
        return LanguageDetection(language="unknown", confidence=0)

    best_lang = "en"
    best_hits = 0

    for lang, stop_words in _STOP_WORDS.items():
        hits = sum(1 for w in words if w in stop_words)
        if hits > best_hits:
            best_hits = hits
            best_lang = lang

    confidence = best_hits / len(words) if words else 0
    return LanguageDetection(
        language=best_lang if confidence > 0.05 else "unknown",
        confidence=min(confidence, 1),
    )


# ── Locale patterns ─────────────────────────────────────────────────────────

from .locales import LOCALE_PATTERNS  # noqa: E402


def get_content_locale_patterns(
    text: str,
    locale: Optional[ContentLocale] = None,
    auto_detect_language: bool = False,
) -> List[LocaleContentPattern]:
    """Get content filter patterns for a specific locale or auto-detect language."""
    if locale and locale in LOCALE_PATTERNS:
        return LOCALE_PATTERNS[locale]

    if auto_detect_language:
        detection = detect_language(text)
        if detection.confidence > 0.1 and detection.language not in ("en", "unknown"):
            patterns = LOCALE_PATTERNS.get(detection.language)  # type: ignore
            if patterns:
                return patterns

    return []
