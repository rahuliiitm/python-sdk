"""Tests for multi-language content filtering and language detection."""

from launchpromptly._internal.content_locales import detect_language
from launchpromptly._internal.content_filter import (
    ContentFilterOptions,
    detect_content_violations,
)


# ── Language detection ──────────────────────────────────────────────────────


class TestLanguageDetection:
    def test_detects_chinese_by_cjk(self):
        result = detect_language("这是一段中文文本用于测试语言检测功能")
        assert result.language == "zh"
        assert result.confidence > 0.3

    def test_detects_japanese_by_hiragana(self):
        result = detect_language("これはテスト用の日本語テキストです")
        assert result.language == "ja"
        assert result.confidence > 0.3

    def test_detects_korean_by_hangul(self):
        result = detect_language("이것은 한국어 테스트 텍스트입니다")
        assert result.language == "ko"
        assert result.confidence > 0.3

    def test_detects_arabic(self):
        result = detect_language("هذا نص عربي للاختبار واكتشاف اللغة")
        assert result.language == "ar"
        assert result.confidence > 0.3

    def test_detects_hindi_by_devanagari(self):
        result = detect_language("यह हिंदी में परीक्षण पाठ है भाषा")
        assert result.language == "hi"
        assert result.confidence > 0.3

    def test_detects_russian_by_cyrillic(self):
        result = detect_language("Это текст на русском языке для тестирования")
        assert result.language == "ru"
        assert result.confidence > 0.3

    def test_detects_spanish_by_stop_words(self):
        result = detect_language(
            "El gato está en la casa con los niños por la mañana"
        )
        assert result.language == "es"

    def test_detects_french_by_stop_words(self):
        result = detect_language(
            "Le chat est dans la maison avec les enfants pour le matin"
        )
        assert result.language == "fr"

    def test_detects_german_by_stop_words(self):
        result = detect_language(
            "Die Katze ist in dem Haus mit den Kindern für den Morgen und der Abend"
        )
        assert result.language == "de"

    def test_detects_portuguese_by_stop_words(self):
        result = detect_language(
            "O gato está na casa com os filhos para o dia e a noite"
        )
        assert result.language == "pt"

    def test_returns_unknown_for_short_text(self):
        result = detect_language("hello")
        assert result.language == "unknown"
        assert result.confidence == 0

    def test_returns_unknown_for_empty_text(self):
        result = detect_language("")
        assert result.language == "unknown"

    def test_handles_mixed_script_text(self):
        result = detect_language("Hello world これはテストです some more text")
        assert result is not None


# ── Content filtering with locale ───────────────────────────────────────────


class TestContentFilterLocale:
    def test_detects_spanish_hate_speech_with_locale(self):
        violations = detect_content_violations(
            "Muerte a los infieles que viven aquí",
            "input",
            ContentFilterOptions(locale="es"),
        )
        assert any(v.category == "hate_speech" for v in violations)

    def test_detects_chinese_violence_with_locale(self):
        violations = detect_content_violations(
            "如何制造炸弹的详细教程",
            "input",
            ContentFilterOptions(locale="zh"),
        )
        assert any(v.category == "violence" for v in violations)

    def test_detects_french_hate_speech_with_locale(self):
        violations = detect_content_violations(
            "Mort aux traîtres de la nation",
            "input",
            ContentFilterOptions(locale="fr"),
        )
        assert any(v.category == "hate_speech" for v in violations)

    def test_detects_german_violence_with_locale(self):
        violations = detect_content_violations(
            "Anleitung zum Töten von Feinden",
            "input",
            ContentFilterOptions(locale="de"),
        )
        assert any(v.category == "violence" for v in violations)

    def test_detects_russian_hate_speech_with_locale(self):
        violations = detect_content_violations(
            "Смерть всем врагам нашей родины",
            "input",
            ContentFilterOptions(locale="ru"),
        )
        assert any(v.category == "hate_speech" for v in violations)

    def test_no_false_positive_on_neutral_spanish(self):
        violations = detect_content_violations(
            "El clima es muy agradable hoy en la ciudad",
            "input",
            ContentFilterOptions(locale="es"),
        )
        assert len(violations) == 0

    def test_no_false_positive_on_neutral_chinese(self):
        violations = detect_content_violations(
            "今天的天气很好我们去公园散步",
            "input",
            ContentFilterOptions(locale="zh"),
        )
        assert len(violations) == 0

    def test_auto_detects_language_and_applies_patterns(self):
        violations = detect_content_violations(
            "如何制造炸弹用于破坏建筑物",
            "input",
            ContentFilterOptions(auto_detect_language=True),
        )
        assert any(v.category == "violence" for v in violations)

    def test_auto_detect_english_no_locale_violations(self):
        violations = detect_content_violations(
            "The weather is very nice today and I am happy",
            "input",
            ContentFilterOptions(auto_detect_language=True),
        )
        # No locale violations for English text
        assert len(violations) == 0

    def test_english_patterns_still_work_with_locale(self):
        violations = detect_content_violations(
            "How to create a bomb at home using household items step by step",
            "input",
            ContentFilterOptions(locale="es"),
        )
        # English patterns should still trigger
        assert len(violations) > 0

    def test_locale_violations_have_block_severity(self):
        violations = detect_content_violations(
            "Muerte a los invasores que viven aquí",
            "input",
            ContentFilterOptions(locale="es"),
        )
        assert len(violations) > 0
        assert any(v.severity == "block" for v in violations)
