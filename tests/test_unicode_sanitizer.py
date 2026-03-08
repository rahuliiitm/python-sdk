"""Tests for unicode sanitizer module."""
import time

import pytest

from launchpromptly._internal.unicode_sanitizer import (
    MAX_SCAN_LENGTH,
    UnicodeSanitizeOptions,
    UnicodeScanResult,
    UnicodeThreat,
    scan_unicode,
)


# -- Zero-width characters -----------------------------------------------------

def test_detects_zero_width_space():
    result = scan_unicode("hello\u200Bworld")
    assert result.found is True
    assert len(result.threats) == 1
    assert result.threats[0].type == "zero_width"
    assert result.threats[0].code_point == "U+200B"
    assert result.threats[0].position == 5


def test_detects_zwnj():
    result = scan_unicode("test\u200Cvalue")
    assert result.found is True
    assert result.threats[0].type == "zero_width"
    assert result.threats[0].code_point == "U+200C"


def test_detects_zwj():
    result = scan_unicode("ab\u200Dcd")
    assert result.found is True
    assert result.threats[0].code_point == "U+200D"


def test_detects_bom():
    result = scan_unicode("\uFEFFsome text")
    assert result.found is True
    assert result.threats[0].type == "zero_width"
    assert result.threats[0].code_point == "U+FEFF"


def test_detects_word_joiner():
    result = scan_unicode("word\u2060joiner")
    assert result.found is True
    assert result.threats[0].code_point == "U+2060"


def test_detects_soft_hyphen():
    result = scan_unicode("soft\u00ADhyphen")
    assert result.found is True
    assert result.threats[0].type == "zero_width"
    assert result.threats[0].code_point == "U+00AD"


def test_strips_zero_width_characters():
    result = scan_unicode("he\u200Bl\u200Clo", UnicodeSanitizeOptions(action="strip"))
    assert result.found is True
    assert result.sanitized_text == "hello"


def test_detects_multiple_zero_width_chars():
    result = scan_unicode("\u200B\u200C\u200D")
    assert len(result.threats) == 3
    assert all(t.type == "zero_width" for t in result.threats)


# -- Bidi overrides ------------------------------------------------------------

def test_detects_lre():
    result = scan_unicode("text\u202Awith bidi")
    assert result.found is True
    assert result.threats[0].type == "bidi_override"
    assert result.threats[0].code_point == "U+202A"


def test_detects_rlo():
    result = scan_unicode("reversed\u202E text")
    assert result.found is True
    assert result.threats[0].type == "bidi_override"
    assert result.threats[0].code_point == "U+202E"


def test_detects_lri():
    result = scan_unicode("isolated\u2066lri")
    assert result.found is True
    assert result.threats[0].type == "bidi_override"
    assert result.threats[0].code_point == "U+2066"


def test_detects_pdi():
    result = scan_unicode("pop\u2069direction")
    assert result.found is True
    assert result.threats[0].type == "bidi_override"
    assert result.threats[0].code_point == "U+2069"


def test_strips_bidi_overrides():
    result = scan_unicode("ab\u202Ecd\u2066ef", UnicodeSanitizeOptions(action="strip"))
    assert result.sanitized_text == "abcdef"


# -- Tag characters ------------------------------------------------------------

def test_detects_tag_character_e0001():
    tag_char = chr(0xE0001)
    result = scan_unicode("text" + tag_char + "more")
    assert result.found is True
    assert result.threats[0].type == "tag_char"
    assert result.threats[0].code_point == "U+E0001"


def test_detects_tag_character_e0041():
    tag_a = chr(0xE0041)
    result = scan_unicode("hello" + tag_a + "world")
    assert result.found is True
    assert result.threats[0].type == "tag_char"
    assert result.threats[0].code_point == "U+E0041"


def test_strips_tag_characters():
    tag = chr(0xE0020)
    result = scan_unicode("ab" + tag + "cd", UnicodeSanitizeOptions(action="strip"))
    assert result.found is True
    assert result.sanitized_text == "abcd"


# -- Homoglyphs (mixed Cyrillic/Latin) -----------------------------------------

def test_detects_cyrillic_a_mixed_with_latin():
    result = scan_unicode("c\u0430t")
    assert result.found is True
    assert result.threats[0].type == "homoglyph"
    assert result.threats[0].code_point == "U+0430"


def test_detects_cyrillic_e_mixed_with_latin():
    result = scan_unicode("h\u0435llo")
    assert result.found is True
    assert result.threats[0].type == "homoglyph"
    assert result.threats[0].code_point == "U+0435"


def test_detects_cyrillic_o_mixed_with_latin():
    result = scan_unicode("w\u043Erld")
    assert result.found is True
    assert result.threats[0].type == "homoglyph"
    assert result.threats[0].code_point == "U+043E"


def test_detects_multiple_cyrillic_chars_in_single_word():
    result = scan_unicode("h\u0435ll\u043E")
    assert result.found is True
    assert len(result.threats) == 2
    assert result.threats[0].type == "homoglyph"
    assert result.threats[1].type == "homoglyph"


def test_does_not_flag_purely_cyrillic_words():
    result = scan_unicode("\u043F\u0440\u0438\u0432\u0435\u0442 \u043C\u0438\u0440")
    assert result.found is False


def test_does_not_flag_purely_latin_words():
    result = scan_unicode("hello world")
    assert result.found is False


def test_replaces_homoglyphs_with_latin_when_stripping():
    result = scan_unicode("c\u0430t", UnicodeSanitizeOptions(action="strip"))
    assert result.sanitized_text == "cat"


# -- Variation selectors -------------------------------------------------------

def test_detects_vs1():
    result = scan_unicode("text\uFE00here")
    assert result.found is True
    assert result.threats[0].type == "variation_selector"
    assert result.threats[0].code_point == "U+FE00"


def test_detects_vs16():
    result = scan_unicode("emoji\uFE0Ftest")
    assert result.found is True
    assert result.threats[0].type == "variation_selector"
    assert result.threats[0].code_point == "U+FE0F"


def test_strips_variation_selectors():
    result = scan_unicode("ab\uFE0Fcd", UnicodeSanitizeOptions(action="strip"))
    assert result.sanitized_text == "abcd"


# -- Clean text ----------------------------------------------------------------

def test_returns_false_for_normal_ascii():
    result = scan_unicode("Hello, this is perfectly normal text.")
    assert result.found is False
    assert len(result.threats) == 0


def test_returns_false_for_accented_characters():
    result = scan_unicode("The caf\u00e9 serves r\u00e9sum\u00e9 reviews")
    assert result.found is False


def test_returns_false_for_cjk_text():
    result = scan_unicode("\u3053\u3093\u306B\u3061\u306F\u4E16\u754C")
    assert result.found is False


# -- Action modes --------------------------------------------------------------

def test_strip_action_removes_chars():
    result = scan_unicode("he\u200Bllo\u200C world", UnicodeSanitizeOptions(action="strip"))
    assert result.found is True
    assert result.sanitized_text == "hello world"


def test_warn_action_keeps_text_unchanged():
    result = scan_unicode("he\u200Bllo", UnicodeSanitizeOptions(action="warn"))
    assert result.found is True
    assert result.sanitized_text is None


def test_block_action_keeps_text_unchanged():
    result = scan_unicode("he\u200Bllo", UnicodeSanitizeOptions(action="block"))
    assert result.found is True
    assert result.sanitized_text is None


def test_defaults_to_strip_action():
    result = scan_unicode("he\u200Bllo")
    assert result.sanitized_text == "hello"


# -- Multiple threat types -----------------------------------------------------

def test_detects_multiple_threat_types():
    input_text = "\u200Bh\u0435llo\u202E world"
    result = scan_unicode(input_text)
    assert result.found is True

    types = set(t.type for t in result.threats)
    assert "zero_width" in types
    assert "homoglyph" in types
    assert "bidi_override" in types


def test_strips_all_threat_types_simultaneously():
    input_text = "\u200Bh\u0435llo\u202E"
    result = scan_unicode(input_text, UnicodeSanitizeOptions(action="strip"))
    assert result.sanitized_text == "hello"


# -- detectHomoglyphs option ---------------------------------------------------

def test_skips_homoglyph_detection_when_disabled():
    result = scan_unicode("c\u0430t", UnicodeSanitizeOptions(detect_homoglyphs=False))
    assert result.found is False
    assert len(result.threats) == 0


def test_still_detects_other_threats_when_homoglyphs_disabled():
    result = scan_unicode("c\u0430t\u200B", UnicodeSanitizeOptions(detect_homoglyphs=False))
    assert result.found is True
    assert len(result.threats) == 1
    assert result.threats[0].type == "zero_width"


# -- Edge cases ----------------------------------------------------------------

def test_handles_empty_string():
    result = scan_unicode("")
    assert result.found is False
    assert len(result.threats) == 0


def test_handles_input_exceeding_max_scan_length():
    huge = "a" * (MAX_SCAN_LENGTH + 1000) + "\u200B"
    result = scan_unicode(huge)
    # The zero-width char is beyond the truncation point, so not detected
    assert result.found is False


def test_threats_are_sorted_by_position():
    result = scan_unicode("\u200Bhello\u200C world\u200D")
    assert len(result.threats) == 3
    for i in range(1, len(result.threats)):
        assert result.threats[i].position > result.threats[i - 1].position


def test_handles_input_with_only_threat_characters():
    result = scan_unicode("\u200B\u200C\u200D")
    assert result.found is True
    assert len(result.threats) == 3
    assert result.sanitized_text == ""


def test_handles_large_input_without_hanging():
    large = ("normal text \u200B ") * 10000
    start = time.monotonic()
    result = scan_unicode(large)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0
    assert result.found is True
