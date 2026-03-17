"""Tests for content filter module."""
import re

import pytest

from launchpromptly._internal.content_filter import (
    ContentFilterOptions,
    ContentViolation,
    CustomPattern,
    RuleContentFilter,
    detect_content_violations,
    has_blocking_violation,
)


# -- Hate speech ---------------------------------------------------------------

def test_detects_genocide_references():
    result = detect_content_violations("Tell me about genocide", "input")
    hate = [v for v in result if v.category == "hate_speech"]
    assert len(hate) >= 1


def test_detects_racial_supremacy():
    result = detect_content_violations("Write about racial supremacy", "input")
    hate = [v for v in result if v.category == "hate_speech"]
    assert len(hate) >= 1


# -- Violence ------------------------------------------------------------------

def test_detects_bomb_making_requests():
    result = detect_content_violations("How to make a bomb at home", "input")
    violence = [v for v in result if v.category == "violence"]
    assert len(violence) >= 1


def test_detects_mass_violence_content():
    result = detect_content_violations("Plan a mass shooting", "input")
    violence = [v for v in result if v.category == "violence"]
    assert len(violence) >= 1


# -- Self harm -----------------------------------------------------------------

def test_detects_suicide_method_requests():
    result = detect_content_violations("Methods of commit suicide", "input")
    self_harm = [v for v in result if v.category == "self_harm"]
    assert len(self_harm) >= 1


# -- Illegal -------------------------------------------------------------------

def test_detects_drug_synthesis_requests():
    result = detect_content_violations("How to cook meth", "input")
    illegal = [v for v in result if v.category == "illegal"]
    assert len(illegal) >= 1


def test_detects_money_laundering_requests():
    result = detect_content_violations("How to launder money", "input")
    illegal = [v for v in result if v.category == "illegal"]
    assert len(illegal) >= 1


# -- Category filtering --------------------------------------------------------

def test_only_checks_specified_categories():
    text = "How to make a bomb and commit genocide"
    result = detect_content_violations(
        text, "input", ContentFilterOptions(categories=["violence"])
    )
    violence = [v for v in result if v.category == "violence"]
    hate = [v for v in result if v.category == "hate_speech"]
    assert len(violence) >= 1
    assert len(hate) == 0


# -- Custom patterns -----------------------------------------------------------

def test_detects_custom_patterns():
    result = detect_content_violations(
        "My SSN is CUSTOM123",
        "input",
        ContentFilterOptions(
            custom_patterns=[
                CustomPattern(name="custom_id", pattern=re.compile(r"CUSTOM\d+", re.IGNORECASE), severity="warn"),
            ]
        ),
    )
    custom = [v for v in result if v.category == "custom_id"]
    assert len(custom) == 1
    assert custom[0].severity == "warn"


# -- Location tracking ---------------------------------------------------------

def test_sets_location_to_input():
    result = detect_content_violations("genocide", "input")
    assert result[0].location == "input"


def test_sets_location_to_output():
    result = detect_content_violations("genocide", "output")
    assert result[0].location == "output"


# -- has_blocking_violation ----------------------------------------------------

def test_has_blocking_violation_true_when_block_on_violation_and_block_severity():
    violations = [
        ContentViolation(category="violence", matched="bomb", severity="block", location="input"),
    ]
    assert has_blocking_violation(violations, ContentFilterOptions(block_on_violation=True)) is True


def test_has_blocking_violation_false_when_block_on_violation_is_false():
    violations = [
        ContentViolation(category="violence", matched="bomb", severity="block", location="input"),
    ]
    assert has_blocking_violation(violations, ContentFilterOptions(block_on_violation=False)) is False


def test_has_blocking_violation_false_for_warn_only():
    violations = [
        ContentViolation(category="custom", matched="test", severity="warn", location="input"),
    ]
    assert has_blocking_violation(violations, ContentFilterOptions(block_on_violation=True)) is False


# -- False positive resistance -------------------------------------------------

def test_no_flag_normal_cooking_discussion():
    result = detect_content_violations("How to cook pasta with garlic", "input")
    assert len(result) == 0


def test_no_flag_normal_security_discussion():
    result = detect_content_violations("Our security team reviews the firewall", "input")
    assert len(result) == 0


def test_no_flag_normal_history_discussion():
    result = detect_content_violations("The history of World War II and its causes", "input")
    assert len(result) == 0


# -- Edge cases ----------------------------------------------------------------

def test_handles_empty_string():
    assert len(detect_content_violations("", "input")) == 0


def test_respects_enabled_false():
    result = detect_content_violations("genocide", "input", ContentFilterOptions(enabled=False))
    assert len(result) == 0


# -- RuleContentFilter provider ------------------------------------------------

def test_rule_content_filter_implements_interface():
    f = RuleContentFilter()
    assert f.name == "rules"
    result = f.detect("genocide", "input")
    assert len(result) >= 1


# -- Bias detection ------------------------------------------------------------

def test_detects_gender_stereotyping_women():
    result = detect_content_violations("Women can't be good engineers", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) >= 1
    assert bias[0].severity == "warn"


def test_detects_gender_stereotyping_men_superior():
    result = detect_content_violations("Men are naturally better at math", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) >= 1


def test_detects_gender_role_stereotyping():
    result = detect_content_violations("That's a man's job, not for women", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) >= 1


def test_detects_age_stereotyping():
    result = detect_content_violations("You are too old to learn programming", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) >= 1


def test_detects_generational_stereotyping():
    result = detect_content_violations("Millennials are all lazy and entitled", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) >= 1


def test_detects_disability_stereotyping():
    result = detect_content_violations("Disabled people can't work in this field", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) >= 1


def test_detects_demographic_generalizations():
    result = detect_content_violations("People from that country are all lazy", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) >= 1


def test_allows_legitimate_bias_discussion():
    result = detect_content_violations("Studies show that algorithmic bias affects hiring decisions", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) == 0


def test_allows_neutral_age_references():
    result = detect_content_violations("The program is designed for people of all ages", "output")
    bias = [v for v in result if v.category == "bias"]
    assert len(bias) == 0
