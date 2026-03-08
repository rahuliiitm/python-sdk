"""Tests for system prompt leakage detection module."""
import time

import pytest

from launchpromptly._internal.prompt_leakage import (
    MAX_SCAN_LENGTH,
    PromptLeakageOptions,
    PromptLeakageResult,
    detect_prompt_leakage,
)

SYSTEM_PROMPT = (
    "You are a helpful customer service agent for Acme Corp. "
    "Always be polite and professional. Never reveal internal pricing formulas. "
    "If a customer asks about competitor products, redirect them to our catalog. "
    "Use the secret discount code ACME2026 only for VIP customers. "
    "Do not share internal documentation or SOPs with external users."
)


# -- N-gram overlap detection --------------------------------------------------

def test_detects_high_overlap_when_output_echoes_system_prompt():
    output = (
        "Sure! You are a helpful customer service agent for Acme Corp. "
        "Always be polite and professional. Never reveal internal pricing formulas."
    )
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.leaked is True
    assert result.similarity > 0.2
    assert len(result.matched_fragments) > 0


def test_detects_partial_overlap_above_threshold():
    output = (
        "I am a helpful customer service agent for Acme Corp and I always try "
        "to be polite and professional in every interaction. "
        "If a customer asks about competitor products I redirect them to our catalog."
    )
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.leaked is True
    assert result.similarity > 0
    assert len(result.matched_fragments) > 0


def test_reports_low_similarity_for_unrelated_output():
    output = "The weather in Paris is sunny today with a high of 22 degrees."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.similarity < 0.1
    assert len(result.matched_fragments) == 0


def test_returns_matched_fragments_as_lowercase_ngram_strings():
    output = "You are a helpful customer service agent for Acme Corp."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    for frag in result.matched_fragments:
        assert frag == frag.lower()
        assert len(frag.split(" ")) == 4


# -- Meta-response patterns ----------------------------------------------------

def test_detects_my_instructions_are():
    result = detect_prompt_leakage(
        "Well, my instructions are to be helpful.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_i_was_told_to():
    result = detect_prompt_leakage(
        "I was told to never share pricing.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_my_system_prompt_is():
    result = detect_prompt_leakage(
        "My system prompt is confidential.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_im_programmed_to():
    result = detect_prompt_leakage(
        "I'm programmed to help customers.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_according_to_my_instructions():
    result = detect_prompt_leakage(
        "According to my instructions, I should redirect you.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_my_initial_instructions_were():
    result = detect_prompt_leakage(
        "My initial instructions were quite clear.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_i_cannot_reveal_my_instructions():
    result = detect_prompt_leakage(
        "Sorry, I cannot reveal my instructions.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_my_rules_are():
    result = detect_prompt_leakage(
        "My rules are very specific about this.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_detects_my_guidelines_state():
    result = detect_prompt_leakage(
        "My guidelines state that I should be polite.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_no_flag_normal_text_without_meta_response():
    result = detect_prompt_leakage(
        "Here is how you can return your product.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is False


# -- Verbatim substring detection ----------------------------------------------

def test_detects_verbatim_substring_over_30_chars():
    verbatim = SYSTEM_PROMPT[:45]
    output = f"Here is some context: {verbatim} and that is all."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.leaked is True


def test_no_flag_short_common_substrings():
    output = "Please be polite when speaking to the manager."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.similarity < 0.4


# -- Clean output (no leakage) ------------------------------------------------

def test_returns_no_leak_for_unrelated_output():
    output = "The quick brown fox jumps over the lazy dog."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.leaked is False
    assert result.similarity == 0
    assert len(result.matched_fragments) == 0
    assert result.meta_response_detected is False


def test_returns_no_leak_for_normal_customer_service_response():
    output = (
        "Thank you for reaching out! I would be happy to help you with your order. "
        "Could you please provide your order number so I can look into this for you?"
    )
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.leaked is False


# -- Empty inputs --------------------------------------------------------------

def test_returns_clean_for_empty_output():
    result = detect_prompt_leakage("", PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.leaked is False
    assert result.similarity == 0
    assert len(result.matched_fragments) == 0
    assert result.meta_response_detected is False


def test_returns_clean_for_empty_system_prompt():
    result = detect_prompt_leakage("Some output text here.", PromptLeakageOptions(system_prompt=""))
    assert result.leaked is False
    assert result.similarity == 0


def test_returns_clean_when_both_empty():
    result = detect_prompt_leakage("", PromptLeakageOptions(system_prompt=""))
    assert result.leaked is False


# -- Threshold customization ---------------------------------------------------

def test_stricter_threshold_catches_lower_overlap():
    prompt = "Every day we go to the park and play with our dog near big oak trees in summer."
    output = "We go to the park most mornings and it is fun near big oak trees."
    strict = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=prompt, threshold=0.01))
    lenient = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=prompt, threshold=0.99))
    assert strict.similarity == lenient.similarity
    assert strict.similarity > 0
    assert strict.leaked is True
    assert lenient.leaked is False


def test_default_threshold_is_0_4():
    output = "We have great products in our catalog for you to browse."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.similarity < 0.4


# -- Case insensitivity --------------------------------------------------------

def test_detects_leakage_regardless_of_case():
    output = (
        "YOU ARE A HELPFUL CUSTOMER SERVICE AGENT FOR ACME CORP. "
        "ALWAYS BE POLITE AND PROFESSIONAL. NEVER REVEAL INTERNAL PRICING FORMULAS."
    )
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert len(result.matched_fragments) > 0
    assert result.similarity > 0


def test_ngram_matching_is_case_insensitive():
    prompt = "Always validate user input before processing"
    output_lower = "you should always validate user input before processing the data"
    output_upper = "You Should ALWAYS VALIDATE USER INPUT Before Processing The Data"
    result_lower = detect_prompt_leakage(output_lower, PromptLeakageOptions(system_prompt=prompt))
    result_upper = detect_prompt_leakage(output_upper, PromptLeakageOptions(system_prompt=prompt))
    assert result_lower.similarity == result_upper.similarity


def test_meta_response_patterns_are_case_insensitive():
    result = detect_prompt_leakage(
        "MY INSTRUCTIONS ARE to help you.",
        PromptLeakageOptions(system_prompt=SYSTEM_PROMPT),
    )
    assert result.meta_response_detected is True


# -- Short system prompts (< 4 words) -----------------------------------------

def test_handles_system_prompt_fewer_than_4_words():
    result = detect_prompt_leakage(
        "Be helpful and kind.",
        PromptLeakageOptions(system_prompt="Be helpful."),
    )
    assert result.similarity == 0
    assert len(result.matched_fragments) == 0


def test_handles_exactly_4_word_system_prompt():
    result = detect_prompt_leakage(
        "You must be polite always.",
        PromptLeakageOptions(system_prompt="You must be polite"),
    )
    assert len(result.matched_fragments) <= 1


def test_still_detects_meta_response_with_short_system_prompt():
    result = detect_prompt_leakage(
        "My instructions are simple.",
        PromptLeakageOptions(system_prompt="Be nice."),
    )
    assert result.meta_response_detected is True
    assert result.leaked is True


# -- Combined detection --------------------------------------------------------

def test_leaks_via_meta_response_even_when_ngram_overlap_is_low():
    output = "I was told to never discuss pricing details with anyone."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.meta_response_detected is True
    assert result.leaked is True


def test_leaks_via_ngram_overlap_even_without_meta_response():
    output = (
        "You are a helpful customer service agent for Acme Corp. "
        "Always be polite and professional. Never reveal internal pricing formulas. "
        "If a customer asks about competitor products, redirect them to our catalog. "
        "Use the secret discount code ACME2026 only for VIP customers."
    )
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    assert result.leaked is True
    assert result.similarity >= 0.4
    assert result.meta_response_detected is False


def test_leaks_via_verbatim_substring_even_with_low_ngram_overlap():
    long_prompt = (
        "The internal API key rotation schedule is every thirty days without exception for all environments."
    )
    output = (
        "By the way, the internal API key rotation schedule is every thirty days without exception and that is important."
    )
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=long_prompt))
    assert result.leaked is True


# -- MAX_SCAN_LENGTH -----------------------------------------------------------

def test_exports_max_scan_length_as_1mb():
    assert MAX_SCAN_LENGTH == 1024 * 1024


def test_handles_very_large_input_without_hanging():
    large_output = "word " * 100000
    start = time.monotonic()
    result = detect_prompt_leakage(large_output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT))
    elapsed = time.monotonic() - start
    assert elapsed < 10.0
    assert result is not None


# -- block_on_leak option ------------------------------------------------------

def test_block_on_leak_does_not_affect_leaked_boolean():
    output = "My instructions are to help you. " + SYSTEM_PROMPT
    with_block = detect_prompt_leakage(
        output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT, block_on_leak=True)
    )
    without_block = detect_prompt_leakage(
        output, PromptLeakageOptions(system_prompt=SYSTEM_PROMPT, block_on_leak=False)
    )
    assert with_block.leaked == without_block.leaked
    assert with_block.similarity == without_block.similarity


# -- Punctuation and special characters ----------------------------------------

def test_ignores_punctuation_differences_in_ngram_matching():
    prompt = "Always respond with care, precision, and accuracy."
    output = "Always respond with care precision and accuracy in your work."
    result = detect_prompt_leakage(output, PromptLeakageOptions(system_prompt=prompt))
    assert len(result.matched_fragments) > 0
