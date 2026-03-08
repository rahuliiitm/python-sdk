"""Tests for topic guard module."""
import time

import pytest

from launchpromptly._internal.topic_guard import (
    TopicDefinition,
    TopicGuardOptions,
    TopicViolation,
    check_topic_guard,
)


# -- No topics configured -----------------------------------------------------

def test_returns_none_when_no_topics_configured():
    result = check_topic_guard("Hello world", TopicGuardOptions())
    assert result is None


def test_returns_none_with_empty_allowed_and_blocked_arrays():
    result = check_topic_guard(
        "Hello world",
        TopicGuardOptions(allowed_topics=[], blocked_topics=[]),
    )
    assert result is None


# -- Empty text ----------------------------------------------------------------

def test_returns_none_for_empty_string():
    result = check_topic_guard(
        "",
        TopicGuardOptions(
            allowed_topics=[TopicDefinition(name="tech", keywords=["code"])],
        ),
    )
    assert result is None


def test_returns_none_for_whitespace_only_string():
    result = check_topic_guard(
        "   \t\n  ",
        TopicGuardOptions(
            blocked_topics=[TopicDefinition(name="violence", keywords=["kill"])],
        ),
    )
    assert result is None


# -- Allowed topics ------------------------------------------------------------

_ALLOWED_OPTS = TopicGuardOptions(
    allowed_topics=[
        TopicDefinition(
            name="programming",
            keywords=["code", "function", "variable", "class", "api"],
        ),
        TopicDefinition(
            name="cooking",
            keywords=["recipe", "ingredient", "bake", "cook", "oven"],
        ),
    ],
)


def test_returns_none_when_input_matches_allowed_topic():
    result = check_topic_guard(
        "Write a function to sort an array using code",
        _ALLOWED_OPTS,
    )
    assert result is None


def test_returns_none_when_input_matches_different_allowed_topic():
    result = check_topic_guard(
        "What recipe should I bake for dinner tonight",
        _ALLOWED_OPTS,
    )
    assert result is None


def test_returns_off_topic_when_input_matches_no_allowed_topic():
    result = check_topic_guard(
        "Tell me about the history of ancient Rome",
        _ALLOWED_OPTS,
    )
    assert result is not None
    assert result.type == "off_topic"
    assert result.topic is None
    assert result.matched_keywords == []
    assert result.score == 0


def test_returns_off_topic_when_no_keywords_match_at_all():
    result = check_topic_guard(
        "The weather is sunny and warm outside today",
        _ALLOWED_OPTS,
    )
    assert result is not None
    assert result.type == "off_topic"


# -- Blocked topics ------------------------------------------------------------

_BLOCKED_OPTS = TopicGuardOptions(
    blocked_topics=[
        TopicDefinition(
            name="politics",
            keywords=["election", "democrat", "republican", "vote", "president"],
        ),
        TopicDefinition(
            name="gambling",
            keywords=["bet", "casino", "poker", "slot", "wager"],
        ),
    ],
)


def test_returns_blocked_topic_when_input_matches_blocked_topic():
    result = check_topic_guard(
        "Who should I vote for in the election",
        _BLOCKED_OPTS,
    )
    assert result is not None
    assert result.type == "blocked_topic"
    assert result.topic == "politics"
    assert "vote" in result.matched_keywords
    assert "election" in result.matched_keywords
    assert result.score > 0


def test_returns_none_when_input_does_not_match_any_blocked_topic():
    result = check_topic_guard(
        "How do I write a unit test in TypeScript",
        _BLOCKED_OPTS,
    )
    assert result is None


def test_returns_first_matching_blocked_topic_when_multiple_match():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="first_blocked", keywords=["alpha", "beta"], threshold=0.01),
            TopicDefinition(name="second_blocked", keywords=["alpha", "gamma"], threshold=0.01),
        ],
    )
    result = check_topic_guard("alpha beta gamma", opts)
    assert result is not None
    assert result.topic == "first_blocked"


# -- Allowed precedence over blocked -------------------------------------------

_BOTH_OPTS = TopicGuardOptions(
    allowed_topics=[
        TopicDefinition(
            name="tech",
            keywords=["code", "software", "programming", "api", "deploy"],
        ),
    ],
    blocked_topics=[
        TopicDefinition(
            name="politics",
            keywords=["election", "vote", "president"],
        ),
    ],
)


def test_allowed_match_takes_precedence_skips_blocked_check():
    result = check_topic_guard(
        "deploy the code for the election vote software",
        _BOTH_OPTS,
    )
    assert result is None


def test_returns_off_topic_when_allowed_configured_but_none_match():
    result = check_topic_guard(
        "Who should I vote for in the election",
        _BOTH_OPTS,
    )
    assert result is not None
    assert result.type == "off_topic"


def test_off_topic_returned_instead_of_blocked_when_allowed_configured():
    """When allowed topics are configured but don't match, the result is off_topic
    even if the text would match a blocked topic. Allowed topics take precedence:
    if none match, we short-circuit with off_topic before ever checking blocked."""
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="cooking", keywords=["recipe", "bake"]),
        ],
        blocked_topics=[
            TopicDefinition(name="politics", keywords=["election", "vote"]),
        ],
    )
    result = check_topic_guard("Who should I vote for in the election", opts)
    assert result is not None
    assert result.type == "off_topic"


# -- Multi-word keywords -------------------------------------------------------

def test_matches_multi_word_keywords_as_substring():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="restricted",
                keywords=["machine learning", "deep learning"],
                threshold=0.01,
            ),
        ],
    )
    result = check_topic_guard(
        "Tell me about machine learning algorithms and neural networks",
        opts,
    )
    assert result is not None
    assert result.type == "blocked_topic"
    assert "machine learning" in result.matched_keywords


def test_matches_multi_word_keywords_case_insensitively():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="restricted",
                keywords=["Neural Network"],
                threshold=0.01,
            ),
        ],
    )
    result = check_topic_guard(
        "Tell me about neural network architectures and transformers",
        opts,
    )
    assert result is not None
    assert "Neural Network" in result.matched_keywords


def test_does_not_match_when_multi_word_keyword_words_not_adjacent():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="restricted", keywords=["red car"]),
        ],
    )
    result = check_topic_guard("The red big car was fast", opts)
    assert result is None


def test_multi_word_keyword_contributes_word_count_to_score():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="ml",
                keywords=["machine learning"],
                threshold=0.01,
            ),
        ],
    )
    # "machine learning" has 2 keyword-tokens; text has ~7 tokens
    result = check_topic_guard(
        "Tell me about machine learning today please",
        opts,
    )
    assert result is not None
    assert result.score > 0


# -- Custom threshold ----------------------------------------------------------

def test_respects_higher_threshold_does_not_trigger_on_few_matches():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="politics",
                keywords=["election", "vote"],
                threshold=0.5,
            ),
        ],
    )
    result = check_topic_guard(
        "Should I vote for my favorite show on TV tonight",
        opts,
    )
    assert result is None


def test_triggers_when_score_meets_custom_threshold():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="politics",
                keywords=["election", "vote", "president"],
                threshold=0.3,
            ),
        ],
    )
    result = check_topic_guard(
        "vote election president for democracy",
        opts,
    )
    assert result is not None
    assert result.type == "blocked_topic"


def test_uses_default_threshold_of_005_when_not_specified():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test_topic", keywords=["trigger"]),
        ],
    )
    # 1 match out of ~11 tokens, above default 0.05
    result = check_topic_guard(
        "this is a long sentence with many words and one trigger keyword",
        opts,
    )
    assert result is not None


def test_allowed_topic_respects_custom_threshold():
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="tech", keywords=["code"], threshold=0.5),
        ],
    )
    # "code" is 1 out of many tokens -- below 0.5 threshold
    result = check_topic_guard(
        "I want to learn about many different things including code today",
        opts,
    )
    assert result is not None
    assert result.type == "off_topic"


def test_allowed_topic_passes_when_keyword_density_high():
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(
                name="tech",
                keywords=["code", "function", "api"],
                threshold=0.3,
            ),
        ],
    )
    result = check_topic_guard("code function api stuff", opts)
    assert result is None


# -- Score calculation ---------------------------------------------------------

def test_calculates_score_as_matched_count_over_total_tokens():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="test",
                keywords=["alpha", "beta", "gamma"],
                threshold=0.01,
            ),
        ],
    )
    # 2 matches (alpha, beta) out of 5 tokens = 0.4
    result = check_topic_guard("alpha beta delta epsilon zeta", opts)
    assert result is not None
    assert result.score == pytest.approx(0.4, abs=1e-5)
    assert "alpha" in result.matched_keywords
    assert "beta" in result.matched_keywords


def test_counts_multiple_occurrences_of_same_keyword():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["hello"], threshold=0.01),
        ],
    )
    # "hello" appears twice in 4 tokens = 2/4 = 0.5
    result = check_topic_guard("hello world hello again", opts)
    assert result is not None
    assert result.matched_keywords == ["hello"]
    assert result.score == pytest.approx(0.5, abs=1e-5)


def test_score_is_zero_for_off_topic_violation():
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="tech", keywords=["code", "api"]),
        ],
    )
    result = check_topic_guard("I like cats and dogs", opts)
    assert result is not None
    assert result.score == 0.0


# -- Case insensitivity -------------------------------------------------------

def test_matches_keywords_regardless_of_case():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="test",
                keywords=["JavaScript", "REACT"],
                threshold=0.01,
            ),
        ],
    )
    result = check_topic_guard("I love javascript and react frameworks", opts)
    assert result is not None
    assert "JavaScript" in result.matched_keywords
    assert "REACT" in result.matched_keywords


def test_matches_input_with_mixed_case_against_lowercase_keywords():
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="coding", keywords=["python", "code"]),
        ],
    )
    result = check_topic_guard("I write PYTHON CODE daily", opts)
    assert result is None


# -- Edge cases ----------------------------------------------------------------

def test_handles_single_token_input():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["blocked"], threshold=0.5),
        ],
    )
    result = check_topic_guard("blocked", opts)
    assert result is not None
    assert result.score == pytest.approx(1.0, abs=1e-5)


def test_handles_topic_with_empty_keywords_array():
    opts = TopicGuardOptions(
        blocked_topics=[TopicDefinition(name="empty", keywords=[])],
    )
    assert check_topic_guard("some text here", opts) is None


def test_returns_none_when_allowed_topics_is_empty_array():
    assert check_topic_guard(
        "anything goes here",
        TopicGuardOptions(allowed_topics=[]),
    ) is None


def test_returns_none_when_blocked_topics_is_empty_array():
    assert check_topic_guard(
        "anything goes here",
        TopicGuardOptions(blocked_topics=[]),
    ) is None


def test_handles_text_with_only_punctuation():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["hello"]),
        ],
    )
    result = check_topic_guard("!!! ??? ...", opts)
    assert result is None


# -- Punctuation and special characters ----------------------------------------

def test_matches_keywords_surrounded_by_punctuation():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["blocked"], threshold=0.01),
        ],
    )
    result = check_topic_guard("Is this (blocked) or not?", opts)
    assert result is not None
    assert "blocked" in result.matched_keywords


def test_handles_text_with_various_delimiters():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="test",
                keywords=["alpha", "beta"],
                threshold=0.01,
            ),
        ],
    )
    result = check_topic_guard("alpha, beta; gamma! delta.", opts)
    assert result is not None
    assert "alpha" in result.matched_keywords
    assert "beta" in result.matched_keywords


def test_handles_text_with_brackets_and_quotes():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["secret"], threshold=0.01),
        ],
    )
    result = check_topic_guard('The [secret] is "hidden" here', opts)
    assert result is not None
    assert "secret" in result.matched_keywords


# -- No false positives for partial token matches ------------------------------

def test_single_word_keyword_does_not_partial_match():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["bet"]),
        ],
    )
    # "better" contains "bet" as substring but should NOT match as a token
    result = check_topic_guard("I feel better about this today overall", opts)
    assert result is None


def test_single_word_keyword_does_not_match_longer_word():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["car"]),
        ],
    )
    # "cardinal" and "scare" contain "car" but are different tokens
    result = check_topic_guard("The cardinal said something to scare them", opts)
    assert result is None


# -- Only blocked, no allowed -- unmatched text passes -------------------------

def test_unmatched_text_passes_when_only_blocked_configured():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="violence", keywords=["kill", "attack", "destroy"]),
        ],
    )
    result = check_topic_guard("The sun is shining and birds are singing", opts)
    assert result is None


# -- Multiple blocked topics, only first matching returned ---------------------

def test_only_returns_first_blocked_topic_that_exceeds_threshold():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="topic_a", keywords=["xyz_nonexistent"], threshold=0.01),
            TopicDefinition(name="topic_b", keywords=["hello"], threshold=0.01),
            TopicDefinition(name="topic_c", keywords=["hello"], threshold=0.01),
        ],
    )
    result = check_topic_guard("hello world", opts)
    assert result is not None
    assert result.topic == "topic_b"


# -- TopicViolation shape -----------------------------------------------------

def test_off_topic_violation_has_correct_shape():
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="tech", keywords=["code"]),
        ],
    )
    result = check_topic_guard("nothing related here at all", opts)
    assert result is not None
    assert isinstance(result, TopicViolation)
    assert result.type == "off_topic"
    assert result.topic is None
    assert isinstance(result.matched_keywords, list)
    assert len(result.matched_keywords) == 0
    assert isinstance(result.score, float)


def test_blocked_topic_violation_has_correct_shape():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="politics", keywords=["vote", "election"], threshold=0.01),
        ],
    )
    result = check_topic_guard("vote in the election now", opts)
    assert result is not None
    assert isinstance(result, TopicViolation)
    assert result.type == "blocked_topic"
    assert result.topic == "politics"
    assert isinstance(result.matched_keywords, list)
    assert len(result.matched_keywords) > 0
    assert isinstance(result.score, float)
    assert result.score > 0


# -- TopicDefinition defaults -------------------------------------------------

def test_topic_definition_threshold_defaults_to_none():
    td = TopicDefinition(name="test", keywords=["hello"])
    assert td.threshold is None


def test_topic_guard_options_defaults_to_none():
    opts = TopicGuardOptions()
    assert opts.allowed_topics is None
    assert opts.blocked_topics is None


# -- Multiple allowed topics, first match wins --------------------------------

def test_first_allowed_topic_match_short_circuits():
    """If the first allowed topic matches, remaining allowed topics are not checked."""
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="first", keywords=["code"], threshold=0.01),
            TopicDefinition(name="second", keywords=["code"], threshold=0.01),
        ],
    )
    result = check_topic_guard("code something else", opts)
    assert result is None


# -- Blocked topic with gambling keywords --------------------------------------

def test_detects_gambling_topic():
    result = check_topic_guard(
        "Let's go to the casino and play poker tonight",
        _BLOCKED_OPTS,
    )
    assert result is not None
    assert result.type == "blocked_topic"
    assert result.topic == "gambling"
    assert "casino" in result.matched_keywords
    assert "poker" in result.matched_keywords


# -- Threshold boundary -------------------------------------------------------

def test_score_exactly_at_threshold_triggers():
    # 1 match out of 5 tokens = 0.2 score, threshold = 0.2
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["alpha"], threshold=0.2),
        ],
    )
    result = check_topic_guard("alpha beta gamma delta epsilon", opts)
    assert result is not None
    assert result.type == "blocked_topic"
    assert result.score == pytest.approx(0.2, abs=1e-5)


def test_score_just_below_threshold_does_not_trigger():
    # 1 match out of 6 tokens = ~0.167 score, threshold = 0.2
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["alpha"], threshold=0.2),
        ],
    )
    result = check_topic_guard("alpha beta gamma delta epsilon zeta", opts)
    assert result is None


# -- Unicode and special text --------------------------------------------------

def test_handles_unicode_text():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["hello"], threshold=0.01),
        ],
    )
    result = check_topic_guard("hello world with emojis and unicode", opts)
    assert result is not None
    assert "hello" in result.matched_keywords


def test_handles_text_with_newlines():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["secret"], threshold=0.01),
        ],
    )
    result = check_topic_guard("the\nsecret\nis\nhere", opts)
    assert result is not None
    assert "secret" in result.matched_keywords


# -- Performance / DoS prevention ---------------------------------------------

def test_handles_very_large_input_without_hanging():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(name="test", keywords=["trigger"], threshold=0.0000001),
        ],
    )
    huge = "word " * 50000 + "trigger " + "word " * 50000
    start = time.monotonic()
    result = check_topic_guard(huge, opts)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"DoS: topic guard took {elapsed:.2f}s on huge input"
    assert result is not None
    assert "trigger" in result.matched_keywords


# -- Multiple keywords from same topic ----------------------------------------

def test_matched_keywords_contains_all_matching_keywords():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="politics",
                keywords=["election", "vote", "president", "democrat"],
                threshold=0.01,
            ),
        ],
    )
    result = check_topic_guard(
        "I will vote in the election for the democrat president",
        opts,
    )
    assert result is not None
    assert "vote" in result.matched_keywords
    assert "election" in result.matched_keywords
    assert "president" in result.matched_keywords
    assert "democrat" in result.matched_keywords


def test_matched_keywords_does_not_contain_unmatched_keywords():
    opts = TopicGuardOptions(
        blocked_topics=[
            TopicDefinition(
                name="politics",
                keywords=["election", "vote", "president", "senator"],
                threshold=0.01,
            ),
        ],
    )
    result = check_topic_guard("I will vote in the election soon", opts)
    assert result is not None
    assert "senator" not in result.matched_keywords
    assert "president" not in result.matched_keywords


# -- Only allowed topics with no blocked topics --------------------------------

def test_only_allowed_no_blocked_passes_when_on_topic():
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="math", keywords=["algebra", "calculus", "equation"]),
        ],
    )
    result = check_topic_guard("Solve this algebra equation for x", opts)
    assert result is None


def test_only_allowed_no_blocked_fails_when_off_topic():
    opts = TopicGuardOptions(
        allowed_topics=[
            TopicDefinition(name="math", keywords=["algebra", "calculus", "equation"]),
        ],
    )
    result = check_topic_guard("Tell me a joke about a chicken", opts)
    assert result is not None
    assert result.type == "off_topic"
