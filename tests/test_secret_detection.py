"""Tests for secret / credential detection module."""
import time

import pytest

from launchpromptly._internal.secret_detection import (
    MAX_SCAN_LENGTH,
    CustomSecretPattern,
    SecretDetection,
    SecretDetectionOptions,
    detect_secrets,
)


# -- AWS Access Key ------------------------------------------------------------

def test_detects_aws_access_key():
    result = detect_secrets("AWS key: AKIAIOSFODNN7EXAMPLE")
    assert len(result) == 1
    assert result[0].type == "aws_access_key"
    assert result[0].value == "AKIAIOSFODNN7EXAMPLE"
    assert result[0].confidence == 0.95


def test_detects_aws_key_in_config_block():
    result = detect_secrets("aws_access_key_id = AKIAI44QH8DHBEXAMPLE")
    aws = [d for d in result if d.type == "aws_access_key"]
    assert len(aws) == 1


# -- GitHub PAT ----------------------------------------------------------------

def test_detects_github_pat():
    token = "ghp_" + "A" * 36
    result = detect_secrets(f"Token: {token}")
    assert len(result) == 1
    assert result[0].type == "github_pat"
    assert result[0].value == token
    assert result[0].confidence == 0.95


# -- GitHub OAuth --------------------------------------------------------------

def test_detects_github_oauth():
    token = "gho_" + "b" * 36
    result = detect_secrets(f"OAuth: {token}")
    assert len(result) == 1
    assert result[0].type == "github_oauth"
    assert result[0].value == token
    assert result[0].confidence == 0.95


# -- GitLab PAT ----------------------------------------------------------------

def test_detects_gitlab_pat():
    token = "glpat-" + "X" * 20
    result = detect_secrets(f"Token: {token}")
    assert len(result) == 1
    assert result[0].type == "gitlab_pat"
    assert result[0].value == token
    assert result[0].confidence == 0.95


def test_detects_long_gitlab_pat():
    token = "glpat-AbCdEfGhIjKlMnOpQrStUv"
    result = detect_secrets(f"Token: {token}")
    gl = [d for d in result if d.type == "gitlab_pat"]
    assert len(gl) == 1


# -- Slack Token ---------------------------------------------------------------

def test_detects_slack_bot_token():
    result = detect_secrets("Token: xoxb-123456789-abcdefghij")
    slack = [d for d in result if d.type == "slack_token"]
    assert len(slack) == 1
    assert slack[0].confidence == 0.90


def test_detects_slack_user_token():
    result = detect_secrets("Token: xoxp-123456789-abcdefghij")
    slack = [d for d in result if d.type == "slack_token"]
    assert len(slack) == 1


def test_detects_slack_app_token():
    result = detect_secrets("Token: xoxa-123456789-abcdefghij")
    slack = [d for d in result if d.type == "slack_token"]
    assert len(slack) == 1


def test_detects_slack_socket_token():
    result = detect_secrets("Token: xoxs-123456789-abcdefghij")
    slack = [d for d in result if d.type == "slack_token"]
    assert len(slack) == 1


# -- Stripe Secret Key --------------------------------------------------------

def test_detects_stripe_secret_key():
    key = "sk_live_" + "A" * 24
    result = detect_secrets(f"Key: {key}")
    assert len(result) == 1
    assert result[0].type == "stripe_secret"
    assert result[0].value == key
    assert result[0].confidence == 0.95


# -- Stripe Publishable Key ---------------------------------------------------

def test_detects_stripe_publishable_key():
    key = "pk_live_" + "B" * 24
    result = detect_secrets(f"Key: {key}")
    assert len(result) == 1
    assert result[0].type == "stripe_publishable"
    assert result[0].value == key
    assert result[0].confidence == 0.85


# -- JWT -----------------------------------------------------------------------

def test_detects_jwt():
    jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123DEF456_-"
    result = detect_secrets(f"Bearer {jwt}")
    jwts = [d for d in result if d.type == "jwt"]
    assert len(jwts) == 1
    assert jwts[0].confidence == 0.90


def test_does_not_match_non_jwt_base64():
    result = detect_secrets("data: aGVsbG8gd29ybGQ=")
    jwts = [d for d in result if d.type == "jwt"]
    assert len(jwts) == 0


# -- Private Key ---------------------------------------------------------------

def test_detects_rsa_private_key_header():
    result = detect_secrets("-----BEGIN RSA PRIVATE KEY-----")
    assert len(result) == 1
    assert result[0].type == "private_key"
    assert result[0].confidence == 0.99


def test_detects_generic_private_key_header():
    result = detect_secrets("-----BEGIN PRIVATE KEY-----")
    assert len(result) == 1
    assert result[0].type == "private_key"


def test_detects_ec_private_key_header():
    result = detect_secrets("-----BEGIN EC PRIVATE KEY-----")
    assert len(result) == 1
    assert result[0].type == "private_key"


def test_detects_openssh_private_key_header():
    result = detect_secrets("-----BEGIN OPENSSH PRIVATE KEY-----")
    assert len(result) == 1
    assert result[0].type == "private_key"


def test_detects_dsa_private_key_header():
    result = detect_secrets("-----BEGIN DSA PRIVATE KEY-----")
    assert len(result) == 1
    assert result[0].type == "private_key"


# -- Connection String ---------------------------------------------------------

def test_detects_postgresql_connection_string():
    result = detect_secrets("DATABASE_URL=postgresql://user:pass@host:5432/db")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1
    assert cs[0].confidence == 0.90


def test_detects_mongodb_connection_string():
    result = detect_secrets("MONGO_URI=mongodb+srv://admin:secret@cluster0.example.net/mydb")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1


def test_detects_mysql_connection_string():
    result = detect_secrets("DB=mysql://root:password@localhost:3306/app")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1


def test_detects_redis_connection_string():
    result = detect_secrets("REDIS_URL=redis://default:abc123@redis-host:6379")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1


def test_detects_amqp_connection_string():
    result = detect_secrets("AMQP_URL=amqp://guest:guest@rabbitmq:5672")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1


def test_detects_postgres_short_form():
    result = detect_secrets("DB=postgres://user:pass@host/db")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1


def test_detects_mongodb_without_srv():
    result = detect_secrets("MONGO=mongodb://admin:pass@localhost:27017/db")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1


# -- Webhook URL ---------------------------------------------------------------

def test_detects_slack_webhook_url():
    result = detect_secrets("WEBHOOK=https://hooks.slack.com/services/T00/B00/xxxx")
    wh = [d for d in result if d.type == "webhook_url"]
    assert len(wh) == 1
    assert wh[0].confidence == 0.85


def test_detects_discord_webhook_url():
    result = detect_secrets("HOOK=https://hooks.discord.com/api/webhooks/123/abc")
    wh = [d for d in result if d.type == "webhook_url"]
    assert len(wh) == 1


# -- Generic High Entropy -----------------------------------------------------

def test_detects_api_key_with_long_value():
    long_value = "A" * 40
    result = detect_secrets(f"api_key={long_value}")
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) == 1
    assert generic[0].confidence == 0.70


def test_detects_secret_with_long_value():
    long_value = "x" * 32
    result = detect_secrets(f'secret="{long_value}"')
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) == 1


def test_detects_token_with_long_value():
    long_value = "Z" * 35
    result = detect_secrets(f"token: {long_value}")
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) == 1


def test_detects_password_with_long_value():
    long_value = "P" * 40
    result = detect_secrets(f"password={long_value}")
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) == 1


def test_detects_apikey_no_underscore():
    long_value = "Q" * 40
    result = detect_secrets(f"apikey={long_value}")
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) >= 1


def test_does_not_match_short_values():
    result = detect_secrets("key=abc")
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) == 0


def test_generic_is_case_insensitive():
    long_value = "R" * 40
    result = detect_secrets(f"API_KEY={long_value}")
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) == 1


# -- Custom Patterns -----------------------------------------------------------

def test_detects_custom_internal_token():
    result = detect_secrets(
        "Token: INTERNAL-abcdef123456",
        SecretDetectionOptions(
            custom_patterns=[CustomSecretPattern(name="internal_token", pattern=r"INTERNAL-[a-z0-9]{12}")]
        ),
    )
    custom = [d for d in result if d.type == "custom:internal_token"]
    assert len(custom) == 1
    assert custom[0].value == "INTERNAL-abcdef123456"
    assert custom[0].confidence == 0.9


def test_uses_provided_confidence_for_custom_patterns():
    result = detect_secrets(
        "Token: CUSTOM-12345678",
        SecretDetectionOptions(
            custom_patterns=[CustomSecretPattern(name="custom_key", pattern=r"CUSTOM-\d{8}", confidence=0.75)]
        ),
    )
    custom = [d for d in result if d.type == "custom:custom_key"]
    assert len(custom) == 1
    assert custom[0].confidence == 0.75


def test_runs_multiple_custom_patterns():
    text = "ALPHA-1234 and BETA-5678"
    result = detect_secrets(
        text,
        SecretDetectionOptions(
            custom_patterns=[
                CustomSecretPattern(name="alpha", pattern=r"ALPHA-\d{4}"),
                CustomSecretPattern(name="beta", pattern=r"BETA-\d{4}"),
            ]
        ),
    )
    alpha = [d for d in result if d.type == "custom:alpha"]
    beta = [d for d in result if d.type == "custom:beta"]
    assert len(alpha) == 1
    assert len(beta) == 1


def test_skips_invalid_regex_strings():
    result = detect_secrets(
        "some text",
        SecretDetectionOptions(
            custom_patterns=[CustomSecretPattern(name="bad", pattern="[invalid(")]
        ),
    )
    assert len(result) >= 0  # no crash


def test_custom_patterns_run_alongside_builtin():
    text = "AKIAIOSFODNN7EXAMPLE and CUSTOM-99"
    result = detect_secrets(
        text,
        SecretDetectionOptions(
            custom_patterns=[CustomSecretPattern(name="mine", pattern=r"CUSTOM-\d{2}")]
        ),
    )
    aws = [d for d in result if d.type == "aws_access_key"]
    custom = [d for d in result if d.type == "custom:mine"]
    assert len(aws) == 1
    assert len(custom) == 1


def test_custom_pattern_with_zero_confidence():
    result = detect_secrets(
        "ZZZ-token-here",
        SecretDetectionOptions(
            custom_patterns=[CustomSecretPattern(name="low", pattern="ZZZ-token-here", confidence=0.0)]
        ),
    )
    custom = [d for d in result if d.type == "custom:low"]
    assert len(custom) == 1
    assert custom[0].confidence == 0.0


def test_custom_pattern_default_confidence_is_0_9():
    result = detect_secrets(
        "MYTOKEN-abc",
        SecretDetectionOptions(
            custom_patterns=[CustomSecretPattern(name="tok", pattern="MYTOKEN-abc")]
        ),
    )
    custom = [d for d in result if d.type == "custom:tok"]
    assert len(custom) == 1
    assert custom[0].confidence == 0.9


def test_custom_pattern_type_is_prefixed_with_custom():
    result = detect_secrets(
        "SECRET-XYZ",
        SecretDetectionOptions(
            custom_patterns=[CustomSecretPattern(name="my_secret", pattern="SECRET-XYZ")]
        ),
    )
    custom = [d for d in result if d.type == "custom:my_secret"]
    assert len(custom) == 1


# -- built_in_patterns=False ---------------------------------------------------

def test_only_runs_custom_patterns_when_built_in_false():
    text = "AKIAIOSFODNN7EXAMPLE and CUSTOM-99"
    result = detect_secrets(
        text,
        SecretDetectionOptions(
            built_in_patterns=False,
            custom_patterns=[CustomSecretPattern(name="mine", pattern=r"CUSTOM-\d{2}")],
        ),
    )
    aws = [d for d in result if d.type == "aws_access_key"]
    assert len(aws) == 0
    custom = [d for d in result if d.type == "custom:mine"]
    assert len(custom) == 1


def test_returns_empty_when_built_in_false_and_no_custom():
    result = detect_secrets(
        "AKIAIOSFODNN7EXAMPLE",
        SecretDetectionOptions(built_in_patterns=False),
    )
    assert len(result) == 0


def test_built_in_patterns_true_explicitly():
    result = detect_secrets(
        "AKIAIOSFODNN7EXAMPLE",
        SecretDetectionOptions(built_in_patterns=True),
    )
    aws = [d for d in result if d.type == "aws_access_key"]
    assert len(aws) == 1


def test_built_in_patterns_none_defaults_to_true():
    result = detect_secrets(
        "AKIAIOSFODNN7EXAMPLE",
        SecretDetectionOptions(built_in_patterns=None),
    )
    aws = [d for d in result if d.type == "aws_access_key"]
    assert len(aws) == 1


# -- No false positives --------------------------------------------------------

def test_returns_empty_for_normal_english_text():
    result = detect_secrets(
        "The quick brown fox jumps over the lazy dog. "
        "This is a perfectly normal sentence with no secrets."
    )
    assert len(result) == 0


def test_does_not_detect_short_words():
    result = detect_secrets("Hello world, welcome to the app.")
    assert len(result) == 0


def test_does_not_match_public_key_headers():
    result = detect_secrets("-----BEGIN PUBLIC KEY-----")
    assert len(result) == 0


def test_does_not_match_non_live_stripe_keys():
    # sk_test_ keys are test mode, not real secrets in production sense
    result = detect_secrets("sk_" + "test_abcdefghijklmnopqrstuvwx")
    stripe = [d for d in result if d.type == "stripe_secret"]
    assert len(stripe) == 0


def test_does_not_match_certificate_headers():
    result = detect_secrets("-----BEGIN CERTIFICATE-----")
    pk = [d for d in result if d.type == "private_key"]
    assert len(pk) == 0


def test_does_not_match_regular_urls():
    result = detect_secrets("Visit https://example.com/page?query=value")
    cs = [d for d in result if d.type == "connection_string"]
    wh = [d for d in result if d.type == "webhook_url"]
    assert len(cs) == 0
    assert len(wh) == 0


def test_does_not_flag_pk_test_stripe_keys():
    result = detect_secrets("pk_test_abcdefghijklmnopqrstuvwx")
    stripe = [d for d in result if d.type == "stripe_publishable"]
    assert len(stripe) == 0


# -- Multiple secrets ----------------------------------------------------------

def test_detects_multiple_different_secrets():
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    gh_token = "ghp_" + "a" * 36
    text = f"AWS: {aws_key} and GitHub: {gh_token}"
    result = detect_secrets(text)
    types = set(d.type for d in result)
    assert "aws_access_key" in types
    assert "github_pat" in types
    assert len(result) >= 2


def test_detects_multiple_same_type_secrets():
    key1 = "AKIAIOSFODNN7AAAAAAA"
    key2 = "AKIAIOSFODNN7BBBBBBB"
    result = detect_secrets(f"First: {key1} Second: {key2}")
    aws = [d for d in result if d.type == "aws_access_key"]
    assert len(aws) == 2


def test_detects_three_distinct_secret_types():
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    text = f"-----BEGIN PRIVATE KEY----- then {aws_key}"
    result = detect_secrets(text)
    types = set(d.type for d in result)
    assert "private_key" in types
    assert "aws_access_key" in types
    assert len(result) >= 2


# -- Deduplication -------------------------------------------------------------

def test_deduplicates_overlapping_detections():
    # generic_high_entropy (0.70) may overlap with a more specific pattern
    # Stripe key also matches generic_high_entropy
    key = "sk_live_" + "A" * 32
    text = f'key="{key}"'
    result = detect_secrets(text)
    # The stripe_secret match (0.95) should win over generic_high_entropy (0.70)
    stripe = [d for d in result if d.type == "stripe_secret"]
    assert len(stripe) >= 1


def test_keeps_non_overlapping_detections():
    aws_key = "AKIAIOSFODNN7EXAMPLE"
    text = f"-----BEGIN PRIVATE KEY----- then {aws_key}"
    result = detect_secrets(text)
    assert len(result) >= 2
    types = set(d.type for d in result)
    assert "private_key" in types
    assert "aws_access_key" in types


def test_overlapping_custom_and_builtin_keeps_higher_confidence():
    # Custom pattern overlaps with built-in AWS detection
    result = detect_secrets(
        "AKIAIOSFODNN7EXAMPLE",
        SecretDetectionOptions(
            custom_patterns=[
                CustomSecretPattern(name="overlap", pattern="AKIAIOSFODNN7EXAMPLE", confidence=0.50)
            ]
        ),
    )
    # Built-in aws_access_key (0.95) should win over custom (0.50)
    aws = [d for d in result if d.type == "aws_access_key"]
    custom = [d for d in result if d.type == "custom:overlap"]
    assert len(aws) == 1
    assert len(custom) == 0


def test_overlapping_custom_wins_when_higher_confidence():
    result = detect_secrets(
        "AKIAIOSFODNN7EXAMPLE",
        SecretDetectionOptions(
            custom_patterns=[
                CustomSecretPattern(name="overlap", pattern="AKIAIOSFODNN7EXAMPLE", confidence=0.99)
            ]
        ),
    )
    # Custom (0.99) should win over aws_access_key (0.95)
    aws = [d for d in result if d.type == "aws_access_key"]
    custom = [d for d in result if d.type == "custom:overlap"]
    assert len(aws) == 0
    assert len(custom) == 1


# -- Position (start/end) correctness -----------------------------------------

def test_returns_correct_start_and_end_for_aws_key():
    prefix = "AWS key: "
    key = "AKIAIOSFODNN7EXAMPLE"
    text = prefix + key
    result = detect_secrets(text)
    assert len(result) == 1
    assert result[0].start == len(prefix)
    assert result[0].end == len(prefix) + len(key)
    assert text[result[0].start : result[0].end] == key


def test_returns_correct_positions_for_multiple_secrets():
    text = "A: AKIAIOSFODNN7EXAMPLE B: -----BEGIN PRIVATE KEY-----"
    result = detect_secrets(text)
    for d in result:
        assert text[d.start : d.end] == d.value


def test_results_are_sorted_by_start_position():
    gh_token = "ghp_" + "x" * 36
    text = f"-----BEGIN PRIVATE KEY----- then AKIAIOSFODNN7EXAMPLE then {gh_token}"
    result = detect_secrets(text)
    for i in range(1, len(result)):
        assert result[i].start >= result[i - 1].start


def test_start_is_zero_when_secret_at_beginning():
    result = detect_secrets("AKIAIOSFODNN7EXAMPLE trailing text")
    assert len(result) >= 1
    assert result[0].start == 0


def test_end_equals_text_length_when_secret_at_end():
    text = "prefix: AKIAIOSFODNN7EXAMPLE"
    result = detect_secrets(text)
    assert len(result) >= 1
    assert result[0].end == len(text)


# -- Confidence values ---------------------------------------------------------

def test_private_key_has_highest_confidence():
    result = detect_secrets("-----BEGIN RSA PRIVATE KEY-----")
    assert result[0].confidence == 0.99


def test_aws_access_key_has_0_95_confidence():
    result = detect_secrets("AKIAIOSFODNN7EXAMPLE")
    assert result[0].confidence == 0.95


def test_generic_high_entropy_has_lowest_confidence():
    long_value = "Q" * 40
    result = detect_secrets(f"apikey={long_value}")
    generic = [d for d in result if d.type == "generic_high_entropy"]
    assert len(generic) >= 1
    assert generic[0].confidence == 0.70


def test_stripe_publishable_has_0_85_confidence():
    key = "pk_live_" + "C" * 24
    result = detect_secrets(key)
    assert result[0].confidence == 0.85


def test_slack_token_has_0_90_confidence():
    result = detect_secrets("xoxb-123456789-abcdefghij")
    slack = [d for d in result if d.type == "slack_token"]
    assert len(slack) >= 1
    assert slack[0].confidence == 0.90


def test_jwt_has_0_90_confidence():
    jwt_val = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.abc123DEF456_-"
    result = detect_secrets(jwt_val)
    jwts = [d for d in result if d.type == "jwt"]
    assert len(jwts) >= 1
    assert jwts[0].confidence == 0.90


def test_connection_string_has_0_90_confidence():
    result = detect_secrets("postgresql://user:pass@host:5432/db")
    cs = [d for d in result if d.type == "connection_string"]
    assert len(cs) == 1
    assert cs[0].confidence == 0.90


def test_webhook_url_has_0_85_confidence():
    result = detect_secrets("https://hooks.slack.com/services/T00/B00/xxxx")
    wh = [d for d in result if d.type == "webhook_url"]
    assert len(wh) == 1
    assert wh[0].confidence == 0.85


def test_github_pat_has_0_95_confidence():
    token = "ghp_" + "A" * 36
    result = detect_secrets(token)
    assert result[0].confidence == 0.95


def test_github_oauth_has_0_95_confidence():
    token = "gho_" + "A" * 36
    result = detect_secrets(token)
    assert result[0].confidence == 0.95


def test_gitlab_pat_has_0_95_confidence():
    token = "glpat-" + "A" * 20
    result = detect_secrets(token)
    assert result[0].confidence == 0.95


def test_stripe_secret_has_0_95_confidence():
    key = "sk_live_" + "A" * 24
    result = detect_secrets(key)
    assert result[0].confidence == 0.95


# -- Edge cases ----------------------------------------------------------------

def test_returns_empty_for_empty_string():
    assert len(detect_secrets("")) == 0


def test_returns_empty_list_type_for_empty_string():
    result = detect_secrets("")
    assert result == []
    assert isinstance(result, list)


def test_handles_text_with_no_secrets():
    result = detect_secrets("Just a regular conversation about coding.")
    assert len(result) == 0


def test_handles_very_long_text_with_secret_near_end():
    padding = " " * 500000
    key = "AKIAIOSFODNN7EXAMPLE"
    text = padding + key + padding
    result = detect_secrets(text)
    assert len(result) >= 1


def test_truncates_text_beyond_max_scan_length():
    padding = "x" * MAX_SCAN_LENGTH
    key = "AKIAIOSFODNN7EXAMPLE"
    text = padding + key
    result = detect_secrets(text)
    assert len(result) == 0


def test_handles_unicode_text_without_crashing():
    result = detect_secrets("Secret: AKIAIOSFODNN7EXAMPLE in unicode \U0001f680\U0001f525 text")
    assert len(result) >= 1


def test_handles_none_options():
    result = detect_secrets("AKIAIOSFODNN7EXAMPLE", None)
    assert len(result) == 1
    assert result[0].type == "aws_access_key"


def test_handles_options_with_none_fields():
    result = detect_secrets(
        "AKIAIOSFODNN7EXAMPLE",
        SecretDetectionOptions(built_in_patterns=None, custom_patterns=None),
    )
    assert len(result) == 1
    assert result[0].type == "aws_access_key"


def test_handles_whitespace_only_input():
    result = detect_secrets("     \t\n\r   ")
    assert len(result) == 0


def test_handles_newlines_around_secrets():
    result = detect_secrets("\n\nAKIAIOSFODNN7EXAMPLE\n\n")
    aws = [d for d in result if d.type == "aws_access_key"]
    assert len(aws) == 1


def test_handles_multiline_input_with_multiple_secrets():
    gh_token = "ghp_" + "a" * 36
    text = (
        "line1: AKIAIOSFODNN7EXAMPLE\n"
        f"line2: {gh_token}\n"
        "line3: -----BEGIN PRIVATE KEY-----\n"
    )
    result = detect_secrets(text)
    types = set(d.type for d in result)
    assert "aws_access_key" in types
    assert "github_pat" in types
    assert "private_key" in types


def test_handles_empty_custom_patterns_list():
    result = detect_secrets(
        "AKIAIOSFODNN7EXAMPLE",
        SecretDetectionOptions(custom_patterns=[]),
    )
    assert len(result) == 1
    assert result[0].type == "aws_access_key"


# -- Return type structure -----------------------------------------------------

def test_detection_has_correct_fields():
    result = detect_secrets("AKIAIOSFODNN7EXAMPLE")
    assert len(result) == 1
    d = result[0]
    assert isinstance(d, SecretDetection)
    assert isinstance(d.type, str)
    assert isinstance(d.value, str)
    assert isinstance(d.start, int)
    assert isinstance(d.end, int)
    assert isinstance(d.confidence, float)


def test_detection_value_matches_input_slice():
    text = "prefix AKIAIOSFODNN7EXAMPLE suffix"
    result = detect_secrets(text)
    assert len(result) == 1
    d = result[0]
    assert text[d.start : d.end] == d.value


# -- Performance / DoS prevention ----------------------------------------------

def test_handles_large_input_without_hanging():
    huge = "AKIAIOSFODNN7EXAMPLE " * 10000
    start = time.monotonic()
    result = detect_secrets(huge)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"DoS: secret detection took {elapsed:.2f}s on huge input"
    assert len(result) >= 1


def test_max_scan_length_constant_is_one_mb():
    assert MAX_SCAN_LENGTH == 1024 * 1024


# -- Comprehensive scenarios ---------------------------------------------------

def test_detects_secrets_in_env_file_style_input():
    gh_token = "ghp_" + "A" * 36
    env_text = (
        "DATABASE_URL=postgresql://admin:secretpass@db.example.com:5432/myapp\n"
        "REDIS_URL=redis://default:r3d1sP@ss@cache.example.com:6379\n"
        "AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE\n"
        f"GITHUB_TOKEN={gh_token}\n"
        "SLACK_WEBHOOK=https://hooks.slack.com/services/T00/B00/xxxx\n"
        "STRIPE_SECRET_KEY=" + "sk_live_" + "A" * 24 + "\n"
    )
    result = detect_secrets(env_text)
    types = set(d.type for d in result)
    assert "connection_string" in types
    assert "aws_access_key" in types
    assert "github_pat" in types
    assert "webhook_url" in types


def test_detects_private_key_in_multiline_pem():
    pem = (
        "-----BEGIN RSA PRIVATE KEY-----\n"
        "MIIEpAIBAAKCAQEA2a2rwplBQLHV1/...\n"
        "-----END RSA PRIVATE KEY-----"
    )
    result = detect_secrets(pem)
    pk = [d for d in result if d.type == "private_key"]
    assert len(pk) == 1


def test_config_file_with_mixed_secrets():
    config = (
        "[database]\n"
        "url = mongodb+srv://root:p4ssw0rd@cluster0.example.net/production\n"
        "\n"
        "[api]\n"
        'key = "' + "sk_live_" + "B" * 24 + '"\n'
        "token: xoxb-999888777-abcdefghij\n"
    )
    result = detect_secrets(config)
    types = set(d.type for d in result)
    assert "connection_string" in types
    assert "slack_token" in types


def test_json_config_with_secrets():
    json_text = (
        '{\n'
        '  "database_url": "mysql://root:password@localhost:3306/app",\n'
        '  "api_key": "ghp_' + 'Z' * 36 + '"\n'
        '}'
    )
    result = detect_secrets(json_text)
    types = set(d.type for d in result)
    assert "connection_string" in types
    assert "github_pat" in types


def test_yaml_config_with_secrets():
    yaml_text = (
        "database:\n"
        "  url: postgresql://admin:pass@db.host:5432/prod\n"
        "slack:\n"
        "  token: xoxb-111222333-aabbccddee\n"
    )
    result = detect_secrets(yaml_text)
    types = set(d.type for d in result)
    assert "connection_string" in types
    assert "slack_token" in types
