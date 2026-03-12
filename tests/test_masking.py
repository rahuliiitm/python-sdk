"""Tests for the masking redaction strategy."""
import pytest

from launchpromptly._internal.redaction import (
    MaskingOptions,
    RedactionOptions,
    redact_pii,
)


# -- Credit card masking -------------------------------------------------------

def test_mask_credit_card_shows_last_4():
    result = redact_pii(
        "Card: 4111-1111-1111-1111",
        RedactionOptions(strategy="mask"),
    )
    assert "****-****-****-1111" in result.redacted_text
    assert "4111-1111-1111-1111" not in result.redacted_text


def test_mask_credit_card_no_dashes():
    result = redact_pii(
        "Card: 4111111111111111",
        RedactionOptions(strategy="mask"),
    )
    assert "****-****-****-1111" in result.redacted_text


# -- Email masking -------------------------------------------------------------

def test_mask_email_keeps_first_char_and_domain():
    result = redact_pii(
        "Contact john@acme.com",
        RedactionOptions(strategy="mask"),
    )
    assert "j***@acme.com" in result.redacted_text
    assert "john@acme.com" not in result.redacted_text


def test_mask_email_single_char_local():
    result = redact_pii(
        "Email: a@test.com",
        RedactionOptions(strategy="mask"),
    )
    assert "a@test.com" in result.redacted_text


def test_mask_email_long_local_part():
    result = redact_pii(
        "Email: longuser@example.org",
        RedactionOptions(strategy="mask"),
    )
    assert result.redacted_text.startswith("Email: l")
    assert "@example.org" in result.redacted_text
    assert "longuser@example.org" not in result.redacted_text


# -- Phone masking -------------------------------------------------------------

def test_mask_phone_us_shows_last_4():
    result = redact_pii(
        "Call (555) 123-4567",
        RedactionOptions(strategy="mask"),
    )
    phones = [d for d in result.detections if d.type == "phone"]
    assert len(phones) >= 1
    # The last 4 digits should be visible
    assert "4567" in result.redacted_text


def test_mask_phone_dashes_shows_last_4():
    result = redact_pii(
        "Call 555-123-4567",
        RedactionOptions(strategy="mask"),
    )
    assert "4567" in result.redacted_text
    assert "555-123" not in result.redacted_text


# -- SSN masking ---------------------------------------------------------------

def test_mask_ssn_shows_last_4():
    result = redact_pii(
        "SSN: 123-45-6789",
        RedactionOptions(strategy="mask"),
    )
    assert "***-**-6789" in result.redacted_text
    assert "123-45-6789" not in result.redacted_text


def test_mask_ssn_preserves_format():
    result = redact_pii(
        "SSN is 078-05-1120",
        RedactionOptions(strategy="mask"),
    )
    assert "***-**-1120" in result.redacted_text


# -- Generic / default masking -------------------------------------------------

def test_mask_generic_shows_suffix():
    result = redact_pii(
        "Server at 82.45.113.42",
        RedactionOptions(strategy="mask"),
    )
    ips = [d for d in result.detections if d.type == "ip_address"]
    assert len(ips) >= 1
    # Default: last 4 chars visible
    assert ".42" in result.redacted_text


def test_mask_custom_char():
    result = redact_pii(
        "SSN: 123-45-6789",
        RedactionOptions(strategy="mask", masking=MaskingOptions(char="#")),
    )
    assert "###-##-6789" in result.redacted_text


def test_mask_custom_visible_suffix():
    result = redact_pii(
        "Key: sk-abcdefghijklmnopqrstuvwxyz1234",
        RedactionOptions(
            strategy="mask",
            masking=MaskingOptions(visible_suffix=6),
        ),
    )
    keys = [d for d in result.detections if d.type == "api_key"]
    assert len(keys) >= 1
    # Last 6 chars should be visible
    assert "z1234" in result.redacted_text


def test_mask_with_visible_prefix():
    result = redact_pii(
        "Key: sk-abcdefghijklmnopqrstuvwxyz1234",
        RedactionOptions(
            strategy="mask",
            masking=MaskingOptions(visible_prefix=3, visible_suffix=4),
        ),
    )
    keys = [d for d in result.detections if d.type == "api_key"]
    assert len(keys) >= 1
    # First 3 chars and last 4 chars should be visible
    assert "sk-" in result.redacted_text
    assert "1234" in result.redacted_text


# -- Mapping for mask strategy -------------------------------------------------

def test_mask_creates_mapping():
    result = redact_pii(
        "Email john@acme.com",
        RedactionOptions(strategy="mask"),
    )
    assert len(result.mapping) == 1
    # mapping: masked -> original
    replacement, original = next(iter(result.mapping.items()))
    assert original == "john@acme.com"
    assert "j***@acme.com" in replacement


def test_mask_returns_detections():
    result = redact_pii(
        "SSN 123-45-6789 and john@acme.com",
        RedactionOptions(strategy="mask"),
    )
    assert len(result.detections) >= 2
