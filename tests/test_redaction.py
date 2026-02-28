"""Tests for PII redaction module."""
import pytest

from launchpromptly._internal.redaction import (
    RedactionOptions,
    de_redact,
    redact_pii,
)
from launchpromptly._internal.pii import PIIDetection, PIIDetectOptions


# -- Placeholder strategy ------------------------------------------------------

def test_placeholder_replaces_email():
    result = redact_pii("Contact john@acme.com", RedactionOptions(strategy="placeholder"))
    assert "[EMAIL_" in result.redacted_text
    assert "john@acme.com" not in result.redacted_text
    assert len(result.detections) == 1


def test_placeholder_incrementing_indices_for_same_type():
    result = redact_pii("Email a@b.com and c@d.com", RedactionOptions(strategy="placeholder"))
    assert "[EMAIL_1]" in result.redacted_text
    assert "[EMAIL_2]" in result.redacted_text


def test_placeholder_replaces_ssn():
    result = redact_pii("SSN: 123-45-6789", RedactionOptions(strategy="placeholder"))
    assert "[SSN_" in result.redacted_text
    assert "123-45-6789" not in result.redacted_text


def test_placeholder_replaces_multiple_pii_types():
    result = redact_pii("Email john@acme.com SSN 123-45-6789")
    assert "[EMAIL_" in result.redacted_text
    assert "[SSN_" in result.redacted_text
    assert len(result.detections) == 2


def test_placeholder_is_default_strategy():
    result = redact_pii("Email john@acme.com")
    assert "[EMAIL_" in result.redacted_text


# -- Synthetic strategy --------------------------------------------------------

def test_synthetic_replaces_email():
    result = redact_pii("Contact john@acme.com", RedactionOptions(strategy="synthetic"))
    assert "john@acme.com" not in result.redacted_text
    assert "@example." in result.redacted_text


def test_synthetic_replaces_ssn():
    result = redact_pii("SSN: 123-45-6789", RedactionOptions(strategy="synthetic"))
    assert "123-45-6789" not in result.redacted_text
    assert "000-00-" in result.redacted_text


# -- Hash strategy -------------------------------------------------------------

def test_hash_replaces_email():
    result = redact_pii("Contact john@acme.com", RedactionOptions(strategy="hash"))
    assert "john@acme.com" not in result.redacted_text
    assert len(result.redacted_text) > 0


def test_hash_same_input_produces_same_hash():
    r1 = redact_pii("john@acme.com", RedactionOptions(strategy="hash"))
    r2 = redact_pii("john@acme.com", RedactionOptions(strategy="hash"))
    assert r1.redacted_text == r2.redacted_text


def test_hash_different_inputs_produce_different_hashes():
    r1 = redact_pii("a@b.com", RedactionOptions(strategy="hash"))
    r2 = redact_pii("c@d.com", RedactionOptions(strategy="hash"))
    assert r1.redacted_text != r2.redacted_text


# -- De-redaction --------------------------------------------------------------

def test_de_redact_placeholder_roundtrip():
    result = redact_pii("Email john@acme.com", RedactionOptions(strategy="placeholder"))
    restored = de_redact(result.redacted_text, result.mapping)
    assert restored == "Email john@acme.com"


def test_de_redact_synthetic_roundtrip():
    result = redact_pii("Email john@acme.com", RedactionOptions(strategy="synthetic"))
    restored = de_redact(result.redacted_text, result.mapping)
    assert restored == "Email john@acme.com"


def test_de_redact_multiple_pii_values():
    original = "Email john@acme.com SSN 123-45-6789"
    result = redact_pii(original, RedactionOptions(strategy="placeholder"))
    restored = de_redact(result.redacted_text, result.mapping)
    assert restored == original


# -- Mapping -------------------------------------------------------------------

def test_mapping_replacement_to_original():
    result = redact_pii("john@acme.com")
    assert len(result.mapping) == 1
    replacement, original = next(iter(result.mapping.items()))
    assert original == "john@acme.com"
    assert "[EMAIL_" in replacement


def test_mapping_empty_for_no_pii():
    result = redact_pii("Hello world")
    assert len(result.mapping) == 0


# -- Edge cases ----------------------------------------------------------------

def test_handles_empty_string():
    result = redact_pii("")
    assert result.redacted_text == ""
    assert len(result.detections) == 0


def test_returns_original_text_when_no_pii():
    text = "Hello world"
    result = redact_pii(text)
    assert result.redacted_text == text


def test_type_filtering_works():
    text = "Email a@b.com SSN 123-45-6789"
    result = redact_pii(text, RedactionOptions(types=["email"]))
    assert "a@b.com" not in result.redacted_text
    assert "123-45-6789" in result.redacted_text  # SSN not redacted


# -- Provider integration -----------------------------------------------------

def test_merges_detections_from_additional_providers():
    class MockProvider:
        @property
        def name(self):
            return "mock"

        @property
        def supported_types(self):
            return ["email"]

        def detect(self, text, options=None):
            return [
                PIIDetection(
                    type="email",
                    value="hidden@test.com",
                    start=50,
                    end=65,
                    confidence=0.99,
                ),
            ]

    text = "Contact john@acme.com" + " " * 29 + "hidden@test.com"
    result = redact_pii(text, RedactionOptions(providers=[MockProvider()]))
    # Should have detections from both built-in and mock provider
    assert len(result.detections) >= 2


def test_gracefully_handles_provider_errors():
    class FailingProvider:
        @property
        def name(self):
            return "failing"

        @property
        def supported_types(self):
            return ["email"]

        def detect(self, text, options=None):
            raise RuntimeError("Provider crashed")

    result = redact_pii("john@acme.com", RedactionOptions(providers=[FailingProvider()]))
    # Should still work with built-in detector
    assert len(result.detections) == 1
