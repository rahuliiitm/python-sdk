"""Tests for PII detection module."""
import pytest

from launchpromptly._internal.pii import (
    PIIDetection,
    PIIDetectOptions,
    RegexPIIDetector,
    detect_pii,
    merge_detections,
)


# -- Email ---------------------------------------------------------------------

def test_detects_simple_email():
    result = detect_pii("Contact john@acme.com for details")
    assert len(result) == 1
    assert result[0].type == "email"
    assert result[0].value == "john@acme.com"
    assert result[0].confidence > 0.9


def test_detects_multiple_emails():
    result = detect_pii("Send to a@b.com and c@d.org")
    assert len(result) == 2
    assert result[0].value == "a@b.com"
    assert result[1].value == "c@d.org"


def test_detects_email_with_plus_and_dots():
    result = detect_pii("Email: first.last+tag@sub.domain.co.uk")
    assert len(result) == 1
    assert result[0].value == "first.last+tag@sub.domain.co.uk"


def test_no_false_positive_on_at_mentions():
    result = detect_pii("Check the code @line 42")
    emails = [d for d in result if d.type == "email"]
    assert len(emails) == 0


# -- Phone ---------------------------------------------------------------------

def test_detects_us_phone_with_parens():
    result = detect_pii("Call (555) 123-4567")
    phones = [d for d in result if d.type == "phone"]
    assert len(phones) >= 1


def test_detects_us_phone_with_dashes():
    result = detect_pii("Call 555-123-4567")
    phones = [d for d in result if d.type == "phone"]
    assert len(phones) >= 1


def test_detects_us_phone_with_plus1_prefix():
    result = detect_pii("Call +1 555-123-4567")
    phones = [d for d in result if d.type == "phone"]
    assert len(phones) >= 1


def test_detects_international_phone():
    result = detect_pii("Call +442079460958")
    phones = [d for d in result if d.type == "phone"]
    assert len(phones) >= 1


# -- SSN -----------------------------------------------------------------------

def test_detects_ssn_format():
    result = detect_pii("SSN: 123-45-6789")
    assert len(result) == 1
    assert result[0].type == "ssn"
    assert result[0].value == "123-45-6789"


def test_detects_ssn_in_context():
    result = detect_pii("My social security number is 987-65-4321")
    ssns = [d for d in result if d.type == "ssn"]
    assert len(ssns) == 1


# -- Credit Card ---------------------------------------------------------------

def test_detects_visa_number():
    result = detect_pii("Card: 4111111111111111")
    cards = [d for d in result if d.type == "credit_card"]
    assert len(cards) == 1
    assert cards[0].value.replace(" ", "").replace("-", "") == "4111111111111111"


def test_detects_card_with_dashes():
    result = detect_pii("Card: 4111-1111-1111-1111")
    cards = [d for d in result if d.type == "credit_card"]
    assert len(cards) == 1


def test_rejects_invalid_luhn_number():
    result = detect_pii("Number: 1234567890123456")
    cards = [d for d in result if d.type == "credit_card"]
    assert len(cards) == 0


def test_detects_mastercard():
    result = detect_pii("Card: 5500000000000004")
    cards = [d for d in result if d.type == "credit_card"]
    assert len(cards) == 1


# -- IP Address ----------------------------------------------------------------

def test_detects_valid_ip():
    result = detect_pii("Server at 203.0.113.42")
    ips = [d for d in result if d.type == "ip_address"]
    assert len(ips) == 1
    assert ips[0].value == "203.0.113.42"


def test_filters_well_known_ips_localhost():
    result = detect_pii("Connect to 127.0.0.1")
    ips = [d for d in result if d.type == "ip_address"]
    assert len(ips) == 0


def test_filters_0000():
    result = detect_pii("Bind to 0.0.0.0")
    ips = [d for d in result if d.type == "ip_address"]
    assert len(ips) == 0


def test_does_not_match_invalid_octets():
    result = detect_pii("Version 999.999.999.999")
    ips = [d for d in result if d.type == "ip_address"]
    assert len(ips) == 0


# -- API Key -------------------------------------------------------------------

def test_detects_openai_key():
    result = detect_pii("Key: sk-abcdefghijklmnopqrstuvwxyz1234")
    keys = [d for d in result if d.type == "api_key"]
    assert len(keys) == 1


def test_detects_openai_project_key():
    result = detect_pii("Key: sk-proj-abcdefghijklmnopqrstuvwxyz")
    keys = [d for d in result if d.type == "api_key"]
    assert len(keys) == 1


def test_detects_aws_access_key():
    result = detect_pii("AWS key: AKIAIOSFODNN7EXAMPLE")
    keys = [d for d in result if d.type == "api_key"]
    assert len(keys) == 1


def test_detects_github_pat():
    result = detect_pii("Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890")
    keys = [d for d in result if d.type == "api_key"]
    assert len(keys) == 1


def test_detects_slack_token():
    result = detect_pii("Token: xoxb-123456789-abcdef")
    keys = [d for d in result if d.type == "api_key"]
    assert len(keys) == 1


# -- Date of Birth -------------------------------------------------------------

def test_detects_mm_dd_yyyy_format():
    result = detect_pii("DOB: 01/15/1990")
    dobs = [d for d in result if d.type == "date_of_birth"]
    assert len(dobs) == 1
    assert dobs[0].value == "01/15/1990"


def test_detects_mm_dd_yyyy_with_dashes():
    result = detect_pii("Born on 12-25-2000")
    dobs = [d for d in result if d.type == "date_of_birth"]
    assert len(dobs) == 1


def test_does_not_match_invalid_months():
    result = detect_pii("Date: 13/01/2000")
    dobs = [d for d in result if d.type == "date_of_birth"]
    assert len(dobs) == 0


# -- US Address ----------------------------------------------------------------

def test_detects_street_address():
    result = detect_pii("Lives at 123 Main Street")
    addrs = [d for d in result if d.type == "us_address"]
    assert len(addrs) == 1


def test_detects_avenue_address():
    result = detect_pii("Office at 456 Park Ave")
    addrs = [d for d in result if d.type == "us_address"]
    assert len(addrs) == 1


def test_detects_boulevard_address():
    result = detect_pii("Located at 789 Sunset Blvd")
    addrs = [d for d in result if d.type == "us_address"]
    assert len(addrs) == 1


# -- Multi-PII ----------------------------------------------------------------

def test_detects_multiple_pii_types():
    text = "Contact john@acme.com, SSN 123-45-6789, call (555) 123-4567"
    result = detect_pii(text)
    types = {d.type for d in result}
    assert "email" in types
    assert "ssn" in types
    assert "phone" in types


def test_detections_sorted_by_start_position():
    text = "SSN 123-45-6789 and email a@b.com"
    result = detect_pii(text)
    for i in range(1, len(result)):
        assert result[i].start >= result[i - 1].start


# -- Type filtering ------------------------------------------------------------

def test_only_detects_specified_types():
    text = "Email a@b.com and SSN 123-45-6789"
    result = detect_pii(text, PIIDetectOptions(types=["email"]))
    assert len(result) == 1
    assert result[0].type == "email"


def test_returns_empty_for_non_matching_types():
    text = "Email a@b.com"
    result = detect_pii(text, PIIDetectOptions(types=["ssn"]))
    assert len(result) == 0


# -- Edge cases ----------------------------------------------------------------

def test_returns_empty_for_empty_string():
    assert len(detect_pii("")) == 0


def test_returns_empty_for_text_with_no_pii():
    result = detect_pii("The quick brown fox jumps over the lazy dog")
    assert len(result) == 0


def test_handles_very_long_text():
    text = "x" * 100000 + " john@acme.com " + "x" * 100000
    result = detect_pii(text)
    assert len(result) == 1
    assert result[0].type == "email"


def test_provides_correct_start_end_positions():
    text = "prefix john@acme.com suffix"
    result = detect_pii(text)
    assert result[0].start == 7
    assert result[0].end == 20  # 7 + len("john@acme.com") = 20
    assert text[result[0].start : result[0].end] == "john@acme.com"


# -- merge_detections ----------------------------------------------------------

def test_merge_detections_from_multiple_sources():
    a = [PIIDetection(type="email", value="a@b.com", start=0, end=7, confidence=0.9)]
    b = [PIIDetection(type="ssn", value="123-45-6789", start=20, end=31, confidence=0.95)]
    merged = merge_detections(a, b)
    assert len(merged) == 2


def test_merge_deduplicates_overlapping_keeping_higher_confidence():
    a = [PIIDetection(type="phone", value="5551234567", start=5, end=15, confidence=0.7)]
    b = [PIIDetection(type="phone", value="5551234567", start=5, end=15, confidence=0.95)]
    merged = merge_detections(a, b)
    assert len(merged) == 1
    assert merged[0].confidence == 0.95


# -- RegexPIIDetector provider -------------------------------------------------

def test_regex_pii_detector_implements_interface():
    detector = RegexPIIDetector()
    assert detector.name == "regex"
    assert len(detector.supported_types) > 0


def test_regex_pii_detector_detect_works():
    detector = RegexPIIDetector()
    result = detector.detect("Email: a@b.com")
    assert len(result) == 1
    assert result[0].type == "email"
