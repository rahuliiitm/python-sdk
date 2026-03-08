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
    result = detect_pii("My social security number is 078-65-4321")
    ssns = [d for d in result if d.type == "ssn"]
    assert len(ssns) == 1


def test_detects_ssn_without_separators():
    result = detect_pii("SSN: 123456789")
    ssns = [d for d in result if d.type == "ssn"]
    assert len(ssns) == 1
    assert ssns[0].value == "123456789"


def test_detects_ssn_with_spaces():
    result = detect_pii("SSN: 123 45 6789")
    ssns = [d for d in result if d.type == "ssn"]
    assert len(ssns) == 1
    assert ssns[0].value == "123 45 6789"


def test_rejects_ssn_area_000():
    result = detect_pii("SSN: 000-12-3456")
    ssns = [d for d in result if d.type == "ssn"]
    assert len(ssns) == 0


def test_rejects_ssn_area_666():
    result = detect_pii("SSN: 666-12-3456")
    ssns = [d for d in result if d.type == "ssn"]
    assert len(ssns) == 0


def test_rejects_ssn_area_9xx():
    result = detect_pii("SSN: 900-12-3456")
    ssns = [d for d in result if d.type == "ssn"]
    assert len(ssns) == 0


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
    result = detect_pii("Server at 85.12.45.78")
    ips = [d for d in result if d.type == "ip_address"]
    assert len(ips) == 1
    assert ips[0].value == "85.12.45.78"


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


# -- Security regression: ReDoS resistance ------------------------------------

def test_credit_card_regex_no_catastrophic_backtracking():
    import time
    adversarial = "1234 5678 9012 3456 7890 1234 5678 9012 3456 " * 20
    start = time.monotonic()
    detect_pii(adversarial, PIIDetectOptions(types=["credit_card"]))
    elapsed = time.monotonic() - start
    assert elapsed < 0.1, f"ReDoS: credit card regex took {elapsed:.2f}s"


def test_credit_card_regex_digits_only_no_hang():
    import time
    adversarial = "1" * 1000 + "X"
    start = time.monotonic()
    detect_pii(adversarial, PIIDetectOptions(types=["credit_card"]))
    elapsed = time.monotonic() - start
    assert elapsed < 0.1, f"ReDoS: credit card regex took {elapsed:.2f}s"


def test_input_length_capped_for_dos_prevention():
    import time
    huge = "a@b.com " * 200000
    start = time.monotonic()
    result = detect_pii(huge)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"DoS: detection took {elapsed:.2f}s on huge input"
    assert len(result) > 0


# -- False positive prevention (context-aware) --------------------------------

class TestPhoneContextCheck:
    def test_bare_digits_without_context_not_detected(self):
        result = detect_pii("order number 5551234567")
        phones = [d for d in result if d.type == "phone"]
        assert len(phones) == 0

    def test_bare_digits_with_context_detected(self):
        result = detect_pii("call 5551234567")
        phones = [d for d in result if d.type == "phone"]
        assert len(phones) >= 1

    def test_formatted_always_detected(self):
        result = detect_pii("order 555-123-4567")
        phones = [d for d in result if d.type == "phone"]
        assert len(phones) >= 1

    def test_mobile_context_detected(self):
        result = detect_pii("mobile 9876543210")
        phones = [d for d in result if d.type == "phone"]
        assert len(phones) >= 1

    def test_product_number_not_detected(self):
        result = detect_pii("i need to order product with number 95789930")
        phones = [d for d in result if d.type == "phone"]
        assert len(phones) == 0


class TestSSNContextCheck:
    def test_bare_digits_without_context_not_detected(self):
        result = detect_pii("reference 123456789")
        ssns = [d for d in result if d.type == "ssn"]
        assert len(ssns) == 0

    def test_with_context_detected(self):
        result = detect_pii("SSN: 123456789")
        ssns = [d for d in result if d.type == "ssn"]
        assert len(ssns) == 1

    def test_formatted_always_detected(self):
        result = detect_pii("ref 123-45-6789")
        ssns = [d for d in result if d.type == "ssn"]
        assert len(ssns) == 1


class TestDOBContextCheck:
    def test_date_without_context_not_detected(self):
        result = detect_pii("meeting on 03/15/2026")
        dobs = [d for d in result if d.type == "date_of_birth"]
        assert len(dobs) == 0

    def test_with_born_context_detected(self):
        result = detect_pii("born 03/15/1990")
        dobs = [d for d in result if d.type == "date_of_birth"]
        assert len(dobs) == 1

    def test_with_dob_context_detected(self):
        result = detect_pii("DOB: 03/15/1990")
        dobs = [d for d in result if d.type == "date_of_birth"]
        assert len(dobs) == 1

    def test_with_born_on_context_detected(self):
        result = detect_pii("Born on 12-25-2000")
        dobs = [d for d in result if d.type == "date_of_birth"]
        assert len(dobs) == 1

    def test_formatted_without_context_not_detected(self):
        result = detect_pii("scheduled for 01/15/1990")
        dobs = [d for d in result if d.type == "date_of_birth"]
        assert len(dobs) == 0


class TestPassportContextCheck:
    def test_without_context_not_detected(self):
        result = detect_pii("product AB123456")
        passports = [d for d in result if d.type == "passport"]
        assert len(passports) == 0

    def test_with_context_detected(self):
        result = detect_pii("passport C12345678")
        passports = [d for d in result if d.type == "passport"]
        assert len(passports) == 1


class TestIPContextCheck:
    def test_version_number_not_detected(self):
        result = detect_pii("version 1.2.3.4")
        ips = [d for d in result if d.type == "ip_address"]
        assert len(ips) == 0

    def test_real_ip_detected(self):
        result = detect_pii("server at 85.12.45.78")
        ips = [d for d in result if d.type == "ip_address"]
        assert len(ips) == 1

    def test_private_ip_not_detected(self):
        result = detect_pii("connect to 192.168.1.100")
        ips = [d for d in result if d.type == "ip_address"]
        assert len(ips) == 0

    def test_documentation_range_not_detected(self):
        result = detect_pii("example 203.0.113.42")
        ips = [d for d in result if d.type == "ip_address"]
        assert len(ips) == 0


class TestNHSContextCheck:
    def test_bare_digits_without_context_not_detected(self):
        result = detect_pii("reference 9434765919")
        nhs = [d for d in result if d.type == "nhs_number"]
        assert len(nhs) == 0

    def test_with_context_detected(self):
        result = detect_pii("NHS 943 476 5919", PIIDetectOptions(types=["nhs_number"]))
        nhs = [d for d in result if d.type == "nhs_number"]
        assert len(nhs) == 1

    def test_formatted_always_detected(self):
        result = detect_pii("ref 943 476 5919", PIIDetectOptions(types=["nhs_number"]))
        nhs = [d for d in result if d.type == "nhs_number"]
        assert len(nhs) == 1


class TestAadhaarContextCheck:
    def test_bare_digits_without_context_not_detected(self):
        result = detect_pii("transaction 234567891234")
        aadhaar = [d for d in result if d.type == "aadhaar"]
        assert len(aadhaar) == 0

    def test_with_context_detected(self):
        result = detect_pii("aadhaar 2345 6789 1234", PIIDetectOptions(types=["aadhaar"]))
        aadhaar = [d for d in result if d.type == "aadhaar"]
        assert len(aadhaar) == 1


class TestMedicareContextCheck:
    def test_bare_digits_without_context_not_detected(self):
        result = detect_pii("code 2123456701", PIIDetectOptions(types=["medicare"]))
        medicare = [d for d in result if d.type == "medicare"]
        assert len(medicare) == 0

    def test_with_context_detected(self):
        result = detect_pii("medicare 2123 45670 1", PIIDetectOptions(types=["medicare"]))
        medicare = [d for d in result if d.type == "medicare"]
        assert len(medicare) == 1
