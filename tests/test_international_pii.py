"""Tests for international PII detection patterns and custom PII registration."""
import pytest

from launchpromptly._internal.pii import (
    CustomPIIPattern,
    PIIDetectOptions,
    create_custom_detector,
    detect_pii,
)


# -- IBAN ----------------------------------------------------------------------

def test_detects_german_iban():
    result = detect_pii("IBAN: DE89370400440532013000")
    ibans = [d for d in result if d.type == "iban"]
    assert len(ibans) == 1
    assert ibans[0].value == "DE89370400440532013000"
    assert ibans[0].confidence == 0.9


def test_detects_uk_iban():
    result = detect_pii("Transfer to GB29NWBK60161331926819")
    ibans = [d for d in result if d.type == "iban"]
    assert len(ibans) == 1


def test_no_false_positive_iban_on_short_code():
    result = detect_pii("Code: AB12")
    ibans = [d for d in result if d.type == "iban"]
    assert len(ibans) == 0


# -- NHS Number ----------------------------------------------------------------

def test_detects_nhs_number_with_spaces():
    # Use NHS number starting with 0 to avoid overlap with US phone pattern
    # (US phone regex requires first digit 2-9)
    result = detect_pii("NHS number is 010 557 7104")
    nhs = [d for d in result if d.type == "nhs_number"]
    assert len(nhs) == 1
    assert nhs[0].confidence == 0.8


def test_detects_nhs_number_filtered():
    # When filtering to nhs_number type only, unambiguous detection
    result = detect_pii("NHS: 4505577104", PIIDetectOptions(types=["nhs_number"]))
    nhs = [d for d in result if d.type == "nhs_number"]
    assert len(nhs) == 1


def test_no_false_positive_nhs_on_9_digits():
    result = detect_pii("Number: 123 456 789")
    nhs = [d for d in result if d.type == "nhs_number"]
    assert len(nhs) == 0


# -- UK NINO -------------------------------------------------------------------

def test_detects_uk_nino():
    result = detect_pii("NI number: AB 12 34 56 C")
    ninos = [d for d in result if d.type == "uk_nino"]
    assert len(ninos) == 1
    assert ninos[0].confidence == 0.9


def test_detects_uk_nino_no_spaces():
    result = detect_pii("NINO: AB123456C")
    ninos = [d for d in result if d.type == "uk_nino"]
    assert len(ninos) == 1


def test_no_false_positive_nino_invalid_prefix():
    # D and F are not valid first characters
    result = detect_pii("Number: DA123456C")
    ninos = [d for d in result if d.type == "uk_nino"]
    assert len(ninos) == 0


# -- Passport ------------------------------------------------------------------

def test_detects_passport_single_letter():
    result = detect_pii("Passport: L12345678")
    passports = [d for d in result if d.type == "passport"]
    assert len(passports) == 1
    assert passports[0].confidence == 0.7


def test_detects_passport_two_letter():
    result = detect_pii("Passport number AB1234567")
    passports = [d for d in result if d.type == "passport"]
    assert len(passports) == 1


def test_no_false_positive_passport_too_few_digits():
    result = detect_pii("Code: AB12345")
    passports = [d for d in result if d.type == "passport"]
    assert len(passports) == 0


# -- Aadhaar -------------------------------------------------------------------

def test_detects_aadhaar_no_spaces():
    result = detect_pii("Aadhaar: 234567890123")
    aadhaar = [d for d in result if d.type == "aadhaar"]
    assert len(aadhaar) == 1
    assert aadhaar[0].confidence == 0.85


def test_detects_aadhaar_with_spaces():
    result = detect_pii("Aadhaar number is 2345 6789 0123")
    aadhaar = [d for d in result if d.type == "aadhaar"]
    assert len(aadhaar) == 1


def test_no_false_positive_aadhaar_on_8_digits():
    result = detect_pii("Number: 1234 5678")
    aadhaar = [d for d in result if d.type == "aadhaar"]
    assert len(aadhaar) == 0


# -- EU Phone ------------------------------------------------------------------

def test_detects_german_phone():
    result = detect_pii("Call +49 30 1234567")
    phones = [d for d in result if d.type == "eu_phone"]
    assert len(phones) == 1
    assert phones[0].confidence == 0.8


def test_detects_french_phone():
    result = detect_pii("Tel: +33 1 23 45 67 89")
    phones = [d for d in result if d.type == "eu_phone"]
    assert len(phones) == 1


def test_no_false_positive_eu_phone_wrong_country_code():
    # +1 is US, not in the EU pattern
    result = detect_pii("Call +1 555 123 4567")
    eu_phones = [d for d in result if d.type == "eu_phone"]
    assert len(eu_phones) == 0


# -- Medicare (AU) -------------------------------------------------------------

def test_detects_medicare_number():
    result = detect_pii("Medicare: 2123 45670 1")
    medicare = [d for d in result if d.type == "medicare"]
    assert len(medicare) == 1
    assert medicare[0].confidence == 0.75


def test_detects_medicare_no_spaces():
    # When filtering to medicare type only, avoids overlap with phone pattern
    result = detect_pii("Medicare number 2123456701", PIIDetectOptions(types=["medicare"]))
    medicare = [d for d in result if d.type == "medicare"]
    assert len(medicare) == 1


def test_no_false_positive_medicare_wrong_format():
    result = detect_pii("Code: 12345")
    medicare = [d for d in result if d.type == "medicare"]
    assert len(medicare) == 0


# -- Drivers License (US) -----------------------------------------------------

def test_detects_drivers_license():
    result = detect_pii("License: A123-4567-8901")
    dl = [d for d in result if d.type == "drivers_license"]
    assert len(dl) == 1
    assert dl[0].confidence == 0.75


def test_detects_drivers_license_different_letter():
    result = detect_pii("DL: M456-7890-1234")
    dl = [d for d in result if d.type == "drivers_license"]
    assert len(dl) == 1


def test_no_false_positive_drivers_license():
    result = detect_pii("Number: 1234-5678-9012")
    dl = [d for d in result if d.type == "drivers_license"]
    assert len(dl) == 0


# -- Type filtering with international types -----------------------------------

def test_filters_to_only_iban():
    text = "IBAN DE89370400440532013000 and Passport L12345678"
    result = detect_pii(text, PIIDetectOptions(types=["iban"]))
    assert all(d.type == "iban" for d in result)
    assert len(result) >= 1


def test_filters_to_only_aadhaar():
    text = "Aadhaar 2345 6789 0123 and NHS 4505577104"
    result = detect_pii(text, PIIDetectOptions(types=["aadhaar"]))
    assert all(d.type == "aadhaar" for d in result)


# -- Custom PII pattern registration ------------------------------------------

def test_custom_detector_employee_id():
    detector = create_custom_detector([
        CustomPIIPattern(
            name="Employee ID",
            type="employee_id",
            pattern=r"\bEMP-\d{5,8}\b",
            confidence=0.9,
        ),
    ])
    detections = detector.detect("Employee EMP-12345 works here")
    assert len(detections) == 1
    assert detections[0].type == "employee_id"
    assert detections[0].value == "EMP-12345"
    assert detections[0].confidence == 0.9


def test_custom_detector_no_match():
    detector = create_custom_detector([
        CustomPIIPattern(
            name="Employee ID",
            type="employee_id",
            pattern=r"\bEMP-\d{5,8}\b",
        ),
    ])
    detections = detector.detect("No employee IDs here")
    assert len(detections) == 0


def test_custom_detector_multiple_patterns():
    detector = create_custom_detector([
        CustomPIIPattern(
            name="Employee ID",
            type="employee_id",
            pattern=r"\bEMP-\d{5,8}\b",
            confidence=0.9,
        ),
        CustomPIIPattern(
            name="Internal Project Code",
            type="project_code",
            pattern=r"\bPRJ-[A-Z]{2,4}-\d{3}\b",
            confidence=0.85,
        ),
    ])
    text = "Employee EMP-12345 on project PRJ-AB-123"
    detections = detector.detect(text)
    assert len(detections) == 2
    types = {d.type for d in detections}
    assert "employee_id" in types
    assert "project_code" in types


def test_custom_detector_name_and_supported_types():
    detector = create_custom_detector([
        CustomPIIPattern(name="EmpID", type="employee_id", pattern=r"\bEMP-\d+\b"),
    ])
    assert detector.name == "custom"
    assert "employee_id" in detector.supported_types


def test_custom_detector_empty_text():
    detector = create_custom_detector([
        CustomPIIPattern(name="EmpID", type="employee_id", pattern=r"\bEMP-\d+\b"),
    ])
    assert detector.detect("") == []


def test_custom_detector_respects_type_filter():
    detector = create_custom_detector([
        CustomPIIPattern(name="EmpID", type="employee_id", pattern=r"\bEMP-\d+\b"),
        CustomPIIPattern(name="ProjCode", type="project_code", pattern=r"\bPRJ-\d+\b"),
    ])
    text = "EMP-123 and PRJ-456"
    detections = detector.detect(text, PIIDetectOptions(types=["employee_id"]))
    assert len(detections) == 1
    assert detections[0].type == "employee_id"
