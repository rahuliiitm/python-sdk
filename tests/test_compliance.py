"""Tests for compliance module."""
import pytest

from launchpromptly._internal.compliance import (
    ComplianceContext,
    ComplianceOptions,
    ConsentTrackingOptions,
    DataRetentionOptions,
    GeofencingOptions,
    build_compliance_event_data,
    check_compliance,
)


# -- Consent tracking ----------------------------------------------------------

def test_consent_passes_when_recorded():
    result = check_compliance(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(enabled=True, require_consent=True),
        ),
        ComplianceContext(metadata={"consent": "true"}),
    )
    assert result.passed is True
    assert len(result.violations) == 0


def test_consent_fails_when_missing():
    result = check_compliance(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(enabled=True, require_consent=True),
        ),
        ComplianceContext(metadata={}),
    )
    assert result.passed is False
    assert len(result.violations) == 1
    assert result.violations[0].type == "missing_consent"


def test_consent_fails_when_false():
    result = check_compliance(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(enabled=True, require_consent=True),
        ),
        ComplianceContext(metadata={"consent": "false"}),
    )
    assert result.passed is False


def test_consent_uses_custom_field():
    result = check_compliance(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(
                enabled=True, require_consent=True, consent_field="user_agreed"
            ),
        ),
        ComplianceContext(metadata={"user_agreed": "yes"}),
    )
    assert result.passed is True


def test_consent_skips_when_not_enabled():
    result = check_compliance(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(enabled=False, require_consent=True),
        ),
        ComplianceContext(metadata={}),
    )
    assert result.passed is True


# -- Geofencing ----------------------------------------------------------------

def test_geofencing_passes_for_allowed_region():
    result = check_compliance(
        ComplianceOptions(
            geofencing=GeofencingOptions(allowed_regions=["us", "eu"]),
        ),
        ComplianceContext(region="us"),
    )
    assert result.passed is True


def test_geofencing_fails_for_blocked_region():
    result = check_compliance(
        ComplianceOptions(
            geofencing=GeofencingOptions(allowed_regions=["us", "eu"]),
        ),
        ComplianceContext(region="cn"),
    )
    assert result.passed is False
    assert result.violations[0].type == "region_blocked"


def test_geofencing_is_case_insensitive():
    result = check_compliance(
        ComplianceOptions(
            geofencing=GeofencingOptions(allowed_regions=["US"]),
        ),
        ComplianceContext(region="us"),
    )
    assert result.passed is True


def test_geofencing_skips_when_no_region_provided():
    result = check_compliance(
        ComplianceOptions(
            geofencing=GeofencingOptions(allowed_regions=["us"]),
        ),
        ComplianceContext(),
    )
    assert result.passed is True


# -- Multiple violations -------------------------------------------------------

def test_reports_all_violations_at_once():
    result = check_compliance(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(enabled=True, require_consent=True),
            geofencing=GeofencingOptions(allowed_regions=["us"]),
        ),
        ComplianceContext(metadata={}, region="cn"),
    )
    assert result.passed is False
    assert len(result.violations) == 2


# -- No options ----------------------------------------------------------------

def test_passes_when_options_is_none():
    result = check_compliance(None, ComplianceContext())
    assert result.passed is True


def test_passes_when_options_is_empty():
    result = check_compliance(ComplianceOptions(), ComplianceContext())
    assert result.passed is True


# -- build_compliance_event_data -----------------------------------------------

def test_build_records_consent_status():
    data = build_compliance_event_data(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(enabled=True),
        ),
        ComplianceContext(metadata={"consent": "true"}),
    )
    assert data.consent_recorded is True


def test_build_records_no_consent():
    data = build_compliance_event_data(
        ComplianceOptions(
            consent_tracking=ConsentTrackingOptions(enabled=True),
        ),
        ComplianceContext(metadata={}),
    )
    assert data.consent_recorded is False


def test_build_includes_region_and_retention_days():
    data = build_compliance_event_data(
        ComplianceOptions(
            data_retention=DataRetentionOptions(enabled=True, max_age_days=30),
        ),
        ComplianceContext(region="eu"),
    )
    assert data.data_region == "eu"
    assert data.retention_days == 30


def test_build_handles_undefined_options():
    data = build_compliance_event_data(None, ComplianceContext())
    assert data.consent_recorded is False
