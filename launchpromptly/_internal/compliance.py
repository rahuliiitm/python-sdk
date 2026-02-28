"""
Compliance module -- GDPR/CCPA/HIPAA helpers.
Consent tracking, data retention, and geofencing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

ComplianceViolationType = Literal[
    "missing_consent",
    "region_blocked",
    "retention_exceeded",
]


@dataclass
class DataRetentionOptions:
    enabled: Optional[bool] = None
    max_age_days: Optional[int] = None


@dataclass
class ConsentTrackingOptions:
    enabled: Optional[bool] = None
    require_consent: Optional[bool] = None
    consent_field: Optional[str] = None  # Default: 'consent'


@dataclass
class GeofencingOptions:
    allowed_regions: Optional[List[str]] = None
    block_on_violation: Optional[bool] = None


@dataclass
class ComplianceOptions:
    data_retention: Optional[DataRetentionOptions] = None
    consent_tracking: Optional[ConsentTrackingOptions] = None
    geofencing: Optional[GeofencingOptions] = None


@dataclass
class ComplianceViolation:
    type: ComplianceViolationType
    message: str


@dataclass
class ComplianceCheckResult:
    passed: bool
    violations: List[ComplianceViolation]


@dataclass
class ComplianceEventData:
    consent_recorded: bool
    data_region: Optional[str] = None
    retention_days: Optional[int] = None


@dataclass
class ComplianceContext:
    metadata: Optional[Dict[str, str]] = None
    region: Optional[str] = None


def check_compliance(
    options: Optional[ComplianceOptions],
    context: ComplianceContext,
) -> ComplianceCheckResult:
    """Check compliance requirements before an LLM call."""
    if options is None:
        return ComplianceCheckResult(passed=True, violations=[])

    violations: List[ComplianceViolation] = []

    # Consent tracking
    if (
        options.consent_tracking is not None
        and options.consent_tracking.enabled
        and options.consent_tracking.require_consent
    ):
        consent_field = options.consent_tracking.consent_field or "consent"
        consent_value = (context.metadata or {}).get(consent_field)

        if not consent_value or consent_value in ("false", "0"):
            violations.append(
                ComplianceViolation(
                    type="missing_consent",
                    message=f'Consent not recorded. Set metadata.{consent_field} = "true" in request context.',
                )
            )

    # Geofencing
    if (
        options.geofencing is not None
        and options.geofencing.allowed_regions
        and context.region
    ):
        allowed = [r.lower() for r in options.geofencing.allowed_regions]
        if context.region.lower() not in allowed:
            violations.append(
                ComplianceViolation(
                    type="region_blocked",
                    message=f'Region "{context.region}" is not in allowed regions: {", ".join(allowed)}.',
                )
            )

    return ComplianceCheckResult(
        passed=len(violations) == 0,
        violations=violations,
    )


def build_compliance_event_data(
    options: Optional[ComplianceOptions],
    context: ComplianceContext,
) -> ComplianceEventData:
    """Build compliance event data for the event payload."""
    if options is None:
        return ComplianceEventData(consent_recorded=False)

    consent_field = (
        options.consent_tracking.consent_field
        if options.consent_tracking and options.consent_tracking.consent_field
        else "consent"
    )
    consent_value = (context.metadata or {}).get(consent_field)
    consent_recorded = bool(consent_value) and consent_value not in ("false", "0")

    return ComplianceEventData(
        consent_recorded=consent_recorded,
        data_region=context.region,
        retention_days=(
            options.data_retention.max_age_days
            if options.data_retention
            else None
        ),
    )
