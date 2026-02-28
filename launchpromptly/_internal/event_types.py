from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class PIIDetectionsPayload:
    input_count: int
    output_count: int
    types: List[str]
    redaction_applied: bool
    detector_used: Literal["regex", "ml", "both"]


@dataclass
class InjectionRiskPayload:
    score: float
    triggered: List[str]
    action: Literal["allow", "warn", "block"]
    detector_used: Literal["rules", "ml", "both"]


@dataclass
class CostGuardPayload:
    estimated_cost: float
    budget_remaining: float
    limit_triggered: Optional[str] = None


@dataclass
class ContentViolationsPayload:
    input_violations: List[Dict[str, str]]
    output_violations: List[Dict[str, str]]


@dataclass
class CompliancePayload:
    consent_recorded: bool
    data_region: Optional[str] = None
    retention_days: Optional[int] = None


@dataclass
class IngestEventPayload:
    """Payload shape for a single LLM event sent to the LaunchPromptly API."""

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    customer_id: Optional[str] = None
    feature: Optional[str] = None
    system_hash: Optional[str] = None
    full_hash: Optional[str] = None
    prompt_preview: Optional[str] = None
    status_code: Optional[int] = None
    managed_prompt_id: Optional[str] = None
    prompt_version_id: Optional[str] = None
    trace_id: Optional[str] = None
    span_name: Optional[str] = None
    environment_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    # Security metadata
    pii_detections: Optional[PIIDetectionsPayload] = None
    injection_risk: Optional[InjectionRiskPayload] = None
    cost_guard: Optional[CostGuardPayload] = None
    content_violations: Optional[ContentViolationsPayload] = None
    compliance: Optional[CompliancePayload] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serialisable dict, omitting None values."""
        d: Dict[str, Any] = {
            "provider": self.provider,
            "model": self.model,
            "inputTokens": self.input_tokens,
            "outputTokens": self.output_tokens,
            "totalTokens": self.total_tokens,
            "costUsd": self.cost_usd,
            "latencyMs": self.latency_ms,
        }
        if self.customer_id is not None:
            d["customerId"] = self.customer_id
        if self.feature is not None:
            d["feature"] = self.feature
        if self.system_hash is not None:
            d["systemHash"] = self.system_hash
        if self.full_hash is not None:
            d["fullHash"] = self.full_hash
        if self.prompt_preview is not None:
            d["promptPreview"] = self.prompt_preview
        if self.status_code is not None:
            d["statusCode"] = self.status_code
        if self.managed_prompt_id is not None:
            d["managedPromptId"] = self.managed_prompt_id
        if self.prompt_version_id is not None:
            d["promptVersionId"] = self.prompt_version_id
        if self.trace_id is not None:
            d["traceId"] = self.trace_id
        if self.span_name is not None:
            d["spanName"] = self.span_name
        if self.environment_id is not None:
            d["environmentId"] = self.environment_id
        if self.metadata is not None:
            d["metadata"] = self.metadata

        # Security metadata (camelCase keys)
        if self.pii_detections is not None:
            d["piiDetections"] = {
                "inputCount": self.pii_detections.input_count,
                "outputCount": self.pii_detections.output_count,
                "types": self.pii_detections.types,
                "redactionApplied": self.pii_detections.redaction_applied,
                "detectorUsed": self.pii_detections.detector_used,
            }
        if self.injection_risk is not None:
            d["injectionRisk"] = {
                "score": self.injection_risk.score,
                "triggered": self.injection_risk.triggered,
                "action": self.injection_risk.action,
                "detectorUsed": self.injection_risk.detector_used,
            }
        if self.cost_guard is not None:
            cg: Dict[str, Any] = {
                "estimatedCost": self.cost_guard.estimated_cost,
                "budgetRemaining": self.cost_guard.budget_remaining,
            }
            if self.cost_guard.limit_triggered is not None:
                cg["limitTriggered"] = self.cost_guard.limit_triggered
            d["costGuard"] = cg
        if self.content_violations is not None:
            d["contentViolations"] = {
                "inputViolations": self.content_violations.input_violations,
                "outputViolations": self.content_violations.output_violations,
            }
        if self.compliance is not None:
            comp: Dict[str, Any] = {
                "consentRecorded": self.compliance.consent_recorded,
            }
            if self.compliance.data_region is not None:
                comp["dataRegion"] = self.compliance.data_region
            if self.compliance.retention_days is not None:
                comp["retentionDays"] = self.compliance.retention_days
            d["compliance"] = comp

        return d
