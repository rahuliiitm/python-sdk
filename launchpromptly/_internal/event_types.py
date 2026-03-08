from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


@dataclass
class PIIDetailEntry:
    type: str
    start: int
    end: int
    confidence: float


@dataclass
class PIIDetectionsPayload:
    input_count: int
    output_count: int
    types: List[str]
    redaction_applied: bool
    detector_used: Literal["regex", "ml", "both"]
    input_details: Optional[List[PIIDetailEntry]] = None
    output_details: Optional[List[PIIDetailEntry]] = None


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
class StreamGuardEventPayload:
    aborted: bool
    violation_count: int
    violation_types: List[str]
    approximate_output_tokens: int
    response_length: int


@dataclass
class JailbreakRiskPayload:
    score: float
    triggered: List[str]
    action: Literal["allow", "warn", "block"]
    decoded_payloads: Optional[List[str]] = None


@dataclass
class UnicodeThreatsPayload:
    found: bool
    threat_count: int
    threat_types: List[str]
    action: Literal["strip", "warn", "block"]


@dataclass
class SecretDetectionsPayload:
    input_count: int
    output_count: int
    types: List[str]


@dataclass
class TopicViolationPayload:
    type: Literal["off_topic", "blocked_topic"]
    topic: Optional[str]
    matched_keywords: List[str]
    score: float


@dataclass
class OutputSafetyPayload:
    threat_count: int
    categories: List[str]
    threats: List[Dict[str, str]]


@dataclass
class PromptLeakagePayload:
    leaked: bool
    similarity: float
    meta_response_detected: bool


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
    response_text: Optional[str] = None
    status_code: Optional[int] = None
    trace_id: Optional[str] = None
    span_name: Optional[str] = None
    environment_id: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None

    # Security metadata
    pii_detections: Optional[PIIDetectionsPayload] = None
    injection_risk: Optional[InjectionRiskPayload] = None
    cost_guard: Optional[CostGuardPayload] = None
    content_violations: Optional[ContentViolationsPayload] = None
    stream_guard: Optional[StreamGuardEventPayload] = None
    jailbreak_risk: Optional[JailbreakRiskPayload] = None
    unicode_threats: Optional[UnicodeThreatsPayload] = None
    secret_detections: Optional[SecretDetectionsPayload] = None
    topic_violation: Optional[TopicViolationPayload] = None
    output_safety: Optional[OutputSafetyPayload] = None
    prompt_leakage: Optional[PromptLeakagePayload] = None

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
        if self.response_text is not None:
            d["responseText"] = self.response_text
        if self.status_code is not None:
            d["statusCode"] = self.status_code
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
            pii_dict: Dict[str, Any] = {
                "inputCount": self.pii_detections.input_count,
                "outputCount": self.pii_detections.output_count,
                "types": self.pii_detections.types,
                "redactionApplied": self.pii_detections.redaction_applied,
                "detectorUsed": self.pii_detections.detector_used,
            }
            if self.pii_detections.input_details is not None:
                pii_dict["inputDetails"] = [
                    {"type": e.type, "start": e.start, "end": e.end, "confidence": e.confidence}
                    for e in self.pii_detections.input_details
                ]
            if self.pii_detections.output_details is not None:
                pii_dict["outputDetails"] = [
                    {"type": e.type, "start": e.start, "end": e.end, "confidence": e.confidence}
                    for e in self.pii_detections.output_details
                ]
            d["piiDetections"] = pii_dict
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
        if self.stream_guard is not None:
            d["streamGuard"] = {
                "aborted": self.stream_guard.aborted,
                "violationCount": self.stream_guard.violation_count,
                "violationTypes": self.stream_guard.violation_types,
                "approximateOutputTokens": self.stream_guard.approximate_output_tokens,
                "responseLength": self.stream_guard.response_length,
            }
        if self.jailbreak_risk is not None:
            jr: Dict[str, Any] = {
                "score": self.jailbreak_risk.score,
                "triggered": self.jailbreak_risk.triggered,
                "action": self.jailbreak_risk.action,
            }
            if self.jailbreak_risk.decoded_payloads is not None:
                jr["decodedPayloads"] = self.jailbreak_risk.decoded_payloads
            d["jailbreakRisk"] = jr
        if self.unicode_threats is not None:
            d["unicodeThreats"] = {
                "found": self.unicode_threats.found,
                "threatCount": self.unicode_threats.threat_count,
                "threatTypes": self.unicode_threats.threat_types,
                "action": self.unicode_threats.action,
            }
        if self.secret_detections is not None:
            d["secretDetections"] = {
                "inputCount": self.secret_detections.input_count,
                "outputCount": self.secret_detections.output_count,
                "types": self.secret_detections.types,
            }
        if self.topic_violation is not None:
            tv: Dict[str, Any] = {
                "type": self.topic_violation.type,
                "matchedKeywords": self.topic_violation.matched_keywords,
                "score": self.topic_violation.score,
            }
            if self.topic_violation.topic is not None:
                tv["topic"] = self.topic_violation.topic
            d["topicViolation"] = tv
        if self.output_safety is not None:
            d["outputSafety"] = {
                "threatCount": self.output_safety.threat_count,
                "categories": self.output_safety.categories,
                "threats": self.output_safety.threats,
            }
        if self.prompt_leakage is not None:
            d["promptLeakage"] = {
                "leaked": self.prompt_leakage.leaked,
                "similarity": self.prompt_leakage.similarity,
                "metaResponseDetected": self.prompt_leakage.meta_response_detected,
            }

        return d
