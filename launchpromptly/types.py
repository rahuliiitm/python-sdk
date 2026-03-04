from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional

if TYPE_CHECKING:
    from ._internal.content_filter import ContentFilterOptions
    from ._internal.cost_guard import CostGuardOptions
    from ._internal.injection import InjectionAnalysis, InjectionDetectorProvider
    from ._internal.model_policy import ModelPolicyOptions
    from ._internal.pii import PIIDetection, PIIDetectorProvider, PIIType
    from ._internal.redaction import RedactionStrategy
    from ._internal.schema_validator import OutputSchemaOptions


@dataclass
class CustomerContext:
    id: str
    feature: Optional[str] = None


@dataclass
class RequestContext:
    trace_id: Optional[str] = None
    span_name: Optional[str] = None
    customer_id: Optional[str] = None
    feature: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


@dataclass
class LaunchPromptlyOptions:
    api_key: Optional[str] = None
    endpoint: str = "https://launchpromptly-api-950530830180.us-west1.run.app"
    flush_at: int = 10
    flush_interval: float = 5.0  # seconds
    on: Optional[GuardrailEventHandlers] = None  # guardrail event handlers


@dataclass
class PIISecurityOptions:
    enabled: Optional[bool] = None
    redaction: Optional[RedactionStrategy] = None
    types: Optional[List[PIIType]] = None
    scan_response: Optional[bool] = None
    providers: Optional[List[PIIDetectorProvider]] = None
    on_detect: Optional[Callable[[List[PIIDetection]], None]] = None


@dataclass
class InjectionSecurityOptions:
    enabled: Optional[bool] = None
    block_threshold: Optional[float] = None
    block_on_high_risk: Optional[bool] = None
    providers: Optional[List[InjectionDetectorProvider]] = None
    on_detect: Optional[Callable[[InjectionAnalysis], None]] = None


@dataclass
class AuditOptions:
    log_level: Optional[str] = None  # 'none' | 'summary' | 'detailed'


@dataclass
class MaxResponseLength:
    """Response length limits for streaming guard."""

    max_chars: Optional[int] = None
    max_words: Optional[int] = None


@dataclass
class StreamViolation:
    """A violation detected during streaming."""

    type: Literal["pii", "injection", "length"]
    offset: int
    details: Any
    timestamp: float


@dataclass
class StreamGuardOptions:
    """Configuration for real-time streaming guard."""

    pii_scan: Optional[bool] = None
    injection_scan: Optional[bool] = None
    max_response_length: Optional[MaxResponseLength] = None
    scan_interval: int = 500
    window_overlap: int = 200
    on_violation: Literal["abort", "warn", "flag"] = "flag"
    on_stream_violation: Optional[Callable[[StreamViolation], None]] = None
    final_scan: bool = True
    track_tokens: bool = True


@dataclass
class SecurityOptions:
    """Security configuration for the wrap() pipeline."""

    pii: Optional[PIISecurityOptions] = None
    injection: Optional[InjectionSecurityOptions] = None
    cost_guard: Optional[CostGuardOptions] = None
    content_filter: Optional[ContentFilterOptions] = None
    model_policy: Optional[ModelPolicyOptions] = None
    stream_guard: Optional[StreamGuardOptions] = None
    output_schema: Optional[OutputSchemaOptions] = None
    audit: Optional[AuditOptions] = None


# ── Guardrail Events ──────────────────────────────────────────────────────────

#: All guardrail event types emitted by the SDK.
GuardrailEventType = Literal[
    "pii.detected",
    "pii.redacted",
    "injection.detected",
    "injection.blocked",
    "cost.exceeded",
    "content.violated",
    "schema.invalid",
    "model.blocked",
]


@dataclass
class GuardrailEvent:
    """Payload emitted when a guardrail event fires."""

    type: GuardrailEventType
    timestamp: float
    data: Dict[str, Any]


#: Map of event type → handler callback.
GuardrailEventHandlers = Dict[str, Callable[[GuardrailEvent], None]]


@dataclass
class WrapOptions:
    customer: Optional[object] = None  # callable returning CustomerContext
    feature: Optional[str] = None
    trace_id: Optional[str] = None
    span_name: Optional[str] = None
    security: Optional[SecurityOptions] = None
