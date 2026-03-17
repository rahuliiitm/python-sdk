from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Literal, Optional, Union

if TYPE_CHECKING:
    from ._internal.content_filter import ContentFilterOptions
    from ._internal.context_engine import ContextEngineSecurityOptions
    from ._internal.conversation_guard import ConversationGuardOptions
    from ._internal.response_judge import ResponseJudgeSecurityOptions
    from ._internal.cost_guard import CostGuardOptions
    from ._internal.cot_guard import ChainOfThoughtGuardOptions
    from ._internal.injection import InjectionAnalysis, InjectionDetectorProvider
    from ._internal.jailbreak import JailbreakAnalysis, JailbreakDetectorProvider
    from ._internal.model_policy import ModelPolicyOptions
    from ._internal.output_safety import OutputSafetyCategory, OutputSafetyThreat
    from ._internal.pii import PIIDetection, PIIDetectorProvider, PIIType
    from ._internal.prompt_leakage import PromptLeakageResult
    from ._internal.redaction import RedactionStrategy
    from ._internal.schema_validator import OutputSchemaOptions
    from ._internal.secret_detection import CustomSecretPattern, SecretDetection
    from ._internal.tool_guard import ToolGuardOptions
    from ._internal.topic_guard import TopicDefinition, TopicViolation
    from ._internal.unicode_sanitizer import UnicodeScanResult


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
    endpoint: str = "https://api.launchpromptly.dev"
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
    allow_list: Optional[List[str]] = None
    confidence_thresholds: Optional[Dict[str, float]] = None


@dataclass
class CascadeThresholds:
    """Thresholds for cascade logic — skip ML when regex is confident."""

    skip_above: float = 0.85
    """Skip ML if regex risk score is above this value."""
    skip_below: float = 0.10
    """Skip ML if regex risk score is below this value."""


@dataclass
class InjectionSecurityOptions:
    enabled: Optional[bool] = None
    block_threshold: Optional[float] = None
    block_on_high_risk: Optional[bool] = None
    providers: Optional[List[InjectionDetectorProvider]] = None
    on_detect: Optional[Callable[[InjectionAnalysis], None]] = None
    merge_strategy: Optional[Literal["max", "weighted_average", "unanimous"]] = None
    cascade: Optional[bool] = None
    """Enable cascade: skip ML providers when regex gives a confident result."""
    cascade_thresholds: Optional[CascadeThresholds] = None
    """Thresholds for cascade skip logic."""


@dataclass
class JailbreakSecurityOptions:
    enabled: Optional[bool] = None
    block_threshold: Optional[float] = None
    warn_threshold: Optional[float] = None
    block_on_detection: Optional[bool] = None
    providers: Optional[List[JailbreakDetectorProvider]] = None
    on_detect: Optional[Callable[[JailbreakAnalysis], None]] = None
    merge_strategy: Optional[Literal["max", "weighted_average", "unanimous"]] = None
    cascade: Optional[bool] = None
    """Enable cascade: skip ML providers when regex gives a confident result."""
    cascade_thresholds: Optional[CascadeThresholds] = None
    """Thresholds for cascade skip logic."""


@dataclass
class UnicodeSanitizerSecurityOptions:
    enabled: Optional[bool] = None
    action: Optional[Literal["strip", "warn", "block"]] = None
    detect_homoglyphs: Optional[bool] = None
    on_detect: Optional[Callable[[UnicodeScanResult], None]] = None


@dataclass
class SecretDetectionSecurityOptions:
    enabled: Optional[bool] = None
    built_in_patterns: Optional[bool] = None
    custom_patterns: Optional[List[CustomSecretPattern]] = None
    providers: Optional[List[Any]] = None
    """Pluggable secret detection providers (e.g., ML-based)."""
    scan_response: Optional[bool] = None
    action: Optional[Literal["warn", "block", "redact"]] = None
    on_detect: Optional[Callable[[List[SecretDetection]], None]] = None


@dataclass
class TopicGuardSecurityOptions:
    enabled: Optional[bool] = None
    allowed_topics: Optional[List[TopicDefinition]] = None
    blocked_topics: Optional[List[TopicDefinition]] = None
    action: Optional[Literal["warn", "block"]] = None
    on_violation: Optional[Callable[[TopicViolation], None]] = None


@dataclass
class OutputSafetySecurityOptions:
    enabled: Optional[bool] = None
    categories: Optional[List[OutputSafetyCategory]] = None
    action: Optional[Literal["warn", "block"]] = None
    on_detect: Optional[Callable[[List[OutputSafetyThreat]], None]] = None


@dataclass
class PromptLeakageSecurityOptions:
    enabled: Optional[bool] = None
    system_prompt: str = ""
    threshold: Optional[float] = None
    block_on_leak: Optional[bool] = None
    on_detect: Optional[Callable[[PromptLeakageResult], None]] = None


@dataclass
class HallucinationSecurityOptions:
    """Configuration for hallucination detection guardrail."""

    enabled: Optional[bool] = None
    source_text: Optional[str] = None
    """Explicit reference text to compare the LLM response against."""
    extract_from_system_prompt: bool = True
    """Auto-extract source from system message if no explicit source_text given."""
    threshold: float = 0.5
    """Faithfulness score threshold (0-1). Below this = hallucination detected."""
    block_on_detection: bool = False
    """Block the response when hallucination is detected."""
    on_detect: Optional[Callable[[Any], None]] = None
    providers: Optional[List[Any]] = None
    """Pluggable hallucination detection providers (e.g., ML cross-encoder)."""


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

    mode: Optional[Literal["enforce", "shadow"]] = None
    preset: Optional[Literal["strict", "balanced", "permissive"]] = None
    use_ml: Optional[Union[bool, List[str]]] = None
    """Auto-create and register ML-powered providers for guardrails.

    - True: enable ML for all available guardrails
    - False (default): regex/rule-based only
    - List[str]: enable ML only for listed guardrails

    Valid names: 'injection', 'jailbreak', 'pii', 'toxicity' (alias: 'content_filter'), 'hallucination'

    Requires: pip install launchpromptly[ml]
    """
    pii: Optional[PIISecurityOptions] = None
    injection: Optional[InjectionSecurityOptions] = None
    jailbreak: Optional[JailbreakSecurityOptions] = None
    cost_guard: Optional[CostGuardOptions] = None
    content_filter: Optional[ContentFilterOptions] = None
    model_policy: Optional[ModelPolicyOptions] = None
    stream_guard: Optional[StreamGuardOptions] = None
    output_schema: Optional[OutputSchemaOptions] = None
    unicode_sanitizer: Optional[UnicodeSanitizerSecurityOptions] = None
    secret_detection: Optional[SecretDetectionSecurityOptions] = None
    topic_guard: Optional[TopicGuardSecurityOptions] = None
    output_safety: Optional[OutputSafetySecurityOptions] = None
    prompt_leakage: Optional[PromptLeakageSecurityOptions] = None
    hallucination: Optional[HallucinationSecurityOptions] = None
    tool_guard: Optional[ToolGuardOptions] = None
    chain_of_thought: Optional[ChainOfThoughtGuardOptions] = None
    context_engine: Optional[ContextEngineSecurityOptions] = None
    response_judge: Optional[ResponseJudgeSecurityOptions] = None
    audit: Optional[AuditOptions] = None


# ── Guardrail Events ──────────────────────────────────────────────────────────

#: All guardrail event types emitted by the SDK.
GuardrailEventType = Literal[
    "pii.detected",
    "pii.redacted",
    "injection.detected",
    "injection.blocked",
    "jailbreak.detected",
    "jailbreak.blocked",
    "cost.exceeded",
    "content.violated",
    "schema.invalid",
    "model.blocked",
    "unicode.suspicious",
    "hallucination.detected",
    "hallucination.blocked",
    "secret.detected",
    "topic.violated",
    "output.unsafe",
    "prompt.leaked",
    "tool.blocked",
    "tool.violation",
    "cot.injection",
    "cot.system_leak",
    "cot.goal_drift",
    "conversation.max_turns",
    "conversation.topic_drift",
    "conversation.risk_threshold",
    "conversation.agent_loop",
    "conversation.pii_spread",
    "context.extracted",
    "context.violation",
    "response.boundary_violation",
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
