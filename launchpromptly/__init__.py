"""LaunchPromptly Python SDK — LLM privacy & security toolkit."""

from .client import LaunchPromptly
from .errors import (
    PromptInjectionError,
    CostLimitError,
    ContentViolationError,
    ModelPolicyError,
    OutputSchemaError,
    StreamAbortError,
)
from .types import (
    CustomerContext,
    LaunchPromptlyOptions,
    PIISecurityOptions,
    InjectionSecurityOptions,
    AuditOptions,
    SecurityOptions,
    RequestContext,
    WrapOptions,
    StreamGuardOptions,
    MaxResponseLength,
    StreamViolation,
    GuardrailEvent,
    GuardrailEventHandlers,
)

# Security modules — public API
from ._internal.pii import (
    detect_pii,
    merge_detections,
    RegexPIIDetector,
    PIIType,
    PIIDetection,
    PIIDetectOptions,
    PIIDetectorProvider,
    CustomPIIPattern,
    create_custom_detector,
)
from ._internal.redaction import (
    redact_pii,
    de_redact,
    RedactionStrategy,
    RedactionOptions,
    RedactionResult,
    MaskingOptions,
)
from ._internal.injection import (
    detect_injection,
    merge_injection_analyses,
    RuleInjectionDetector,
    InjectionAnalysis,
    InjectionOptions,
    InjectionDetectorProvider,
)
from ._internal.cost_guard import CostGuard, CostGuardOptions, BudgetViolation
from ._internal.model_policy import check_model_policy, ModelPolicyOptions, ModelPolicyViolation
from ._internal.schema_validator import (
    validate_schema,
    validate_output_schema,
    OutputSchemaOptions,
    SchemaValidationError,
    OutputValidationResult,
)
from ._internal.content_filter import (
    detect_content_violations,
    has_blocking_violation,
    RuleContentFilter,
    ContentCategory,
    ContentFilterOptions,
    CustomPattern,
    ContentViolation,
    ContentFilterProvider,
)
from ._internal.streaming import SecurityStream, StreamSecurityReport

# Provider adapters
from .providers.anthropic import (
    wrap_anthropic_client,
    extract_anthropic_message_texts,
    extract_anthropic_response_text,
    extract_anthropic_tool_calls,
    extract_anthropic_stream_chunk,
    extract_content_block_text,
)
from .providers.gemini import (
    wrap_gemini_client,
    extract_gemini_message_texts,
    extract_gemini_response_text,
    extract_gemini_function_calls,
    extract_gemini_stream_chunk,
    extract_gemini_content_text,
)

__all__ = [
    # Core
    "LaunchPromptly",
    "CustomerContext",
    "LaunchPromptlyOptions",
    "RequestContext",
    "WrapOptions",
    # Security types
    "PIISecurityOptions",
    "InjectionSecurityOptions",
    "AuditOptions",
    "SecurityOptions",
    "StreamGuardOptions",
    "MaxResponseLength",
    "StreamViolation",
    "GuardrailEvent",
    "GuardrailEventHandlers",
    # Security errors
    "PromptInjectionError",
    "CostLimitError",
    "ContentViolationError",
    "ModelPolicyError",
    "OutputSchemaError",
    "StreamAbortError",
    # PII
    "detect_pii",
    "merge_detections",
    "RegexPIIDetector",
    "PIIType",
    "PIIDetection",
    "PIIDetectOptions",
    "PIIDetectorProvider",
    "CustomPIIPattern",
    "create_custom_detector",
    # Redaction
    "redact_pii",
    "de_redact",
    "RedactionStrategy",
    "RedactionOptions",
    "RedactionResult",
    "MaskingOptions",
    # Injection
    "detect_injection",
    "merge_injection_analyses",
    "RuleInjectionDetector",
    "InjectionAnalysis",
    "InjectionOptions",
    "InjectionDetectorProvider",
    # Cost guard
    "CostGuard",
    "CostGuardOptions",
    "BudgetViolation",
    # Model policy
    "check_model_policy",
    "ModelPolicyOptions",
    "ModelPolicyViolation",
    # Schema validator
    "validate_schema",
    "validate_output_schema",
    "OutputSchemaOptions",
    "SchemaValidationError",
    "OutputValidationResult",
    # Content filter
    "detect_content_violations",
    "has_blocking_violation",
    "RuleContentFilter",
    "ContentCategory",
    "ContentFilterOptions",
    "CustomPattern",
    "ContentViolation",
    "ContentFilterProvider",
    # Streaming
    "SecurityStream",
    "StreamSecurityReport",
    # Provider adapters
    "wrap_anthropic_client",
    "extract_anthropic_message_texts",
    "extract_anthropic_response_text",
    "extract_anthropic_tool_calls",
    "extract_anthropic_stream_chunk",
    "extract_content_block_text",
    "wrap_gemini_client",
    "extract_gemini_message_texts",
    "extract_gemini_response_text",
    "extract_gemini_function_calls",
    "extract_gemini_stream_chunk",
    "extract_gemini_content_text",
]

__version__ = "0.1.0"
