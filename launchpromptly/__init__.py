"""LaunchPromptly Python SDK — LLM privacy & security toolkit."""

from .client import LaunchPromptly
from .errors import (
    PromptInjectionError,
    CostLimitError,
    ContentViolationError,
    ModelPolicyError,
    OutputSchemaError,
    StreamAbortError,
    JailbreakError,
    TopicViolationError,
    ToolGuardError,
    ChainOfThoughtError,
    ConversationGuardError,
)
from .types import (
    CustomerContext,
    LaunchPromptlyOptions,
    PIISecurityOptions,
    InjectionSecurityOptions,
    JailbreakSecurityOptions,
    CascadeThresholds,
    UnicodeSanitizerSecurityOptions,
    SecretDetectionSecurityOptions,
    TopicGuardSecurityOptions,
    OutputSafetySecurityOptions,
    PromptLeakageSecurityOptions,
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
from ._internal.jailbreak import (
    detect_jailbreak,
    merge_jailbreak_analyses,
    RuleJailbreakDetector,
    JailbreakAnalysis,
    JailbreakOptions,
    JailbreakDetectorProvider,
)
from ._internal.unicode_sanitizer import (
    scan_unicode,
    UnicodeScanResult,
    UnicodeThreat,
    UnicodeSanitizeOptions,
)
from ._internal.secret_detection import (
    detect_secrets,
    SecretDetection,
    SecretDetectionOptions,
    CustomSecretPattern,
)
from ._internal.topic_guard import (
    check_topic_guard,
    TopicDefinition,
    TopicViolation,
    TopicGuardOptions,
)
from ._internal.output_safety import (
    scan_output_safety,
    OutputSafetyThreat,
    OutputSafetyOptions,
    OutputSafetyCategory,
)
from ._internal.prompt_leakage import (
    detect_prompt_leakage,
    PromptLeakageResult,
    PromptLeakageOptions,
)
from ._internal.topic_templates import (
    competitor_endorsement,
    POLITICAL_BIAS,
    MEDICAL_ADVICE,
    LEGAL_ADVICE,
    FINANCIAL_ADVICE,
)
from ._internal.compliance import (
    ComplianceTemplate,
    HEALTHCARE_COMPLIANCE,
    FINANCE_COMPLIANCE,
    ECOMMERCE_COMPLIANCE,
    INSURANCE_COMPLIANCE,
)
from ._internal.streaming import SecurityStream, StreamSecurityReport
from ._internal.tool_guard import (
    check_tool_calls,
    detect_dangerous_args,
    scan_tool_result,
    ToolGuardOptions,
    ToolGuardViolation,
    ToolGuardCheckResult,
    ToolCallInfo,
    ToolResultScanReport,
    ToolResultThreat,
)
from ._internal.cot_guard import (
    scan_chain_of_thought,
    extract_reasoning_text,
    ChainOfThoughtGuardOptions,
    ChainOfThoughtViolation,
    ChainOfThoughtScanResult,
)
from ._internal.conversation_guard import (
    ConversationGuard,
    ConversationGuardOptions,
    ConversationGuardViolation,
    ConversationSummary,
    RecordTurnInput,
)

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
    "JailbreakSecurityOptions",
    "CascadeThresholds",
    "UnicodeSanitizerSecurityOptions",
    "SecretDetectionSecurityOptions",
    "TopicGuardSecurityOptions",
    "OutputSafetySecurityOptions",
    "PromptLeakageSecurityOptions",
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
    "JailbreakError",
    "TopicViolationError",
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
    # Jailbreak
    "detect_jailbreak",
    "merge_jailbreak_analyses",
    "RuleJailbreakDetector",
    "JailbreakAnalysis",
    "JailbreakOptions",
    "JailbreakDetectorProvider",
    # Unicode sanitizer
    "scan_unicode",
    "UnicodeScanResult",
    "UnicodeThreat",
    "UnicodeSanitizeOptions",
    # Secret detection
    "detect_secrets",
    "SecretDetection",
    "SecretDetectionOptions",
    "CustomSecretPattern",
    # Topic guard
    "check_topic_guard",
    "TopicDefinition",
    "TopicViolation",
    "TopicGuardOptions",
    # Output safety
    "scan_output_safety",
    "OutputSafetyThreat",
    "OutputSafetyOptions",
    "OutputSafetyCategory",
    # Topic templates
    "competitor_endorsement",
    "POLITICAL_BIAS",
    "MEDICAL_ADVICE",
    "LEGAL_ADVICE",
    "FINANCIAL_ADVICE",
    # Compliance templates
    "ComplianceTemplate",
    "HEALTHCARE_COMPLIANCE",
    "FINANCE_COMPLIANCE",
    "ECOMMERCE_COMPLIANCE",
    "INSURANCE_COMPLIANCE",
    # Prompt leakage
    "detect_prompt_leakage",
    "PromptLeakageResult",
    "PromptLeakageOptions",
    # Tool guard
    "check_tool_calls",
    "detect_dangerous_args",
    "scan_tool_result",
    "ToolGuardOptions",
    "ToolGuardViolation",
    "ToolGuardCheckResult",
    "ToolCallInfo",
    "ToolResultScanReport",
    "ToolResultThreat",
    "ToolGuardError",
    # Chain-of-thought guard
    "scan_chain_of_thought",
    "extract_reasoning_text",
    "ChainOfThoughtGuardOptions",
    "ChainOfThoughtViolation",
    "ChainOfThoughtScanResult",
    "ChainOfThoughtError",
    # Conversation guard
    "ConversationGuard",
    "ConversationGuardOptions",
    "ConversationGuardViolation",
    "ConversationSummary",
    "RecordTurnInput",
    "ConversationGuardError",
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
