"""LaunchPromptly Python SDK — LLM privacy & security toolkit."""

from .client import LaunchPromptly
from .errors import (
    PromptNotFoundError,
    PromptInjectionError,
    CostLimitError,
    ContentViolationError,
    ComplianceError,
)
from .template import extract_variables, interpolate
from .types import (
    CustomerContext,
    LaunchPromptlyOptions,
    PIISecurityOptions,
    InjectionSecurityOptions,
    AuditOptions,
    SecurityOptions,
    PromptOptions,
    RequestContext,
    WrapOptions,
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
from ._internal.compliance import (
    check_compliance,
    build_compliance_event_data,
    ComplianceOptions,
    ComplianceCheckResult,
    ComplianceViolation,
    ComplianceEventData,
)

__all__ = [
    # Core
    "LaunchPromptly",
    "PromptNotFoundError",
    "extract_variables",
    "interpolate",
    "CustomerContext",
    "LaunchPromptlyOptions",
    "PromptOptions",
    "RequestContext",
    "WrapOptions",
    # Security types
    "PIISecurityOptions",
    "InjectionSecurityOptions",
    "AuditOptions",
    "SecurityOptions",
    # Security errors
    "PromptInjectionError",
    "CostLimitError",
    "ContentViolationError",
    "ComplianceError",
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
    # Compliance
    "check_compliance",
    "build_compliance_event_data",
    "ComplianceOptions",
    "ComplianceCheckResult",
    "ComplianceViolation",
    "ComplianceEventData",
]

__version__ = "0.1.0"
