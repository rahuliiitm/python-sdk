from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from ._internal.compliance import ComplianceOptions
    from ._internal.content_filter import ContentFilterOptions
    from ._internal.cost_guard import CostGuardOptions
    from ._internal.injection import InjectionAnalysis, InjectionDetectorProvider
    from ._internal.pii import PIIDetection, PIIDetectorProvider, PIIType
    from ._internal.redaction import RedactionStrategy


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
    prompt_cache_ttl: float = 60.0  # seconds
    max_cache_size: int = 1000


@dataclass
class PromptOptions:
    customer_id: Optional[str] = None
    variables: Optional[Dict[str, str]] = None


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
class SecurityOptions:
    """Security configuration for the wrap() pipeline."""

    pii: Optional[PIISecurityOptions] = None
    injection: Optional[InjectionSecurityOptions] = None
    cost_guard: Optional[CostGuardOptions] = None
    content_filter: Optional[ContentFilterOptions] = None
    compliance: Optional[ComplianceOptions] = None
    audit: Optional[AuditOptions] = None


@dataclass
class WrapOptions:
    customer: Optional[object] = None  # callable returning CustomerContext
    feature: Optional[str] = None
    trace_id: Optional[str] = None
    span_name: Optional[str] = None
    security: Optional[SecurityOptions] = None
