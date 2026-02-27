from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    metadata: Optional[dict[str, str]] = None


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
    variables: Optional[dict[str, str]] = None


@dataclass
class WrapOptions:
    customer: Optional[object] = None  # callable returning CustomerContext
    feature: Optional[str] = None
    trace_id: Optional[str] = None
    span_name: Optional[str] = None
