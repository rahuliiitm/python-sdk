from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


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
    metadata: Optional[dict[str, str]] = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serialisable dict, omitting None values."""
        d: dict = {
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
        return d
