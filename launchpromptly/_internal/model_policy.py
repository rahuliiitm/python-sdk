"""Model policy enforcement — pre-call guard that validates LLM
request parameters against a configurable policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Union


@dataclass
class ModelPolicyOptions:
    """Configuration for the model policy guard."""

    allowed_models: Optional[List[str]] = None
    """Whitelist of allowed model identifiers. If set, calls to other models are blocked."""

    max_tokens: Optional[int] = None
    """Maximum allowed value for max_tokens. Requests exceeding this are blocked."""

    max_temperature: Optional[float] = None
    """Maximum allowed temperature. Requests exceeding this are blocked."""

    block_system_prompt_override: bool = False
    """When True, requests that include a system prompt are blocked."""

    on_violation: Optional[Callable[["ModelPolicyViolation"], None]] = None
    """Called when a policy violation is detected (before raising)."""


@dataclass
class ModelPolicyViolation:
    """Describes a single model policy violation."""

    rule: str  # 'model_not_allowed' | 'max_tokens_exceeded' | 'temperature_exceeded' | 'system_prompt_blocked'
    message: str
    actual: Optional[Union[str, int, float]] = None
    """The value that violated the policy."""
    limit: Optional[Union[str, int, float, List[str]]] = None
    """The policy limit that was violated."""


def check_model_policy(
    params: dict,
    options: ModelPolicyOptions,
) -> Optional[ModelPolicyViolation]:
    """Enforce model policy on an outgoing LLM request.

    Returns the first violation found, or ``None`` if the request passes all checks.
    Checks are run in order: model whitelist -> max tokens -> temperature -> system prompt.
    """
    model = params.get("model", "")

    # 1. Model whitelist
    if options.allowed_models:
        if not any(model == m or model.startswith(m + '-') for m in options.allowed_models):
            return ModelPolicyViolation(
                rule="model_not_allowed",
                message=f'Model "{model}" is not in the allowed list: {", ".join(options.allowed_models)}',
                actual=model,
                limit=options.allowed_models,
            )

    # 2. Max tokens cap
    max_tokens = params.get("max_tokens")
    if options.max_tokens is not None and max_tokens is not None:
        if max_tokens > options.max_tokens:
            return ModelPolicyViolation(
                rule="max_tokens_exceeded",
                message=f"max_tokens ({max_tokens}) exceeds policy limit ({options.max_tokens})",
                actual=max_tokens,
                limit=options.max_tokens,
            )

    # 3. Temperature cap
    temperature = params.get("temperature")
    if options.max_temperature is not None and temperature is not None:
        if temperature > options.max_temperature:
            return ModelPolicyViolation(
                rule="temperature_exceeded",
                message=f"temperature ({temperature}) exceeds policy limit ({options.max_temperature})",
                actual=temperature,
                limit=options.max_temperature,
            )

    # 4. Block system prompt override
    if options.block_system_prompt_override:
        messages = params.get("messages", [])
        has_system_message = any(m.get("role") == "system" for m in messages)
        has_system_field = params.get("system") is not None

        if has_system_message or has_system_field:
            return ModelPolicyViolation(
                rule="system_prompt_blocked",
                message="System prompts are blocked by model policy",
            )

    return None
