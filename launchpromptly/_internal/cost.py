from __future__ import annotations

# Prices per 1,000 tokens
_MODEL_PRICING: dict[str, dict[str, float]] = {
    # OpenAI GPT-4o family
    "gpt-4o": {"input": 0.0025, "output": 0.01},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4": {"input": 0.01, "output": 0.03},
    "gpt-4-mini": {"input": 0.002, "output": 0.006},
    "gpt-4-turbo": {"input": 0.01, "output": 0.03},
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "o1": {"input": 0.015, "output": 0.06},
    "o1-mini": {"input": 0.003, "output": 0.012},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
    # Anthropic Claude
    "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
    "claude-3-5-sonnet-20241022": {"input": 0.003, "output": 0.015},
    "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
    "claude-opus-4-20250514": {"input": 0.015, "output": 0.075},
    "claude-3-5-haiku-latest": {"input": 0.0008, "output": 0.004},
    "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
    # Google Gemini
    "gemini-2.0-flash": {"input": 0.0001, "output": 0.0004},
    "gemini-2.0-flash-lite": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
    "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
    "gemini-1.5-flash-8b": {"input": 0.0000375, "output": 0.00015},
}


def calculate_event_cost(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate cost in USD from a live LLM event.

    Returns 0 if the model is not in the pricing table.
    """
    pricing = _MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    return (input_tokens / 1000) * pricing["input"] + (output_tokens / 1000) * pricing["output"]
