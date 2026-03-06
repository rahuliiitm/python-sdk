# launchpromptly

Official Python SDK for [LaunchPromptly](https://launchpromptly.dev) — runtime safety layer for LLM applications. PII redaction, prompt injection detection, cost guards, and content filtering with zero core dependencies.

## Install

```bash
pip install launchpromptly
```

For ML-enhanced detection (NER-based PII, semantic injection analysis):

```bash
pip install launchpromptly[ml]
```

## Quick Start

```python
from launchpromptly import LaunchPromptly
from openai import OpenAI

lp = LaunchPromptly(
    api_key="lp_live_...",
    security={
        "pii": {"enabled": True, "redaction": "placeholder"},
        "injection": {"enabled": True, "block_on_high_risk": True},
        "cost_guard": {"max_cost_per_request": 0.50},
    },
)

# Wrap your OpenAI client — all security features activate automatically
openai = lp.wrap(OpenAI())

response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": user_input}],
)
# If user_input contains "john@acme.com", the LLM receives "[EMAIL_1]"
# The response is de-redacted before being returned to your code

await lp.flush()  # On server shutdown
```

## Features

- **PII Redaction** — 16 built-in regex detectors (email, phone, SSN, credit card, IP, etc.) with pluggable ML providers
- **Prompt Injection Detection** — Rule-based scoring across 5 attack categories with configurable thresholds
- **Cost Guards** — Per-request, per-minute, per-hour, per-day, and per-customer budget limits
- **Content Filtering** — Block or warn on hate speech, violence, self-harm, and custom patterns
- **Model Policy** — Restrict which models, providers, and parameters are allowed
- **Output Schema Validation** — Validate LLM responses against JSON schemas
- **Streaming Guards** — Mid-stream PII scanning, injection detection, and response length limits
- **Multi-Provider** — Wrap OpenAI, Anthropic (`wrap_anthropic`), and Google Gemini (`wrap_gemini`) clients
- **Context Propagation** — `with lp.context(trace_id=...)` propagates context via `contextvars`
- **Singleton Pattern** — `LaunchPromptly.init()` / `LaunchPromptly.shared()` for app-wide usage
- **Zero Dependencies** — No runtime dependencies for core features
- **Event Dashboard** — Enriched security events sent to your LaunchPromptly dashboard

## Security Pipeline

On every LLM call, the SDK runs these checks in order:

1. Cost guard (estimate cost, check budgets)
2. PII scan & redact (replace PII with placeholders)
3. Injection detection (score risk, warn/block)
4. Content filter (check input policy violations)
5. **LLM API Call** (with redacted content)
6. Response PII scan (defense-in-depth)
7. Response content filter
8. Output schema validation
9. De-redact response (restore original values)
10. Send enriched event to dashboard

## API

### `LaunchPromptly(api_key, endpoint, ...)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `LAUNCHPROMPTLY_API_KEY` env | Your LaunchPromptly API key |
| `endpoint` | `https://launchpromptly-api-950530830180.us-west1.run.app` | API base URL |
| `flush_at` | `10` | Batch size threshold for auto-flush |
| `flush_interval` | `5.0` | Timer interval for auto-flush (seconds) |
| `on` | — | Guardrail event handlers |

### `lp.wrap(client, options?)` / `lp.wrap_anthropic(client)` / `lp.wrap_gemini(client)`

Wrap an LLM client with security guardrails.

```python
from launchpromptly.types import WrapOptions, SecurityOptions

wrapped = lp.wrap(OpenAI(), WrapOptions(
    feature="chat",
    security=SecurityOptions(
        pii={"enabled": True, "redaction": "placeholder"},
        injection={"enabled": True, "block_on_high_risk": True},
        cost_guard={"max_cost_per_request": 1.00},
        content_filter={"enabled": True, "categories": ["hate_speech", "violence"]},
        model_policy={"allowed_models": ["gpt-4o", "gpt-4o-mini"]},
        stream_guard={"pii_scan": True, "on_violation": "abort"},
        output_schema={"schema": my_json_schema, "strict": True},
    ),
))
```

### PII Redaction

```python
{
    "pii": {
        "enabled": True,
        "redaction": "placeholder",  # "placeholder" | "mask" | "hash" | "none"
        "types": ["email", "phone", "ssn", "credit_card", "ip_address"],
        "scan_response": True,
        "on_detect": lambda detections: print(f"Found {len(detections)} PII entities"),
    }
}
```

**Built-in PII types:** `email`, `phone`, `ssn`, `credit_card`, `ip_address`, `iban`, `drivers_license`, `uk_nino`, `nhs_number`, `passport`, `aadhaar`, `eu_phone`, `us_address`, `api_key`, `date_of_birth`, `medicare`

### Injection Detection

```python
{
    "injection": {
        "enabled": True,
        "block_threshold": 0.7,     # 0-1 risk score
        "block_on_high_risk": True,  # raise PromptInjectionError
        "on_detect": lambda analysis: print(f"Risk: {analysis.risk_score}"),
    }
}
```

### Cost Guards

```python
{
    "cost_guard": {
        "max_cost_per_request": 1.00,
        "max_cost_per_minute": 10.00,
        "max_cost_per_hour": 50.00,
        "max_cost_per_day": 200.00,
        "max_cost_per_customer": 5.00,
        "max_tokens_per_request": 100000,
        "block_on_exceed": True,
    }
}
```

### Context Propagation

```python
with lp.context(trace_id="req-123", customer_id="user-42"):
    # All SDK calls inside inherit the context
    result = await wrapped.chat.completions.create(...)
```

### Singleton Pattern

```python
# Initialize once at app startup
LaunchPromptly.init(api_key="lp_live_...")

# Access anywhere
lp = LaunchPromptly.shared()
```

### `await lp.flush()` / `await lp.shutdown()` / `lp.destroy()`

- `flush()` — send all pending events
- `shutdown()` — flush then destroy (for graceful server shutdown)
- `destroy()` — stop timers and release resources

## Error Handling

```python
from launchpromptly import (
    PromptInjectionError,
    CostLimitError,
    ContentViolationError,
    ModelPolicyError,
    OutputSchemaError,
    StreamAbortError,
)

try:
    res = await wrapped.chat.completions.create(...)
except PromptInjectionError as e:
    print(f"Injection blocked: risk={e.analysis.risk_score}")
except CostLimitError as e:
    print(f"Budget exceeded: {e.violation.violation_type}")
except ContentViolationError as e:
    print(f"Content violation: {e.violations}")
```

## Guardrail Events

```python
lp = LaunchPromptly(
    api_key="lp_live_...",
    on={
        "pii.detected": lambda e: log("PII found", e.data),
        "injection.blocked": lambda e: alert("Injection blocked", e.data),
        "cost.exceeded": lambda e: alert("Budget exceeded", e.data),
    },
)
```

**Event types:** `pii.detected`, `pii.redacted`, `injection.detected`, `injection.blocked`, `cost.exceeded`, `content.violated`, `schema.invalid`, `model.blocked`

## ML-Enhanced Detection (Optional)

The core SDK uses regex and rule-based detection — zero dependencies, sub-millisecond. For higher accuracy on obfuscated attacks and nuanced content, opt in to local ML models:

```bash
pip install launchpromptly[ml]
```

```python
from launchpromptly import LaunchPromptly
from launchpromptly.ml import MLToxicityDetector, MLInjectionDetector, PresidioPIIDetector

# Initialize detectors (first run downloads models)
toxicity = MLToxicityDetector()     # unitary/toxic-bert
injection = MLInjectionDetector()   # protectai/deberta-v3
pii = PresidioPIIDetector()         # Microsoft Presidio + spaCy NER

lp = LaunchPromptly(
    api_key="lp_live_...",
    security={
        "pii": {
            "enabled": True,
            "redaction": "placeholder",
            "providers": [pii],       # Adds NER: person names, orgs, locations
        },
        "injection": {
            "enabled": True,
            "providers": [injection], # Semantic injection detection via DeBERTa
        },
        "content_filter": {
            "enabled": True,
            "providers": [toxicity],  # ML toxicity: hate speech, threats, obscenity
        },
    },
)
```

### Layered Defense

ML providers **merge with** the built-in regex/rule detectors — they don't replace them:

| Layer | Speed | Catches | Dependencies |
|-------|-------|---------|-------------|
| **Layer 1: Regex/Rules** (always on) | <1ms | Obvious patterns — emails, SSNs, keyword injection | None |
| **Layer 2: Local ML** (opt-in) | <100ms | Obfuscated attacks, person names, nuanced hate speech | `transformers`, `presidio-analyzer` |

All ML inference runs locally — no data leaves your infrastructure.

### ML Detectors

| Detector | Model | What it adds |
|----------|-------|-------------|
| `MLToxicityDetector` | `unitary/toxic-bert` | Hate speech, threats, obscenity, identity attacks |
| `MLInjectionDetector` | `protectai/deberta-v3-base-prompt-injection-v2` | Semantic prompt injection (catches obfuscated/encoded attacks) |
| `PresidioPIIDetector` | Microsoft Presidio + spaCy | Person names, organization names, locations, medical records |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LAUNCHPROMPTLY_API_KEY` | API key (alternative to passing in constructor) |
| `LP_API_KEY` | Shorthand alias |

## License

MIT
