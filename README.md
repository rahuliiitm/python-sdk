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
- **Jailbreak Detection** — Catches role-play exploits, DAN prompts, and system prompt override attempts
- **Prompt Leakage Detection** — Flags when the LLM accidentally echoes back system instructions or internal context
- **Unicode Sanitizer** — Strips or blocks invisible characters, homoglyphs, and zero-width sequences used to bypass filters
- **Secret/Credential Detection** — Catches API keys, tokens, passwords, and connection strings before they reach the LLM
- **Topic Guard** — Block or warn when conversations drift into off-limits subjects you define
- **Output Safety Scanning** — Scans LLM responses for harmful instructions, unsafe code patterns, and dangerous content
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

On every LLM call, the SDK runs a 20-step pipeline split across pre-call and post-call phases:

**PRE-CALL:**

1. Unicode Sanitizer — strip/warn/block invisible characters
2. Model Policy Check — enforce allowed models and parameters
3. Cost Guard Pre-Check — estimate cost, check budgets
4. PII Detection (input) — scan for personal data
5. PII Redaction (input) — replace PII with placeholders
6. Secret Detection (input) — catch API keys, tokens, passwords
7. Injection Detection — score risk, warn/block
8. Jailbreak Detection — catch role-play exploits and system overrides
9. Content Filter (input) — check policy violations
10. Topic Guard — block off-limits subjects

**>>> LLM API Call >>>**

**POST-CALL:**

11. Content Filter (output) — scan response for policy violations
12. Output Safety Scan — check for harmful instructions or unsafe code
13. Prompt Leakage Detection — flag leaked system prompts
14. Schema Validation — enforce JSON structure
15. Secret Detection (output) — catch credentials in response
16. PII Detection (output) — defense-in-depth scan
17. PII De-redaction — restore original values
18. Cost Guard Post-Recording — track actual cost
19. Event Batching — queue enriched event
20. Guardrail Events — fire registered callbacks

## API

### `LaunchPromptly(api_key, endpoint, ...)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `LAUNCHPROMPTLY_API_KEY` env | Your LaunchPromptly API key |
| `endpoint` | `https://api.launchpromptly.dev` | API base URL |
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
        jailbreak={"enabled": True, "block_on_detect": True},
        prompt_leakage={"enabled": True},
        unicode_sanitizer={"enabled": True, "action": "strip"},
        secret_detection={"enabled": True, "block_on_detect": True},
        topic_guard={"enabled": True, "blocked_topics": ["politics", "medical-advice"]},
        output_safety={"enabled": True, "block_unsafe": True},
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

### Jailbreak Detection

```python
{
    "jailbreak": {
        "enabled": True,
        "block_on_detect": True,       # raise JailbreakError on detection
        "sensitivity": "medium",       # "low" | "medium" | "high"
        "on_detect": lambda result: print(f"Jailbreak: {result.technique}"),
    }
}
```

Catches DAN ("Do Anything Now") prompts, role-play exploits ("You are now EvilGPT"), system prompt override attempts, and similar jailbreak techniques. The detector runs pattern matching and structural analysis on the input.

### Prompt Leakage Detection

```python
{
    "prompt_leakage": {
        "enabled": True,
        "system_prompt": "You are a helpful assistant...",  # optional: provide for exact matching
        "sensitivity": "medium",       # "low" | "medium" | "high"
        "on_detect": lambda result: print(f"Leaked: {result.leaked_content}"),
    }
}
```

Scans LLM output for signs that the model is echoing back system instructions, internal context, or tool definitions. If you provide the `system_prompt`, the detector can do exact substring matching in addition to heuristic checks.

### Unicode Sanitizer

```python
{
    "unicode_sanitizer": {
        "enabled": True,
        "action": "strip",             # "strip" | "warn" | "block"
        "allow_emoji": True,           # keep standard emoji (default: True)
        "on_suspicious": lambda result: print(f"Found: {result.found}"),
    }
}
```

Detects and handles invisible characters (zero-width joiners, RTL overrides, homoglyphs, tag characters) that attackers use to sneak prompts past text-based filters. The `strip` action removes them silently, `warn` lets the request through but fires an event, and `block` rejects the request.

### Secret Detection

```python
{
    "secret_detection": {
        "enabled": True,
        "block_on_detect": True,       # raise SecretDetectedError
        "scan_response": True,         # also scan LLM output (default: True)
        "types": ["api_key", "aws_key", "github_token", "jwt", "connection_string", "private_key"],
        "on_detect": lambda secrets: print(f"Secrets found: {len(secrets)}"),
    }
}
```

Catches API keys, AWS credentials, GitHub tokens, JWTs, database connection strings, and private keys in both input and output. Uses pattern matching tuned to minimize false positives on normal text.

### Topic Guard

```python
{
    "topic_guard": {
        "enabled": True,
        "blocked_topics": ["politics", "medical-advice", "legal-advice", "financial-advice"],
        "action": "block",             # "block" | "warn"
        "custom_topics": [
            {"name": "competitor-discussion", "patterns": ["CompetitorCo", "their product"]},
        ],
        "on_violation": lambda result: print(f"Topic: {result.topic}"),
    }
}
```

Prevents the conversation from going into subjects you want to keep off-limits. Comes with built-in topic categories and supports custom topics defined by keyword patterns.

### Output Safety

```python
{
    "output_safety": {
        "enabled": True,
        "block_unsafe": True,          # raise OutputSafetyError
        "categories": ["harmful_instructions", "unsafe_code", "dangerous_content"],
        "on_detect": lambda result: print(f"Unsafe: {result.category}"),
    }
}
```

Scans LLM responses for harmful instructions (e.g., "how to build a weapon"), unsafe code patterns (e.g., `eval()` with user input, SQL without parameterization), and other dangerous content. This is separate from content filtering -- content filters check for policy violations like hate speech, while output safety checks for responses that could cause real-world harm if followed.

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
    JailbreakError,
    CostLimitError,
    ContentViolationError,
    ModelPolicyError,
    OutputSchemaError,
    OutputSafetyError,
    SecretDetectedError,
    TopicViolationError,
    UnicodeBlockError,
    StreamAbortError,
)

try:
    res = await wrapped.chat.completions.create(...)
except PromptInjectionError as e:
    print(f"Injection blocked: risk={e.analysis.risk_score}")
except JailbreakError as e:
    print(f"Jailbreak detected: technique={e.technique}")
except CostLimitError as e:
    print(f"Budget exceeded: {e.violation.violation_type}")
except ContentViolationError as e:
    print(f"Content violation: {e.violations}")
except SecretDetectedError as e:
    print(f"Secret found: types={e.secret_types}")
except TopicViolationError as e:
    print(f"Off-limits topic: {e.topic}")
except OutputSafetyError as e:
    print(f"Unsafe output: {e.category}")
```

## Guardrail Events

```python
lp = LaunchPromptly(
    api_key="lp_live_...",
    on={
        "pii.detected": lambda e: log("PII found", e.data),
        "injection.blocked": lambda e: alert("Injection blocked", e.data),
        "cost.exceeded": lambda e: alert("Budget exceeded", e.data),
        "jailbreak.detected": lambda e: log("Jailbreak attempt", e.data),
        "jailbreak.blocked": lambda e: alert("Jailbreak blocked", e.data),
        "unicode.suspicious": lambda e: log("Suspicious unicode", e.data),
        "secret.detected": lambda e: alert("Secret found in text", e.data),
        "topic.violated": lambda e: log("Off-limits topic", e.data),
        "output.unsafe": lambda e: alert("Unsafe output detected", e.data),
        "prompt.leaked": lambda e: alert("System prompt leaked", e.data),
    },
)
```

**Event types:** `pii.detected`, `pii.redacted`, `injection.detected`, `injection.blocked`, `jailbreak.detected`, `jailbreak.blocked`, `unicode.suspicious`, `secret.detected`, `topic.violated`, `output.unsafe`, `prompt.leaked`, `cost.exceeded`, `content.violated`, `schema.invalid`, `model.blocked`

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

## Privacy & data practices

### What the SDK reports

Events sent to your LaunchPromptly endpoint contain metadata only:

- Token counts (input, output, total)
- Model name (e.g. `gpt-4o`)
- Estimated cost (USD)
- Latency (ms)
- Guardrail trigger types and counts (e.g. "2 PII detections", "injection blocked")
- Injection risk score
- Whether redaction was applied (boolean)
- Timestamps
- Customer ID and feature name (if you provided them)

### What the SDK does not send

- Prompt text or response text (by default)
- PII values (emails, SSNs, phone numbers, etc.)
- Raw user content
- API keys or secrets
- File uploads or attachments
- IP addresses of your end users

### Optional fields

You can opt in to sending `prompt_preview` and `response_text` for debugging. When enabled, these are encrypted with AES-256-GCM at rest on the dashboard.

### No telemetry

The SDK does not phone home. It makes no telemetry calls to LaunchPromptly. Events go to your configured endpoint only. If you don't set an endpoint, no network calls happen at all.

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LAUNCHPROMPTLY_API_KEY` | API key (alternative to passing in constructor) |
| `LP_API_KEY` | Shorthand alias |

## License

MIT
