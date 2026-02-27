# launchpromptly

Official Python SDK for [LaunchPromptly](https://launchpromptly.dev) — manage, version, and deploy AI prompts without redeploying your app.

## Install

```bash
pip install launchpromptly
```

## Quick Start

```python
import asyncio
from launchpromptly import LaunchPromptly

lp = LaunchPromptly(api_key="lp_live_...")

async def main():
    # Fetch a managed prompt (cached, with stale-while-error fallback)
    system_prompt = await lp.prompt("onboarding-assistant", variables={"userName": "Alice"})

    # Use with any LLM provider
    response = await openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "How do I reset my password?"},
        ],
    )

asyncio.run(main())
```

## Features

- **Prompt fetching** — `await lp.prompt(slug)` fetches the active deployed version
- **Template variables** — `{{variable}}` placeholders are interpolated at runtime
- **Caching** — prompts are cached in-memory with configurable TTL (default 60s) and LRU eviction
- **Stale-while-error** — returns expired cache on network failure (404s always throw)
- **Auto-tracking** — wrap your OpenAI client to automatically track cost, latency, and tokens
- **Event batching** — LLM events are batched and sent asynchronously
- **Context propagation** — `with lp.context(trace_id=...)` propagates context via `contextvars`
- **Singleton pattern** — `LaunchPromptly.init()` / `LaunchPromptly.shared()` for app-wide usage

## API

### `LaunchPromptly(api_key, endpoint, ...)`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | `LAUNCHPROMPTLY_API_KEY` env | Your LaunchPromptly API key |
| `endpoint` | `https://api.launchpromptly.dev` | API base URL |
| `prompt_cache_ttl` | `60.0` | Prompt cache TTL in seconds |
| `flush_at` | `10` | Batch size threshold for auto-flush |
| `flush_interval` | `5.0` | Timer interval for auto-flush (seconds) |
| `max_cache_size` | `1000` | Maximum cached prompts (LRU eviction) |

### `await lp.prompt(slug, *, customer_id, variables)`

Fetch a managed prompt by slug. Returns the interpolated content string.

```python
content = await lp.prompt("my-prompt", variables={"name": "Alice", "role": "admin"}, customer_id="user-42")
```

### `lp.wrap(client, options)`

Wrap an OpenAI client to automatically capture LLM events.

```python
from launchpromptly.types import WrapOptions

wrapped = lp.wrap(openai_client, WrapOptions(
    feature="chat",
    trace_id="req-abc-123",
    span_name="generate",
))
```

### `with lp.context(trace_id, customer_id, feature, span_name, metadata)`

Context manager for request-scoped context propagation via `contextvars`.

```python
with lp.context(trace_id="req-123", customer_id="user-42"):
    prompt = await lp.prompt("greeting")
    result = await wrapped.chat.completions.create(...)
    # trace_id and customer_id flow to all SDK calls inside this block
```

### Singleton

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

## Environment Variables

| Variable | Description |
|----------|-------------|
| `LAUNCHPROMPTLY_API_KEY` | API key (alternative to passing in constructor) |
| `LP_API_KEY` | Shorthand alias |

## License

MIT
