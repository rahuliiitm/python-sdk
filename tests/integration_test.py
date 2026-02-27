#!/usr/bin/env python3
"""
LaunchPromptly Python SDK — Integration Test Suite

Self-bootstrapping: creates its own user, prompt, deployment, and API key,
then exercises every SDK feature against the live local API.

Prerequisites:
  - API server running on localhost:3001
  - Postgres running with migrations applied

Run:
  python tests/integration_test.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

# Ensure the package root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from launchpromptly import (
    LaunchPromptly,
    PromptNotFoundError,
    extract_variables,
    interpolate,
)
from launchpromptly.types import RequestContext
from tests.helpers import setup, teardown, api_call, TestContext

# ── Test Harness ──────────────────────────────────────────────────────────────

GREEN = "\033[32m"
RED = "\033[31m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"

passed = 0
failed = 0
failures: list[str] = []


def assert_true(label: str, condition: bool, detail: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"  {GREEN}✓{RESET} {label}")
    else:
        failed += 1
        msg = f"{label} — {detail}" if detail else label
        failures.append(msg)
        print(f"  {RED}✗{RESET} {label}" + (f" {DIM}({detail}){RESET}" if detail else ""))


def assert_equal(label: str, actual, expected) -> None:
    ok = actual == expected
    assert_true(label, ok, "" if ok else f"expected {expected!r}, got {actual!r}")


def assert_includes(label: str, haystack: str, needle: str) -> None:
    ok = needle in haystack
    assert_true(label, ok, "" if ok else f'"{needle}" not found in "{haystack[:100]}..."')


def assert_array_equal(label: str, actual: list, expected: list) -> None:
    ok = actual == expected
    assert_true(label, ok, "" if ok else f"expected {expected}, got {actual}")


async def assert_throws(label: str, fn, error_class=None) -> None:
    try:
        await fn()
        assert_true(label, False, "expected to throw but did not")
    except Exception as err:
        if error_class:
            assert_true(label, isinstance(err, error_class),
                        f"expected {error_class.__name__} but got {type(err).__name__}")
        else:
            assert_true(label, True)


def section(name: str) -> None:
    print(f"\n{BOLD}▸ {name}{RESET}")


# ── Test Suites ───────────────────────────────────────────────────────────────

API_ENDPOINT = os.environ.get("API_URL", "http://localhost:3001")


async def test_initialization(ctx: TestContext) -> None:
    section("Test 1: Initialization")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT)
    assert_true("Creates instance with explicit api_key + endpoint", isinstance(lp, LaunchPromptly))
    lp.destroy()

    # Missing API key raises
    orig = os.environ.pop("LAUNCHPROMPTLY_API_KEY", None)
    orig_lp = os.environ.pop("LP_API_KEY", None)
    try:
        threw = False
        try:
            LaunchPromptly()
        except ValueError:
            threw = True
        assert_true("Raises when no API key provided", threw)
    finally:
        if orig:
            os.environ["LAUNCHPROMPTLY_API_KEY"] = orig
        if orig_lp:
            os.environ["LP_API_KEY"] = orig_lp


async def test_singleton(ctx: TestContext) -> None:
    section("Test 2: Singleton Pattern")

    LaunchPromptly.reset()

    threw = False
    try:
        LaunchPromptly.shared()
    except RuntimeError:
        threw = True
    assert_true("shared() raises before init()", threw)

    lp = LaunchPromptly.init(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT)
    assert_true("init() returns an instance", isinstance(lp, LaunchPromptly))
    assert_equal("shared() returns the same instance", LaunchPromptly.shared(), lp)

    lp2 = LaunchPromptly.init(api_key="lp_live_different", endpoint="http://other:9999")
    assert_equal("Second init() returns existing instance", lp2, lp)

    content = await LaunchPromptly.shared().prompt(ctx.prompt_slug)
    assert_true("Singleton fetches prompt", len(content) > 0)

    LaunchPromptly.reset()
    threw = False
    try:
        LaunchPromptly.shared()
    except RuntimeError:
        threw = True
    assert_true("reset() clears the singleton", threw)


async def test_prompt_fetch(ctx: TestContext) -> None:
    section("Test 3: Prompt Fetch")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT)
    content = await lp.prompt(ctx.prompt_slug)
    assert_true("Fetches prompt successfully", isinstance(content, str) and len(content) > 0)
    assert_includes("Contains {{name}} placeholder", content, "{{name}}")
    assert_includes("Contains {{role}} placeholder", content, "{{role}}")
    assert_includes("Contains {{company}} placeholder", content, "{{company}}")
    lp.destroy()


async def test_template_variables(ctx: TestContext) -> None:
    section("Test 4: Template Variables")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT)
    content = await lp.prompt(ctx.prompt_slug, variables={"name": "Alice", "role": "admin", "company": "Acme Corp"})
    assert_equal("Interpolates name", content, "Hello Alice, you are a admin. Welcome to Acme Corp!")

    lp2 = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, prompt_cache_ttl=0.001)
    await asyncio.sleep(0.01)
    partial = await lp2.prompt(ctx.prompt_slug, variables={"name": "Bob"})
    assert_includes("Partial variables: name interpolated", partial, "Hello Bob")
    assert_includes("Partial variables: role left as placeholder", partial, "{{role}}")

    lp.destroy()
    lp2.destroy()


async def test_caching(ctx: TestContext) -> None:
    section("Test 5: Caching")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, prompt_cache_ttl=60.0)

    t1 = time.monotonic()
    first = await lp.prompt(ctx.prompt_slug)
    network_ms = (time.monotonic() - t1) * 1000

    t2 = time.monotonic()
    second = await lp.prompt(ctx.prompt_slug)
    cache_ms = (time.monotonic() - t2) * 1000

    assert_equal("Cached result matches first result", first, second)
    assert_true("Cached fetch is faster than network fetch",
                cache_ms < network_ms or cache_ms < 5,
                f"network: {network_ms:.1f}ms, cache: {cache_ms:.1f}ms")
    lp.destroy()


async def test_stale_cache_fallback(ctx: TestContext) -> None:
    section("Test 6: Stale Cache Fallback")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, prompt_cache_ttl=0.05)
    original = await lp.prompt(ctx.prompt_slug)
    assert_true("Primed cache with prompt", len(original) > 0)

    await asyncio.sleep(0.1)

    await assert_throws("404 throws PromptNotFoundError (no stale fallback)",
                        lambda: lp.prompt("nonexistent-slug-12345"), PromptNotFoundError)
    lp.destroy()


async def test_not_found_handling(ctx: TestContext) -> None:
    section("Test 7: 404 Handling")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT)

    await assert_throws("Throws PromptNotFoundError for missing slug",
                        lambda: lp.prompt("this-prompt-does-not-exist"), PromptNotFoundError)

    try:
        await lp.prompt("another-missing-slug")
    except PromptNotFoundError as err:
        assert_includes("Error message includes slug name", str(err), "another-missing-slug")

    lp.destroy()


async def test_template_utilities() -> None:
    section("Test 8: Template Utilities")

    vars_ = extract_variables("Hello {{name}}, your role is {{role}}. Email: {{email}}")
    assert_array_equal("extract_variables finds all variables", vars_, ["name", "role", "email"])

    empty = extract_variables("No variables here")
    assert_array_equal("extract_variables returns empty for no variables", empty, [])

    dupes = extract_variables("{{x}} and {{x}} again")
    assert_array_equal("extract_variables deduplicates", dupes, ["x"])

    result = interpolate("Hi {{name}}, welcome to {{place}}!", {"name": "World", "place": "Earth"})
    assert_equal("interpolate replaces all variables", result, "Hi World, welcome to Earth!")

    partial = interpolate("{{a}} + {{b}} = {{c}}", {"a": "1", "b": "2"})
    assert_equal("interpolate leaves unmatched as-is", partial, "1 + 2 = {{c}}")

    special = interpolate("Price: {{amount}}", {"amount": "$100.00 (USD)"})
    assert_equal("interpolate handles special chars in values", special, "Price: $100.00 (USD)")


async def test_openai_wrap(ctx: TestContext) -> None:
    section("Test 9: OpenAI Wrap (Mock Client)")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=100)

    create_called = False

    class MockCompletions:
        async def create(self, **kwargs):
            nonlocal create_called
            create_called = True
            return {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "model": kwargs.get("model"),
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Hello! How can I help?"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 25, "completion_tokens": 10, "total_tokens": 35},
            }

    class MockChat:
        completions = MockCompletions()

    class MockModels:
        async def list(self):
            return {"data": [{"id": "gpt-4o"}]}

    class MockClient:
        chat = MockChat()
        models = MockModels()

    wrapped = lp.wrap(MockClient())

    result = await wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi!"},
        ],
    )

    assert_true("Wrapped create() was called", create_called)
    assert_equal("Response is passed through unchanged", result["choices"][0]["message"]["content"], "Hello! How can I help?")
    assert_equal("Usage data preserved", result["usage"]["total_tokens"], 35)

    models = await wrapped.models.list()
    assert_equal("Non-chat methods pass through", models["data"][0]["id"], "gpt-4o")

    await asyncio.sleep(0.05)
    lp.destroy()


async def test_customer_context_and_tracing(ctx: TestContext) -> None:
    section("Test 10: Customer Context & Tracing")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=100)

    class MockCompletions:
        async def create(self, **kwargs):
            return {
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Sure!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    from launchpromptly.types import WrapOptions
    wrapped = lp.wrap(MockClient(), WrapOptions(
        feature="default-feature",
        trace_id="trace-abc",
        span_name="test-span",
    ))

    await wrapped.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Hello"}],
    )
    await asyncio.sleep(0.05)
    await lp.flush()
    assert_true("Flush with customer context succeeds (no throw)", True)
    lp.destroy()


async def test_with_context(ctx: TestContext) -> None:
    section("Test 11: Context Propagation (contextvars)")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=100)

    # 1. getContext outside returns None
    assert_equal("get_context() outside context is None", lp.get_context(), None)

    # 2. getContext inside returns the context
    with lp.context(trace_id="req-001", customer_id="user-42"):
        ctx_val = lp.get_context()
    assert_equal("get_context() returns trace_id", ctx_val.trace_id, "req-001")
    assert_equal("get_context() returns customer_id", ctx_val.customer_id, "user-42")

    # 3. Context flows across async/await
    async def check_async():
        await asyncio.sleep(0.01)
        return lp.get_context()

    with lp.context(trace_id="async-trace", feature="billing"):
        async_ctx = await check_async()
    assert_equal("Context flows across async/await", async_ctx.trace_id, "async-trace")
    assert_equal("Feature flows across async/await", async_ctx.feature, "billing")

    # 4. Nested contexts
    with lp.context(trace_id="outer"):
        with lp.context(trace_id="inner"):
            inner_ctx = lp.get_context()
        outer_ctx = lp.get_context()
    assert_equal("Inner context overrides outer", inner_ctx.trace_id, "inner")
    assert_equal("Outer context restored after inner", outer_ctx.trace_id, "outer")

    # 5. prompt() inside context
    lp2 = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, prompt_cache_ttl=0.001)
    await asyncio.sleep(0.01)
    with lp2.context(customer_id="als-customer-55"):
        content = await lp2.prompt(ctx.prompt_slug)
        assert_true("prompt() works inside context", len(content) > 0)

    # 6. wrap + context for events
    class MockCompletions:
        async def create(self, **kwargs):
            return {
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    wrapped = lp.wrap(MockClient())
    with lp.context(trace_id="req-777", customer_id="cust-ctx-1", feature="search", span_name="query"):
        await wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "ALS test"}],
        )
    await asyncio.sleep(0.05)
    await lp.flush()
    assert_true("Flush with context events succeeds", True)

    # 7. context with metadata
    with lp.context(trace_id="meta-trace", metadata={"region": "us-east-1", "version": "v2"}):
        await wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "metadata test"}],
        )
    await asyncio.sleep(0.05)
    await lp.flush()
    assert_true("Flush with metadata context succeeds", True)

    lp.destroy()
    lp2.destroy()


async def test_prompt_event_linking(ctx: TestContext) -> None:
    section("Test 12: Prompt→Event Linking")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=100)
    prompt_content = await lp.prompt(ctx.prompt_slug)
    assert_true("Fetched prompt for linking", len(prompt_content) > 0)

    class MockCompletions:
        async def create(self, **kwargs):
            return {
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "Linked!"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 30, "completion_tokens": 8, "total_tokens": 38},
            }

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    wrapped = lp.wrap(MockClient())
    await wrapped.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt_content},
            {"role": "user", "content": "Tell me about my account"},
        ],
    )
    await asyncio.sleep(0.05)
    await lp.flush()
    assert_true("Flush with linked prompt events succeeds", True)
    lp.destroy()


async def test_event_batching(ctx: TestContext) -> None:
    section("Test 13: Event Batching & Flush")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=3, flush_interval=60.0)

    class MockCompletions:
        async def create(self, **kwargs):
            return {
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    wrapped = lp.wrap(MockClient())

    for i in range(5):
        await wrapped.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"msg {i}"}],
        )
    await asyncio.sleep(0.1)
    assert_true("Multiple events generated without error", True)

    await lp.flush()
    assert_true("Manual flush after batch succeeds", True)
    lp.destroy()


async def test_ab_test_resolution(ctx: TestContext) -> None:
    section("Test 14: A/B Test Resolution (customerId)")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, prompt_cache_ttl=0.001)
    await asyncio.sleep(0.01)
    content_a = await lp.prompt(ctx.prompt_slug, customer_id="user-alpha")
    assert_true("Fetch with customer_id succeeds", len(content_a) > 0)

    await asyncio.sleep(0.01)
    content_b = await lp.prompt(ctx.prompt_slug, customer_id="user-beta")
    assert_true("Fetch with different customer_id succeeds", len(content_b) > 0)

    assert_equal("Same prompt returned for both (no active A/B test)", content_a, content_b)
    lp.destroy()


async def test_deployment_status(ctx: TestContext) -> None:
    section("Test 15: Deployment Status & Run Count")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=100)
    prompt_content = await lp.prompt(ctx.prompt_slug)
    assert_true("Fetched prompt for event linking", len(prompt_content) > 0)

    class MockCompletions:
        async def create(self, **kwargs):
            return {
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
            }

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    wrapped = lp.wrap(MockClient())

    for i in range(3):
        await wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt_content},
                {"role": "user", "content": f"deployment status test {i}"},
            ],
        )

    await asyncio.sleep(0.1)
    await lp.flush()
    await asyncio.sleep(0.2)

    # Query deployment usage stats via REST API
    usage_stats = api_call(
        f"/prompt/{ctx.project_id}/{ctx.prompt_id}/deployments/usage",
        token=ctx.jwt,
    )
    assert_true("Usage stats returned for environments", len(usage_stats) > 0)

    env_stats = next((s for s in usage_stats if s["environmentId"] == ctx.environment_id), None)
    assert_true("Stats found for test environment", env_stats is not None)

    if env_stats:
        assert_true("callCount24h > 0 (events tracked)",
                     env_stats["callCount24h"] > 0,
                     f"callCount24h = {env_stats['callCount24h']}")
        assert_true("lastCalledAt is set", env_stats["lastCalledAt"] is not None,
                     env_stats.get("lastCalledAt", "null"))

        if env_stats["lastCalledAt"]:
            from datetime import datetime, timezone
            called_at = datetime.fromisoformat(env_stats["lastCalledAt"].replace("Z", "+00:00"))
            age_s = (datetime.now(timezone.utc) - called_at).total_seconds()
            assert_true("lastCalledAt is recent (< 60s)", age_s < 60, f"age = {age_s:.0f}s")

    # Deployments list
    deployments = api_call(
        f"/prompt/{ctx.project_id}/{ctx.prompt_id}/deployments",
        token=ctx.jwt,
    )
    assert_true("Deployments list returned", len(deployments) > 0)

    env_dep = next((d for d in deployments if d["environmentId"] == ctx.environment_id), None)
    assert_true("Deployment found for test environment", env_dep is not None)

    if env_dep:
        assert_equal("Deployed version ID matches", env_dep["promptVersionId"], ctx.version_id)
        assert_true("deployedAt is set", env_dep["deployedAt"] is not None)

    # Fetch stats
    fetch_stats = api_call(
        f"/prompt/{ctx.project_id}/fetch-stats?days=1",
        token=ctx.jwt,
    )
    assert_true("Fetch stats returned", fetch_stats is not None)
    assert_true("totalFetches > 0 (prompt fetches tracked)",
                fetch_stats["totalFetches"] > 0,
                f"totalFetches = {fetch_stats['totalFetches']}")

    prompt_fetch = next((p for p in fetch_stats["prompts"] if p["id"] == ctx.prompt_id), None)
    assert_true("Fetch count tracked for test prompt",
                prompt_fetch is not None and prompt_fetch["fetchCount"] > 0,
                f"fetchCount = {prompt_fetch['fetchCount']}" if prompt_fetch else "prompt not found")

    lp.destroy()


async def test_shutdown(ctx: TestContext) -> None:
    section("Test 16: Shutdown & Cleanup")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=100, flush_interval=0.1)
    await lp.prompt(ctx.prompt_slug)

    class MockCompletions:
        async def create(self, **kwargs):
            return {
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "bye"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            }

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    wrapped = lp.wrap(MockClient())
    await wrapped.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "shutdown test"}],
    )
    await asyncio.sleep(0.05)

    await lp.shutdown()
    assert_true("shutdown() completes without error", True)
    assert_true("is_destroyed is True after shutdown", lp.is_destroyed)

    lp.destroy()
    assert_true("Double destroy() after shutdown is safe", True)

    lp2 = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_interval=0.1)
    await lp2.prompt(ctx.prompt_slug)
    lp2.destroy()
    assert_true("destroy() completes without error", True)
    assert_true("is_destroyed is True after destroy", lp2.is_destroyed)
    lp2.destroy()
    assert_true("Double destroy() is safe (no throw)", True)


async def test_with_context_event_verification(ctx: TestContext) -> None:
    section("Test 17: Context Event Verification via API")

    lp = LaunchPromptly(api_key=ctx.sdk_api_key, endpoint=API_ENDPOINT, flush_at=100)

    # Get baseline stats
    baseline_stats = api_call(
        f"/prompt/{ctx.project_id}/{ctx.prompt_id}/deployments/usage",
        token=ctx.jwt,
    )
    baseline_count = next(
        (s["callCount24h"] for s in baseline_stats if s["environmentId"] == ctx.environment_id),
        0,
    )

    prompt_content = await lp.prompt(ctx.prompt_slug)

    class MockCompletions:
        async def create(self, **kwargs):
            return {
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ctx ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 15, "completion_tokens": 5, "total_tokens": 20},
            }

    class MockChat:
        completions = MockCompletions()

    class MockClient:
        chat = MockChat()

    wrapped = lp.wrap(MockClient())

    with lp.context(trace_id="verify-trace-1", customer_id="verify-cust", feature="verify-feature"):
        for i in range(2):
            await wrapped.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": prompt_content},
                    {"role": "user", "content": f"withContext verify {i}"},
                ],
            )

    await asyncio.sleep(0.1)
    await lp.flush()
    await asyncio.sleep(0.2)

    after_stats = api_call(
        f"/prompt/{ctx.project_id}/{ctx.prompt_id}/deployments/usage",
        token=ctx.jwt,
    )
    after_count = next(
        (s["callCount24h"] for s in after_stats if s["environmentId"] == ctx.environment_id),
        0,
    )

    assert_true("Events from context increased call count",
                after_count > baseline_count,
                f"before: {baseline_count}, after: {after_count}")
    lp.destroy()


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    keep_data = "--keep" in sys.argv

    print(f"{BOLD}LaunchPromptly Python SDK — Integration Test Suite{RESET}")
    print(f"{DIM}API: {API_ENDPOINT}{RESET}")
    if keep_data:
        print(f"{DIM}Mode: --keep (test data will NOT be deleted){RESET}")
    print()

    # Health check
    try:
        from urllib.request import urlopen as _urlopen
        resp = _urlopen(f"{API_ENDPOINT}/health", timeout=5)
        if resp.status != 200:
            raise Exception(f"HTTP {resp.status}")
        print(f"{GREEN}✓{RESET} API health check passed\n")
    except Exception as err:
        print(f"{RED}✗ API is not reachable at {API_ENDPOINT}{RESET}")
        print(f"  Make sure the API server is running: cd apps/api && npm run dev")
        sys.exit(1)

    # Setup
    print(f"{DIM}Setting up test data...{RESET}")
    try:
        ctx = setup()
        print(f"{GREEN}✓{RESET} Test data created (project: {ctx.project_id[:8]}...)\n")
    except Exception as err:
        print(f"{RED}✗ Setup failed:{RESET} {err}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Run all tests
    try:
        await test_initialization(ctx)
        await test_singleton(ctx)
        await test_prompt_fetch(ctx)
        await test_template_variables(ctx)
        await test_caching(ctx)
        await test_stale_cache_fallback(ctx)
        await test_not_found_handling(ctx)
        await test_template_utilities()
        await test_openai_wrap(ctx)
        await test_customer_context_and_tracing(ctx)
        await test_with_context(ctx)
        await test_prompt_event_linking(ctx)
        await test_event_batching(ctx)
        await test_ab_test_resolution(ctx)
        await test_deployment_status(ctx)
        await test_shutdown(ctx)
        await test_with_context_event_verification(ctx)
    except Exception as err:
        print(f"\n{RED}Unexpected error:{RESET} {err}")
        import traceback
        traceback.print_exc()

    # Teardown
    if keep_data:
        print(f"\n{DIM}Skipping teardown (--keep). Test data persists:{RESET}")
        print(f"  Prompt: {ctx.prompt_slug} ({ctx.prompt_id})")
        print(f"  Environment: {ctx.environment_id}")
        print(f"  API Key: {ctx.sdk_api_key[:16]}...")
    else:
        print(f"\n{DIM}Cleaning up test data...{RESET}")
        teardown(ctx)

    # Summary
    total = passed + failed
    print(f"\n{'─' * 50}")
    if failed == 0:
        print(f"{GREEN}{BOLD}All {total} tests passed!{RESET}")
    else:
        print(f"{RED}{BOLD}{failed}/{total} tests failed:{RESET}")
        for f in failures:
            print(f"  {RED}✗{RESET} {f}")
    print()

    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
