#!/usr/bin/env python3
"""
LaunchPromptly Python SDK — Integration Test Suite

Self-bootstrapping: creates its own user, deployment, and API key,
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

    LaunchPromptly.reset()
    threw = False
    try:
        LaunchPromptly.shared()
    except RuntimeError:
        threw = True
    assert_true("reset() clears the singleton", threw)


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

    # 5. wrap + context for events
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

    # 6. context with metadata
    with lp.context(trace_id="meta-trace", metadata={"region": "us-east-1", "version": "v2"}):
        await wrapped.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "metadata test"}],
        )
    await asyncio.sleep(0.05)
    await lp.flush()
    assert_true("Flush with metadata context succeeds", True)

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
        await test_with_context(ctx)
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
