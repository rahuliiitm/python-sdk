"""Unit tests for the LaunchPromptly client."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, AsyncMock, patch

import pytest

from launchpromptly import LaunchPromptly


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Ensure each test starts with a fresh singleton."""
    LaunchPromptly.reset()
    yield
    LaunchPromptly.reset()


class TestInitialization:
    def test_raises_without_api_key(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="API key not found"):
                LaunchPromptly()

    def test_creates_with_explicit_key(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")
        assert isinstance(lp, LaunchPromptly)
        lp.destroy()

    def test_reads_env_var(self):
        with patch.dict("os.environ", {"LAUNCHPROMPTLY_API_KEY": "lp_live_env"}):
            lp = LaunchPromptly(endpoint="http://localhost:3001")
            assert isinstance(lp, LaunchPromptly)
            lp.destroy()

    def test_reads_short_env_var(self):
        with patch.dict("os.environ", {"LP_API_KEY": "lp_live_short"}, clear=True):
            lp = LaunchPromptly(endpoint="http://localhost:3001")
            assert isinstance(lp, LaunchPromptly)
            lp.destroy()


class TestSingleton:
    def test_init_creates_singleton(self):
        lp = LaunchPromptly.init(api_key="lp_live_test", endpoint="http://localhost:3001")
        assert LaunchPromptly.shared() is lp
        lp.destroy()

    def test_shared_throws_before_init(self):
        with pytest.raises(RuntimeError, match="has not been initialized"):
            LaunchPromptly.shared()

    def test_second_init_returns_existing(self):
        first = LaunchPromptly.init(api_key="lp_live_test", endpoint="http://localhost:3001")
        second = LaunchPromptly.init(api_key="lp_live_other", endpoint="http://other:9999")
        assert second is first

    def test_reset_clears_singleton(self):
        LaunchPromptly.init(api_key="lp_live_test", endpoint="http://localhost:3001")
        LaunchPromptly.reset()
        with pytest.raises(RuntimeError):
            LaunchPromptly.shared()


class TestContext:
    def test_get_context_outside_returns_none(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")
        assert lp.get_context() is None
        lp.destroy()

    def test_get_context_inside_returns_context(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")
        with lp.context(trace_id="t1", customer_id="c1"):
            ctx = lp.get_context()
            assert ctx is not None
            assert ctx.trace_id == "t1"
            assert ctx.customer_id == "c1"
        assert lp.get_context() is None
        lp.destroy()

    def test_nested_context(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")
        with lp.context(trace_id="outer"):
            assert lp.get_context().trace_id == "outer"
            with lp.context(trace_id="inner"):
                assert lp.get_context().trace_id == "inner"
            assert lp.get_context().trace_id == "outer"
        lp.destroy()

    def test_context_with_metadata(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")
        with lp.context(metadata={"region": "us-east-1"}):
            ctx = lp.get_context()
            assert ctx.metadata == {"region": "us-east-1"}
        lp.destroy()


class TestShutdown:
    def test_destroy_is_idempotent(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")
        lp.destroy()
        lp.destroy()  # should not raise
        assert lp.is_destroyed is True

    @pytest.mark.asyncio
    async def test_shutdown_flushes_and_destroys(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")
        await lp.shutdown()
        assert lp.is_destroyed is True
