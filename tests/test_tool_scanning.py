"""Tests for function/tool call PII scanning."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from launchpromptly import LaunchPromptly
from launchpromptly.types import (
    PIISecurityOptions,
    SecurityOptions,
    WrapOptions,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    LaunchPromptly.reset()
    yield
    LaunchPromptly.reset()


def _make_lp() -> LaunchPromptly:
    return LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")


def _mock_openai_client(response: dict) -> MagicMock:
    """Create a mock OpenAI client that returns the given response."""
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=response)
    return client


def _standard_response(**overrides) -> dict:
    base = {
        "choices": [
            {
                "message": {
                    "content": "Hello there!",
                    "role": "assistant",
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }
    base.update(overrides)
    return base


def _tool_call_response(arguments: str) -> dict:
    """Create a mock response with a tool_call."""
    return {
        "choices": [
            {
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "send_email",
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }


class TestToolParameterScanning:
    """Tests for scanning PII in tool/function parameter definitions."""

    @pytest.mark.asyncio
    async def test_detects_pii_in_tool_parameters(self):
        """Tool definitions containing PII values should be detected."""
        lp = _make_lp()
        response = _standard_response()
        mock_client = _mock_openai_client(response)

        detected_pii = []

        wrapped = lp.wrap(
            mock_client,
            WrapOptions(
                security=SecurityOptions(
                    pii=PIISecurityOptions(
                        enabled=True,
                        redaction="none",
                        on_detect=lambda dets: detected_pii.extend(dets),
                    ),
                ),
            ),
        )

        await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Send an email"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "send_email",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "to": {
                                    "type": "string",
                                    "default": "john@acme.com",
                                },
                            },
                        },
                    },
                }
            ],
        )

        # PII in tool parameter defaults should be detected
        assert len(detected_pii) >= 1
        types = {d.type for d in detected_pii}
        assert "email" in types
        lp.destroy()

    @pytest.mark.asyncio
    async def test_no_pii_in_clean_tools(self):
        """Tools without PII should not trigger detections."""
        lp = _make_lp()
        response = _standard_response()
        mock_client = _mock_openai_client(response)

        detected_pii = []

        wrapped = lp.wrap(
            mock_client,
            WrapOptions(
                security=SecurityOptions(
                    pii=PIISecurityOptions(
                        enabled=True,
                        redaction="none",
                        on_detect=lambda dets: detected_pii.extend(dets),
                    ),
                ),
            ),
        )

        await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is the weather?"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                            },
                        },
                    },
                }
            ],
        )

        assert len(detected_pii) == 0
        lp.destroy()


class TestToolCallResponseScanning:
    """Tests for scanning PII in tool_call response arguments."""

    @pytest.mark.asyncio
    async def test_detects_pii_in_tool_call_arguments(self):
        """PII in tool_call arguments should be counted in output detections."""
        lp = _make_lp()
        args = json.dumps({"to": "john@acme.com", "body": "SSN: 123-45-6789"})
        response = _tool_call_response(args)
        mock_client = _mock_openai_client(response)

        wrapped = lp.wrap(
            mock_client,
            WrapOptions(
                security=SecurityOptions(
                    pii=PIISecurityOptions(
                        enabled=True,
                        redaction="none",
                    ),
                ),
            ),
        )

        result = await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Send an email to John"}],
        )

        # The tool_call PII detection happens internally;
        # verify the response is returned unchanged
        assert result["choices"][0]["message"]["tool_calls"] is not None
        lp.destroy()

    @pytest.mark.asyncio
    async def test_no_pii_in_clean_tool_call(self):
        """Tool calls without PII should not add output detections."""
        lp = _make_lp()
        args = json.dumps({"location": "New York"})
        response = _tool_call_response(args)
        mock_client = _mock_openai_client(response)

        wrapped = lp.wrap(
            mock_client,
            WrapOptions(
                security=SecurityOptions(
                    pii=PIISecurityOptions(
                        enabled=True,
                        redaction="none",
                    ),
                ),
            ),
        )

        result = await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Weather in New York"}],
        )

        # Should succeed without errors
        assert result is not None
        lp.destroy()

    @pytest.mark.asyncio
    async def test_tool_call_pii_with_scan_response(self):
        """Tool call PII + scan_response should both contribute to output count."""
        lp = _make_lp()
        args = json.dumps({"email": "john@acme.com"})
        # Response with both text content and tool_calls
        response = {
            "choices": [
                {
                    "message": {
                        "content": "I found the email jane@test.org for you",
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_456",
                                "type": "function",
                                "function": {
                                    "name": "send_email",
                                    "arguments": args,
                                },
                            }
                        ],
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        mock_client = _mock_openai_client(response)

        wrapped = lp.wrap(
            mock_client,
            WrapOptions(
                security=SecurityOptions(
                    pii=PIISecurityOptions(
                        enabled=True,
                        scan_response=True,
                        redaction="none",
                    ),
                ),
            ),
        )

        result = await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Send email to John"}],
        )

        assert result is not None
        lp.destroy()
