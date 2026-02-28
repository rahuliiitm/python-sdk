"""Tests for streaming support (SecurityStream)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from launchpromptly._internal.streaming import SecurityStream, StreamSecurityReport
from launchpromptly import LaunchPromptly
from launchpromptly.types import (
    PIISecurityOptions,
    SecurityOptions,
    WrapOptions,
)


# -- Helper: async iterator from a list of chunks -----------------------------

class _AsyncChunkIter:
    """Simulate an OpenAI streaming response as an async iterator."""

    def __init__(self, chunks: list) -> None:
        self._chunks = list(chunks)
        self._index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def _make_chunk(text: str) -> dict:
    """Create a minimal streaming chunk dict."""
    return {
        "choices": [
            {
                "delta": {
                    "content": text,
                },
            }
        ],
    }


# -- SecurityStream unit tests ------------------------------------------------

class TestSecurityStream:
    @pytest.mark.asyncio
    async def test_buffers_content(self):
        chunks = [_make_chunk("Hello "), _make_chunk("world!")]
        stream = SecurityStream(_AsyncChunkIter(chunks))

        collected = []
        async for chunk in stream:
            collected.append(chunk)

        assert len(collected) == 2
        assert stream.response_text == "Hello world!"

    @pytest.mark.asyncio
    async def test_detects_pii_in_buffered_content(self):
        chunks = [
            _make_chunk("Contact me at "),
            _make_chunk("john@acme.com"),
            _make_chunk(" for details"),
        ]
        stream = SecurityStream(_AsyncChunkIter(chunks))

        async for _ in stream:
            pass

        report = stream.get_report()
        assert isinstance(report, StreamSecurityReport)
        assert report.response_text == "Contact me at john@acme.com for details"
        assert len(report.pii_detections) >= 1
        assert report.pii_detections[0].type == "email"
        assert report.pii_detections[0].value == "john@acme.com"

    @pytest.mark.asyncio
    async def test_no_pii_in_clean_stream(self):
        chunks = [_make_chunk("The weather is nice today")]
        stream = SecurityStream(_AsyncChunkIter(chunks))

        async for _ in stream:
            pass

        report = stream.get_report()
        assert len(report.pii_detections) == 0
        assert report.response_text == "The weather is nice today"

    @pytest.mark.asyncio
    async def test_get_report_before_exhaustion(self):
        chunks = [_make_chunk("SSN: 123-45-6789")]
        stream = SecurityStream(_AsyncChunkIter(chunks))

        # Don't consume the stream, just call get_report
        report = stream.get_report()
        # Should finalise with whatever was buffered (nothing yet)
        assert report.response_text == ""
        assert len(report.pii_detections) == 0

    @pytest.mark.asyncio
    async def test_multiple_pii_types_in_stream(self):
        chunks = [
            _make_chunk("Email: john@acme.com, "),
            _make_chunk("SSN: 123-45-6789"),
        ]
        stream = SecurityStream(_AsyncChunkIter(chunks))

        async for _ in stream:
            pass

        report = stream.get_report()
        types = {d.type for d in report.pii_detections}
        assert "email" in types
        assert "ssn" in types

    @pytest.mark.asyncio
    async def test_type_filtering(self):
        chunks = [
            _make_chunk("Email: john@acme.com, SSN: 123-45-6789"),
        ]
        stream = SecurityStream(
            _AsyncChunkIter(chunks),
            pii_types=["email"],
        )

        async for _ in stream:
            pass

        report = stream.get_report()
        assert all(d.type == "email" for d in report.pii_detections)

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        stream = SecurityStream(_AsyncChunkIter([]))

        async for _ in stream:
            pass

        report = stream.get_report()
        assert report.response_text == ""
        assert len(report.pii_detections) == 0

    @pytest.mark.asyncio
    async def test_chunks_with_no_delta_content(self):
        """Chunks without delta content (e.g., role-only deltas) should not crash."""
        chunks = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            _make_chunk("Hello"),
            {"choices": [{"delta": {}}]},
        ]
        stream = SecurityStream(_AsyncChunkIter(chunks))

        collected = []
        async for chunk in stream:
            collected.append(chunk)

        assert len(collected) == 3
        assert stream.response_text == "Hello"

    @pytest.mark.asyncio
    async def test_get_report_is_idempotent(self):
        chunks = [_make_chunk("john@acme.com")]
        stream = SecurityStream(_AsyncChunkIter(chunks))

        async for _ in stream:
            pass

        report1 = stream.get_report()
        report2 = stream.get_report()
        assert report1 is report2


# -- Integration with wrapped client ------------------------------------------

@pytest.fixture(autouse=True)
def _reset_singleton():
    LaunchPromptly.reset()
    yield
    LaunchPromptly.reset()


class TestStreamingIntegration:
    @pytest.mark.asyncio
    async def test_wrap_returns_security_stream_when_streaming(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")

        raw_chunks = [_make_chunk("Hello "), _make_chunk("world")]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_AsyncChunkIter(raw_chunks)
        )

        wrapped = lp.wrap(
            mock_client,
            WrapOptions(
                security=SecurityOptions(
                    pii=PIISecurityOptions(enabled=True, redaction="none"),
                ),
            ),
        )

        result = await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hi"}],
            stream=True,
        )

        assert isinstance(result, SecurityStream)

        collected = []
        async for chunk in result:
            collected.append(chunk)

        assert len(collected) == 2
        report = result.get_report()
        assert report.response_text == "Hello world"
        lp.destroy()

    @pytest.mark.asyncio
    async def test_streaming_pii_detection_via_wrap(self):
        lp = LaunchPromptly(api_key="lp_live_test", endpoint="http://localhost:3001")

        raw_chunks = [
            _make_chunk("Your SSN is "),
            _make_chunk("123-45-6789"),
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(
            return_value=_AsyncChunkIter(raw_chunks)
        )

        wrapped = lp.wrap(
            mock_client,
            WrapOptions(
                security=SecurityOptions(
                    pii=PIISecurityOptions(enabled=True, redaction="none"),
                ),
            ),
        )

        result = await wrapped.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "What is my SSN?"}],
            stream=True,
        )

        async for _ in result:
            pass

        report = result.get_report()
        assert len(report.pii_detections) >= 1
        assert report.pii_detections[0].type == "ssn"
        lp.destroy()
