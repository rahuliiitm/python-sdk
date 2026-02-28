"""
Streaming support -- wraps an async iterator to buffer chunks and run
PII detection on the full content after the stream completes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator, List, Optional

from .pii import PIIDetection, PIIDetectOptions, detect_pii, merge_detections


@dataclass
class StreamSecurityReport:
    """Report of PII detections found in a streamed response."""

    pii_detections: List[PIIDetection]
    response_text: str


class SecurityStream:
    """Wraps an async iterator (OpenAI streaming response) to buffer content
    and run PII detection on the full buffered text when the stream ends.

    Implements the async iterator protocol so callers can ``async for`` over it
    exactly as they would over the raw stream.

    Usage::

        stream = SecurityStream(raw_stream, pii_types=["email", "ssn"])
        async for chunk in stream:
            handle(chunk)
        report = stream.get_report()
    """

    def __init__(
        self,
        raw_stream: Any,
        *,
        pii_types: Optional[List[str]] = None,
        providers: Optional[list] = None,
    ) -> None:
        self._raw_stream = raw_stream
        self._pii_types = pii_types
        self._providers = providers
        self._chunks: List[Any] = []
        self._content_parts: List[str] = []
        self._report: Optional[StreamSecurityReport] = None
        self._exhausted = False

    # -- Async iterator protocol -----------------------------------------------

    def __aiter__(self) -> SecurityStream:
        return self

    async def __anext__(self) -> Any:
        try:
            chunk = await self._raw_stream.__anext__()
        except StopAsyncIteration:
            self._finalise()
            raise

        self._chunks.append(chunk)

        # Extract text content from the chunk (OpenAI ChatCompletionChunk shape)
        delta_text = self._extract_delta_text(chunk)
        if delta_text:
            self._content_parts.append(delta_text)

        return chunk

    # -- Public API ------------------------------------------------------------

    def get_report(self) -> StreamSecurityReport:
        """Return the security report.

        Must be called after the stream is fully consumed.
        If the stream hasn't been fully consumed yet, it forces finalisation
        with whatever content has been buffered so far.
        """
        if self._report is None:
            self._finalise()
        return self._report  # type: ignore[return-value]

    @property
    def response_text(self) -> str:
        """Return the full buffered response text."""
        return "".join(self._content_parts)

    # -- Internals -------------------------------------------------------------

    def _extract_delta_text(self, chunk: Any) -> Optional[str]:
        """Pull delta content from a streaming chunk (dict or object)."""
        choices = getattr(chunk, "choices", None)
        if choices is None and isinstance(chunk, dict):
            choices = chunk.get("choices")
        if not choices:
            return None

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None and isinstance(choice, dict):
            delta = choice.get("delta")
        if not delta:
            return None

        content = getattr(delta, "content", None)
        if content is None and isinstance(delta, dict):
            content = delta.get("content")
        return content

    def _finalise(self) -> None:
        """Run PII detection on the buffered content and build the report."""
        if self._report is not None:
            return

        full_text = "".join(self._content_parts)
        detect_opts = PIIDetectOptions(types=self._pii_types) if self._pii_types else None
        detections = detect_pii(full_text, detect_opts)

        # Merge with provider detections if available
        if self._providers:
            provider_dets: List[List[PIIDetection]] = []
            for provider in self._providers:
                try:
                    provider_dets.append(provider.detect(full_text, detect_opts))
                except Exception:
                    provider_dets.append([])
            if provider_dets:
                detections = merge_detections(detections, *provider_dets)

        self._report = StreamSecurityReport(
            pii_detections=detections,
            response_text=full_text,
        )
        self._exhausted = True
