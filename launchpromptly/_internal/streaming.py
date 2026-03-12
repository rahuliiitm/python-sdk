"""
Streaming support -- wraps an async iterator to buffer chunks and run
PII detection on the full content after the stream completes.

Also provides StreamGuardEngine for real-time mid-stream security scanning.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from .injection import InjectionAnalysis, InjectionOptions, detect_injection
from .pii import PIIDetection, PIIDetectOptions, detect_pii, merge_detections


@dataclass
class StreamSecurityReport:
    """Report of security detections found in a streamed response."""

    pii_detections: List[PIIDetection] = field(default_factory=list)
    injection_risk: Optional[InjectionAnalysis] = None
    response_text: str = ""
    stream_violations: List[Any] = field(default_factory=list)
    aborted: bool = False
    approximate_tokens: int = 0
    response_length: int = 0
    response_word_count: int = 0


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
        extract_text: Optional[Callable[[Any], Optional[str]]] = None,
    ) -> None:
        self._raw_stream = raw_stream
        self._pii_types = pii_types
        self._providers = providers
        self._extract_fn = extract_text or self._extract_delta_text
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

        delta_text = self._extract_fn(chunk)
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


def _extract_openai_chunk_text(chunk: Any) -> Optional[str]:
    """Extract text content from an OpenAI-style streaming chunk.

    Handles both object and dict formats: choices[0].delta.content
    """
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


# ── StreamGuardEngine ────────────────────────────────────────────────
# Real-time streaming guard with rolling window buffer, periodic scanning,
# response length enforcement, and mid-stream abort.


def _count_words(text: str) -> int:
    """Count words in a text fragment."""
    stripped = text.strip()
    if not stripped:
        return 0
    return len(stripped.split())


class StreamGuardEngine:
    """Core streaming guard engine for real-time security scanning.

    Wraps an async iterable stream, applies periodic PII + injection scanning
    on a rolling window, enforces response length limits, and can abort
    mid-stream when violations are detected.
    """

    def __init__(
        self,
        *,
        stream_guard: Any,  # StreamGuardOptions
        pii_types: Optional[List[str]] = None,
        pii_providers: Optional[list] = None,
        injection_block_threshold: Optional[float] = None,
        extract_text: Callable[[Any], Optional[str]],
        on_complete: Optional[Callable[["StreamSecurityReport"], None]] = None,
    ) -> None:
        self._scan_interval: int = getattr(stream_guard, "scan_interval", 500)
        self._window_overlap: int = getattr(stream_guard, "window_overlap", 200)
        self._on_violation_action: str = getattr(stream_guard, "on_violation", "flag")
        self._on_stream_violation = getattr(stream_guard, "on_stream_violation", None)
        self._do_pii_scan: bool = getattr(stream_guard, "pii_scan", None) is not False
        self._do_injection_scan: bool = getattr(stream_guard, "injection_scan", None) is not False
        self._do_final_scan: bool = getattr(stream_guard, "final_scan", True)
        self._track_tokens: bool = getattr(stream_guard, "track_tokens", True)
        max_rl = getattr(stream_guard, "max_response_length", None)
        self._max_chars: Optional[int] = getattr(max_rl, "max_chars", None) if max_rl else None
        self._max_words: Optional[int] = getattr(max_rl, "max_words", None) if max_rl else None

        self._pii_types = pii_types
        self._pii_providers = pii_providers
        self._injection_block_threshold = injection_block_threshold
        self._extract_text = extract_text
        self._on_complete = on_complete

        # State
        self._buffer = ""
        self._window_start = 0
        self._chars_since_scan = 0
        self._word_count = 0
        self._aborted = False
        self._violations: List[Any] = []  # StreamViolation instances
        self._report: Optional[StreamSecurityReport] = None

    def wrap(self, source: Any) -> _GuardedStream:
        """Wrap an async iterable source into a guarded stream."""
        return _GuardedStream(self, source)

    def get_report(self) -> StreamSecurityReport:
        """Get the security report (available after stream completes or aborts)."""
        if self._report is None:
            self._build_report()
        return self._report  # type: ignore[return-value]

    def get_violations(self) -> List[Any]:
        """Get violations detected so far."""
        return list(self._violations)

    def get_approximate_tokens(self) -> int:
        """Get approximate token count so far."""
        return len(self._buffer) // 4

    def get_response_text(self) -> str:
        """Get the buffered response text so far."""
        return self._buffer

    def is_aborted(self) -> bool:
        """Whether the stream was aborted."""
        return self._aborted

    # ── Internal methods ──

    def _process_text(self, text: str) -> None:
        """Process extracted text: buffer, count, check limits, scan."""
        self._buffer += text
        self._chars_since_scan += len(text)
        self._word_count += _count_words(text)

        self._check_length_limit()

        if not self._aborted and self._chars_since_scan >= self._scan_interval:
            self._periodic_scan()

    def _check_length_limit(self) -> None:
        """Check response length limits."""
        if self._max_chars and len(self._buffer) > self._max_chars:
            self._handle_violation(_make_violation(
                "length", len(self._buffer),
                {"current": len(self._buffer), "limit": self._max_chars, "unit": "chars"},
            ))
        if self._max_words and self._word_count > self._max_words:
            self._handle_violation(_make_violation(
                "length", len(self._buffer),
                {"current": self._word_count, "limit": self._max_words, "unit": "words"},
            ))

    def _periodic_scan(self) -> None:
        """Run PII + injection scan on the current window."""
        self._chars_since_scan = 0
        scan_text = self._buffer[self._window_start:]

        # PII scan
        if self._do_pii_scan:
            detect_opts = PIIDetectOptions(types=self._pii_types) if self._pii_types else None
            detections = detect_pii(scan_text, detect_opts)

            if self._pii_providers:
                provider_dets: List[List[PIIDetection]] = []
                for p in self._pii_providers:
                    try:
                        provider_dets.append(p.detect(scan_text, detect_opts))
                    except Exception:
                        provider_dets.append([])
                if provider_dets:
                    detections = merge_detections(detections, *provider_dets)

            if detections:
                # Adjust offsets relative to full buffer
                adjusted = []
                for d in detections:
                    adj = PIIDetection(
                        type=d.type,
                        value=d.value,
                        start=d.start + self._window_start,
                        end=d.end + self._window_start,
                        confidence=d.confidence,
                    )
                    adjusted.append(adj)
                self._handle_violation(_make_violation(
                    "pii", adjusted[0].start, adjusted,
                ))

        # Injection scan
        if self._do_injection_scan:
            inj_opts = InjectionOptions(
                block_threshold=self._injection_block_threshold or 0.7,
            )
            analysis = detect_injection(scan_text, inj_opts)
            if analysis.action in ("warn", "block"):
                self._handle_violation(_make_violation(
                    "injection", self._window_start, analysis,
                ))

        # Slide window forward
        if len(self._buffer) > self._window_overlap:
            self._window_start = len(self._buffer) - self._window_overlap

    def _final_scan(self) -> None:
        """Run full-text scan after stream ends."""
        full_text = self._buffer
        if not full_text:
            return

        # Full PII scan
        if self._do_pii_scan:
            detect_opts = PIIDetectOptions(types=self._pii_types) if self._pii_types else None
            detections = detect_pii(full_text, detect_opts)
            if self._pii_providers:
                provider_dets_list: List[List[PIIDetection]] = []
                for p in self._pii_providers:
                    try:
                        provider_dets_list.append(p.detect(full_text, detect_opts))
                    except Exception:
                        provider_dets_list.append([])
                if provider_dets_list:
                    detections = merge_detections(detections, *provider_dets_list)
            if detections:
                self._handle_violation(_make_violation(
                    "pii", detections[0].start, detections,
                ))

        # Full injection scan
        if self._do_injection_scan:
            inj_opts = InjectionOptions(
                block_threshold=self._injection_block_threshold or 0.7,
            )
            analysis = detect_injection(full_text, inj_opts)
            if analysis.action in ("warn", "block"):
                self._handle_violation(_make_violation(
                    "injection", 0, analysis,
                ))

    def _handle_violation(self, violation: Any) -> None:
        """Handle a detected violation."""
        self._violations.append(violation)
        if self._on_stream_violation:
            self._on_stream_violation(violation)
        if self._on_violation_action == "abort":
            self._aborted = True

    def _finalize(self) -> None:
        """Run final scan and build report."""
        if self._do_final_scan and not self._aborted:
            self._final_scan()
        self._build_report()
        if self._on_complete and self._report is not None:
            try:
                self._on_complete(self._report)
            except Exception:
                pass  # SDK must never throw

    def _build_report(self) -> None:
        """Build the security report from accumulated state."""
        all_pii: List[PIIDetection] = []
        injection_risk: Optional[InjectionAnalysis] = None

        for v in self._violations:
            if v.type == "pii" and isinstance(v.details, list):
                all_pii.extend(v.details)
            if v.type == "injection":
                analysis = v.details
                if injection_risk is None or analysis.risk_score > injection_risk.risk_score:
                    injection_risk = analysis

        self._report = StreamSecurityReport(
            pii_detections=all_pii,
            injection_risk=injection_risk,
            response_text=self._buffer,
            stream_violations=list(self._violations),
            aborted=self._aborted,
            approximate_tokens=len(self._buffer) // 4 if self._track_tokens else 0,
            response_length=len(self._buffer),
            response_word_count=self._word_count,
        )


class _GuardedStream:
    """Async iterator wrapper that delegates to StreamGuardEngine."""

    def __init__(self, engine: StreamGuardEngine, source: Any) -> None:
        self._engine = engine
        self._source = source

    def __aiter__(self) -> _GuardedStream:
        return self

    async def __anext__(self) -> Any:
        if self._engine._aborted:
            self._engine._finalize()
            raise StopAsyncIteration

        try:
            chunk = await self._source.__anext__()
        except StopAsyncIteration:
            self._engine._finalize()
            raise

        text = self._engine._extract_text(chunk)
        if text:
            self._engine._process_text(text)

        if self._engine._aborted:
            self._engine._finalize()
            raise StopAsyncIteration

        return chunk

    def get_report(self) -> StreamSecurityReport:
        """Get the security report."""
        return self._engine.get_report()


def _make_violation(vtype: str, offset: int, details: Any) -> Any:
    """Create a StreamViolation instance."""
    from ..types import StreamViolation

    return StreamViolation(
        type=vtype,  # type: ignore[arg-type]
        offset=offset,
        details=details,
        timestamp=time.time(),
    )
