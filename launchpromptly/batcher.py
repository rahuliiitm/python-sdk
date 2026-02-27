from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from ._internal.event_types import IngestEventPayload

_MAX_RETRIES = 3
_logger = logging.getLogger("launchpromptly")


class EventBatcher:
    """Batches LLM events and flushes them to the LaunchPromptly API."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        flush_at: int = 10,
        flush_interval: float = 5.0,
    ) -> None:
        self._api_key = api_key
        self._endpoint = endpoint
        self._flush_at = flush_at
        self._flush_interval = flush_interval
        self._queue: list[IngestEventPayload] = []
        self._flushing = False
        self._timer_task: Optional[asyncio.Task[None]] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def enqueue(self, event: IngestEventPayload) -> None:
        self._queue.append(event)
        if len(self._queue) >= self._flush_at:
            # Schedule flush without awaiting
            self._schedule_flush()
        elif self._timer_task is None:
            self._start_timer()

    def _schedule_flush(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.flush())
        except RuntimeError:
            # No event loop — flush synchronously
            self._flush_sync()

    def _start_timer(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            self._timer_task = loop.create_task(self._timer_flush())
        except RuntimeError:
            pass

    async def _timer_flush(self) -> None:
        await asyncio.sleep(self._flush_interval)
        self._timer_task = None
        await self.flush()

    async def flush(self) -> None:
        if self._flushing or not self._queue:
            return
        self._flushing = True

        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

        batch = self._queue[:]
        self._queue.clear()
        self._flushing = False

        await self._send_with_retry(batch, 0)

    def _flush_sync(self) -> None:
        """Synchronous flush for non-async contexts."""
        if self._flushing or not self._queue:
            return
        self._flushing = True

        batch = self._queue[:]
        self._queue.clear()
        self._flushing = False

        self._send_sync(batch, 0)

    async def _send_with_retry(
        self, events: list[IngestEventPayload], attempt: int
    ) -> None:
        try:
            payload = json.dumps({"events": [e.to_dict() for e in events]}).encode()
            req = Request(
                f"{self._endpoint}/v1/events/batch",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            response = urlopen(req, timeout=10)
            status = response.status
            _logger.debug("Event batch sent: %d events, status=%d", len(events), status)
            if status >= 400 and attempt < _MAX_RETRIES:
                await asyncio.sleep(2**attempt)
                await self._send_with_retry(events, attempt + 1)
        except Exception as exc:
            _logger.debug("Event batch failed (attempt %d): %s", attempt, exc)
            if attempt < _MAX_RETRIES:
                await asyncio.sleep(2**attempt)
                await self._send_with_retry(events, attempt + 1)

    def _send_sync(self, events: list[IngestEventPayload], attempt: int) -> None:
        try:
            payload = json.dumps({"events": [e.to_dict() for e in events]}).encode()
            req = Request(
                f"{self._endpoint}/v1/events/batch",
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self._api_key}",
                },
                method="POST",
            )
            urlopen(req, timeout=10)
        except Exception:
            if attempt < _MAX_RETRIES:
                import time as _time

                _time.sleep(2**attempt)
                self._send_sync(events, attempt + 1)

    def destroy(self) -> None:
        if self._timer_task is not None:
            self._timer_task.cancel()
            self._timer_task = None

    @property
    def pending_count(self) -> int:
        return len(self._queue)
