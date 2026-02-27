from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Optional, TypeVar
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from .batcher import EventBatcher
from .cache import PromptCache
from .errors import PromptNotFoundError
from .template import interpolate
from .types import LaunchPromptlyOptions, PromptOptions, RequestContext, WrapOptions
from ._internal.cost import calculate_event_cost
from ._internal.fingerprint import fingerprint_messages
from ._internal.event_types import IngestEventPayload

_DEFAULT_ENDPOINT = "https://api.launchpromptly.dev"
_DEFAULT_PROMPT_CACHE_TTL = 60.0  # seconds

_T = TypeVar("_T")

# Module-level context var for AsyncLocalStorage-equivalent behaviour
_lp_context: ContextVar[Optional[RequestContext]] = ContextVar("lp_context", default=None)


class LaunchPromptly:
    """LaunchPromptly Python SDK client."""

    # ── Singleton ──────────────────────────────────────────────────────────────
    _instance: Optional[LaunchPromptly] = None

    @classmethod
    def init(cls, **kwargs: Any) -> LaunchPromptly:
        """Initialise the global singleton instance.

        Subsequent calls return the existing instance.
        """
        if cls._instance is not None:
            return cls._instance
        cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def shared(cls) -> LaunchPromptly:
        """Access the global singleton. Raises if init() hasn't been called."""
        if cls._instance is None:
            raise RuntimeError(
                "LaunchPromptly has not been initialized. "
                "Call LaunchPromptly.init(api_key=...) first."
            )
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (primarily for testing)."""
        if cls._instance is not None:
            cls._instance.destroy()
            cls._instance = None

    # ── Constructor ────────────────────────────────────────────────────────────

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = _DEFAULT_ENDPOINT,
        flush_at: int = 10,
        flush_interval: float = 5.0,
        prompt_cache_ttl: float = _DEFAULT_PROMPT_CACHE_TTL,
        max_cache_size: int = 1000,
    ) -> None:
        resolved_key = (
            api_key
            or os.environ.get("LAUNCHPROMPTLY_API_KEY")
            or os.environ.get("LP_API_KEY")
            or ""
        )

        if not resolved_key:
            raise ValueError(
                "LaunchPromptly API key not found. Either:\n"
                '  1. Pass it directly: LaunchPromptly(api_key="lp_live_...")\n'
                "  2. Set LAUNCHPROMPTLY_API_KEY environment variable\n"
                "  3. Set LP_API_KEY environment variable\n"
                "Get your key from Settings → Environments in the LaunchPromptly dashboard."
            )

        self._api_key = resolved_key
        self._endpoint = endpoint
        self._prompt_cache_ttl = prompt_cache_ttl
        self._cache = PromptCache(max_cache_size)
        self._batcher = EventBatcher(resolved_key, endpoint, flush_at, flush_interval)
        self._destroyed = False

        # Maps interpolated content → (managed_prompt_id, prompt_version_id)
        self._resolved_prompts: dict[str, tuple[str, str]] = {}

    # ── Context propagation ────────────────────────────────────────────────────

    @contextmanager
    def context(
        self,
        *,
        trace_id: Optional[str] = None,
        span_name: Optional[str] = None,
        customer_id: Optional[str] = None,
        feature: Optional[str] = None,
        metadata: Optional[dict[str, str]] = None,
    ):
        """Context manager for request-scoped context propagation.

        Works across async/await boundaries via contextvars.

        Usage::

            with lp.context(trace_id="req-123", customer_id="user-42"):
                prompt = await lp.prompt("greeting")
                result = await wrapped.chat.completions.create(...)
        """
        ctx = RequestContext(
            trace_id=trace_id,
            span_name=span_name,
            customer_id=customer_id,
            feature=feature,
            metadata=metadata,
        )
        token = _lp_context.set(ctx)
        try:
            yield ctx
        finally:
            _lp_context.reset(token)

    def get_context(self) -> Optional[RequestContext]:
        """Get the current context (or None if outside a context manager)."""
        return _lp_context.get()

    # ── Prompt fetching ────────────────────────────────────────────────────────

    async def prompt(
        self,
        slug: str,
        *,
        customer_id: Optional[str] = None,
        variables: Optional[dict[str, str]] = None,
    ) -> str:
        """Fetch a managed prompt by slug.

        Returns the interpolated content string.
        """
        als_ctx = _lp_context.get()
        effective_customer_id = customer_id or (als_ctx.customer_id if als_ctx else None)

        # Check cache first
        cached = self._cache.get(slug)
        if cached is not None:
            content = interpolate(cached.content, variables) if variables else cached.content
            self._resolved_prompts[content] = (
                cached.managed_prompt_id,
                cached.prompt_version_id,
            )
            return content

        # Fetch from API
        query = f"?customerId={effective_customer_id}" if effective_customer_id else ""
        url = f"{self._endpoint}/v1/prompts/resolve/{slug}{query}"

        try:
            req = Request(
                url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                method="GET",
            )
            response = urlopen(req, timeout=10)
            data = json.loads(response.read().decode())

            # Cache the raw template
            self._cache.set(
                slug,
                content=data["content"],
                managed_prompt_id=data["managedPromptId"],
                prompt_version_id=data["promptVersionId"],
                version=data["version"],
                ttl=self._prompt_cache_ttl,
            )

            content = interpolate(data["content"], variables) if variables else data["content"]
            self._resolved_prompts[content] = (
                data["managedPromptId"],
                data["promptVersionId"],
            )
            return content

        except HTTPError as e:
            if e.code == 404:
                raise PromptNotFoundError(slug) from e
            # On other HTTP errors, try stale cache
            stale = self._cache.get_stale(slug)
            if stale is not None:
                content = interpolate(stale.content, variables) if variables else stale.content
                return content
            raise

        except PromptNotFoundError:
            raise

        except Exception:
            # On network error, try stale cache
            stale = self._cache.get_stale(slug)
            if stale is not None:
                content = interpolate(stale.content, variables) if variables else stale.content
                return content
            raise

    # ── OpenAI wrapping ────────────────────────────────────────────────────────

    def wrap(self, client: Any, options: Optional[WrapOptions] = None) -> Any:
        """Wrap an OpenAI client to automatically capture LLM events.

        Returns a proxy that intercepts chat.completions.create().
        """
        opts = options or WrapOptions()
        return _WrappedClient(client, self, opts)

    # ── Flush / Destroy / Shutdown ─────────────────────────────────────────────

    async def flush(self) -> None:
        """Flush all pending events to the API."""
        await self._batcher.flush()

    def destroy(self) -> None:
        """Stop timers and release resources. Safe to call multiple times."""
        if self._destroyed:
            return
        self._destroyed = True
        self._batcher.destroy()

    async def shutdown(self) -> None:
        """Graceful shutdown: flush pending events, then destroy."""
        await self.flush()
        self.destroy()

    @property
    def is_destroyed(self) -> bool:
        return self._destroyed


class _WrappedCompletions:
    """Proxy for client.chat.completions that intercepts create()."""

    def __init__(self, original: Any, lp: LaunchPromptly, opts: WrapOptions) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts

    async def create(self, **kwargs: Any) -> Any:
        start = time.monotonic()
        result = await self._original.create(**kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        # Fire-and-forget event capture
        try:
            self._capture_event(kwargs, result, latency_ms)
        except Exception:
            pass  # SDK must never throw

        return result

    def _capture_event(self, params: dict, result: Any, latency_ms: float) -> None:
        usage = getattr(result, "usage", None)
        if usage is None:
            # Try dict access for mock clients
            if isinstance(result, dict):
                usage = result.get("usage")
            if usage is None:
                return

        if isinstance(usage, dict):
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
        else:
            input_tokens = getattr(usage, "prompt_tokens", 0)
            output_tokens = getattr(usage, "completion_tokens", 0)
            total_tokens = getattr(usage, "total_tokens", 0)

        model = params.get("model", "unknown")
        messages = params.get("messages", [])

        cost_usd = calculate_event_cost("openai", model, input_tokens, output_tokens)

        system_msg = next((m for m in messages if m.get("role") == "system"), None)
        non_system = [m for m in messages if m.get("role") != "system"]

        fingerprint = fingerprint_messages(
            non_system,
            system_msg.get("content") if system_msg else None,
        )

        # Resolve context: ALS > WrapOptions
        als_ctx = _lp_context.get()

        customer_id = (als_ctx.customer_id if als_ctx else None)
        feature = (als_ctx.feature if als_ctx else None) or self._opts.feature

        if not customer_id and self._opts.customer:
            ctx_result = self._opts.customer()
            if asyncio.iscoroutine(ctx_result):
                pass  # Can't await in sync context; skip
            elif hasattr(ctx_result, "id"):
                customer_id = ctx_result.id
                feature = getattr(ctx_result, "feature", None) or feature

        trace_id = (als_ctx.trace_id if als_ctx else None) or self._opts.trace_id
        span_name = (als_ctx.span_name if als_ctx else None) or self._opts.span_name
        metadata = als_ctx.metadata if als_ctx else None

        # Check prompt linking
        prompt_meta = self._lp._resolved_prompts.get(
            system_msg.get("content", "") if system_msg else ""
        )

        event = IngestEventPayload(
            provider="openai",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=round(latency_ms),
            customer_id=customer_id,
            feature=feature,
            system_hash=fingerprint.system_hash,
            full_hash=fingerprint.full_hash,
            prompt_preview=fingerprint.prompt_preview,
            status_code=200,
            managed_prompt_id=prompt_meta[0] if prompt_meta else None,
            prompt_version_id=prompt_meta[1] if prompt_meta else None,
            trace_id=trace_id,
            span_name=span_name,
            metadata=metadata,
        )

        self._lp._batcher.enqueue(event)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _WrappedChat:
    """Proxy for client.chat that provides wrapped completions."""

    def __init__(self, original: Any, lp: LaunchPromptly, opts: WrapOptions) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts
        self.completions = _WrappedCompletions(original.completions, lp, opts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _WrappedClient:
    """Proxy for an OpenAI-like client that intercepts chat.completions.create()."""

    def __init__(self, original: Any, lp: LaunchPromptly, opts: WrapOptions) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts
        self.chat = _WrappedChat(original.chat, lp, opts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)
