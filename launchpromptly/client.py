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
from .errors import (
    PromptInjectionError, CostLimitError, ContentViolationError,
    ModelPolicyError, OutputSchemaError, JailbreakError, TopicViolationError,
)
from ._internal.schema_validator import validate_output_schema
from .types import LaunchPromptlyOptions, RequestContext, WrapOptions
from ._internal.cost import calculate_event_cost
from ._internal.fingerprint import fingerprint_messages
from ._internal.event_types import (
    IngestEventPayload, PIIDetectionsPayload, PIIDetailEntry,
    InjectionRiskPayload,
    CostGuardPayload, ContentViolationsPayload,
    JailbreakRiskPayload, UnicodeThreatsPayload, SecretDetectionsPayload,
    TopicViolationPayload, OutputSafetyPayload, PromptLeakagePayload,
)
from ._internal.pii import detect_pii, merge_detections, PIIDetection, PIIDetectOptions
from ._internal.redaction import redact_pii, de_redact, RedactionOptions
from ._internal.injection import detect_injection, merge_injection_analyses, InjectionAnalysis, InjectionOptions
from ._internal.jailbreak import detect_jailbreak, merge_jailbreak_analyses, JailbreakAnalysis, JailbreakOptions
from ._internal.unicode_sanitizer import scan_unicode, UnicodeScanResult, UnicodeSanitizeOptions
from ._internal.secret_detection import detect_secrets, SecretDetection, SecretDetectionOptions
from ._internal.topic_guard import check_topic_guard, TopicViolation, TopicGuardOptions
from ._internal.output_safety import scan_output_safety, OutputSafetyThreat, OutputSafetyOptions
from ._internal.prompt_leakage import detect_prompt_leakage, PromptLeakageResult, PromptLeakageOptions
from ._internal.cost_guard import CostGuard
from ._internal.content_filter import detect_content_violations, has_blocking_violation, ContentViolation
from ._internal.model_policy import check_model_policy
from ._internal.streaming import SecurityStream, StreamGuardEngine, StreamSecurityReport
from ._internal.event_types import StreamGuardEventPayload

_DEFAULT_ENDPOINT = "https://api.launchpromptly.dev"

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
        on: Optional[dict] = None,
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

        # Validate endpoint URL to prevent SSRF
        from urllib.parse import urlparse
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("https", "http"):
            raise ValueError(f"Endpoint must use HTTPS or HTTP protocol, got: {parsed.scheme!r}")
        self._event_handlers: dict = on or {}
        self._batcher = EventBatcher(resolved_key, endpoint, flush_at, flush_interval)
        self._destroyed = False

    def _emit(self, event_type: str, data: dict) -> None:
        """Emit a guardrail event. Never throws."""
        handler = self._event_handlers.get(event_type)
        if not handler:
            return
        try:
            from .types import GuardrailEvent
            handler(GuardrailEvent(type=event_type, timestamp=time.time(), data=data))
        except Exception:
            pass  # Event handlers must never break the pipeline

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

    # ── Provider wrapping ────────────────────────────────────────────────────

    def wrap(self, client: Any, options: Optional[WrapOptions] = None) -> Any:
        """Wrap an OpenAI client to automatically capture LLM events.

        Returns a proxy that intercepts chat.completions.create().
        """
        opts = options or WrapOptions()
        return _WrappedClient(client, self, opts)

    def wrap_anthropic(self, client: Any, options: Optional[WrapOptions] = None) -> Any:
        """Wrap an Anthropic client with security pipeline interception.

        Intercepts ``client.messages.create()`` to run PII redaction, injection
        detection, cost controls, and content filtering.

        Usage::

            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key="...")
            wrapped = lp.wrap_anthropic(client, WrapOptions(
                security=SecurityOptions(pii=PIISecurityOptions(redaction="placeholder")),
            ))
            result = await wrapped.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": "Hello"}],
            )
        """
        from .providers.anthropic import wrap_anthropic_client
        opts = options or WrapOptions()
        return wrap_anthropic_client(client, self, opts)

    def wrap_gemini(self, client: Any, options: Optional[WrapOptions] = None) -> Any:
        """Wrap a Google Gemini client with security pipeline interception.

        Intercepts ``client.models.generate_content()`` and
        ``client.models.generate_content_stream()`` to run PII redaction,
        injection detection, cost controls, and content filtering.

        Usage::

            from google import genai

            client = genai.Client(api_key="...")
            wrapped = lp.wrap_gemini(client, WrapOptions(
                security=SecurityOptions(pii=PIISecurityOptions(redaction="placeholder")),
            ))
            result = await wrapped.models.generate_content(
                model="gemini-2.0-flash",
                contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
            )
        """
        from .providers.gemini import wrap_gemini_client
        opts = options or WrapOptions()
        return wrap_gemini_client(client, self, opts)

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


def _extract_tool_call_pii(result: Any, pii_opts: Any) -> list[PIIDetection]:
    """Extract PII from tool_call arguments in the LLM response."""
    detections: list[PIIDetection] = []
    choices = getattr(result, "choices", None)
    if choices is None and isinstance(result, dict):
        choices = result.get("choices")
    if not choices:
        return detections

    for choice in choices:
        msg = getattr(choice, "message", None)
        if msg is None and isinstance(choice, dict):
            msg = choice.get("message")
        if not msg:
            continue

        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls is None and isinstance(msg, dict):
            tool_calls = msg.get("tool_calls")
        if not tool_calls:
            continue

        for tc in tool_calls:
            func = getattr(tc, "function", None)
            if func is None and isinstance(tc, dict):
                func = tc.get("function")
            if not func:
                continue
            args_str = getattr(func, "arguments", None)
            if args_str is None and isinstance(func, dict):
                args_str = func.get("arguments")
            if args_str:
                detect_opts = PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None
                dets = detect_pii(args_str, detect_opts)
                detections.extend(dets)

    return detections


class _WrappedCompletions:
    """Proxy for client.chat.completions that intercepts create()."""

    def __init__(self, original: Any, lp: LaunchPromptly, opts: WrapOptions) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts
        # Initialize cost guard if configured
        security = opts.security
        self._cost_guard: Optional[CostGuard] = None
        if security and security.cost_guard:
            self._cost_guard = CostGuard(security.cost_guard)

    async def create(self, **kwargs: Any) -> Any:
        security = self._opts.security
        als_ctx = _lp_context.get()

        # ── PRE-CALL SECURITY PIPELINE ──────────────────────────────
        input_pii_detections: list[PIIDetection] = []
        output_pii_detections: list[PIIDetection] = []
        injection_result: Optional[InjectionAnalysis] = None
        jailbreak_result: Optional[JailbreakAnalysis] = None
        unicode_scan_result: Optional[UnicodeScanResult] = None
        input_secret_detections: list[SecretDetection] = []
        output_secret_detections: list[SecretDetection] = []
        topic_violation_result: Optional[TopicViolation] = None
        output_safety_threats: list[OutputSafetyThreat] = []
        prompt_leakage_result: Optional[PromptLeakageResult] = None
        cost_violation = None
        input_content_violations: list[ContentViolation] = []
        output_content_violations: list[ContentViolation] = []
        redaction_mapping: dict[str, str] = {}
        redaction_applied = False

        effective_kwargs = kwargs

        if security:
            # 0a. Unicode sanitizer (must run first)
            if security.unicode_sanitizer and security.unicode_sanitizer.enabled is not False:
                messages = effective_kwargs.get("messages", [])
                all_input = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
                unicode_opts = UnicodeSanitizeOptions(
                    action=security.unicode_sanitizer.action or "strip",
                    detect_homoglyphs=security.unicode_sanitizer.detect_homoglyphs,
                )
                unicode_scan_result = scan_unicode(all_input, unicode_opts)

                if unicode_scan_result.found:
                    self._lp._emit("unicode.suspicious", {"result": unicode_scan_result})
                    if security.unicode_sanitizer.on_detect:
                        security.unicode_sanitizer.on_detect(unicode_scan_result)

                    if security.unicode_sanitizer.action == "block":
                        raise RuntimeError(f"Unicode threat detected: {len(unicode_scan_result.threats)} suspicious characters found")

                    if security.unicode_sanitizer.action == "strip" and unicode_scan_result.sanitized_text is not None:
                        sanitized_messages = []
                        for msg in messages:
                            if isinstance(msg.get("content"), str):
                                msg_scan = scan_unicode(msg["content"], UnicodeSanitizeOptions(action="strip", detect_homoglyphs=security.unicode_sanitizer.detect_homoglyphs))
                                sanitized_messages.append({**msg, "content": msg_scan.sanitized_text or msg["content"]})
                            else:
                                sanitized_messages.append(msg)
                        effective_kwargs = {**effective_kwargs, "messages": sanitized_messages}

            # 0b. Model policy enforcement
            if security.model_policy:
                violation = check_model_policy(kwargs, security.model_policy)
                if violation:
                    self._lp._emit("model.blocked", {"violation": violation})
                    if security.model_policy.on_violation:
                        security.model_policy.on_violation(violation)
                    raise ModelPolicyError(violation)

            # 1. Cost guard pre-check
            if self._cost_guard:
                customer_id = als_ctx.customer_id if als_ctx else None
                if not customer_id and self._opts.customer:
                    try:
                        ctx_result = self._opts.customer()
                        if hasattr(ctx_result, "id"):
                            customer_id = ctx_result.id
                    except Exception:
                        pass

                cost_violation = self._cost_guard.check_pre_call(
                    model=kwargs.get("model", "unknown"),
                    max_tokens=kwargs.get("max_tokens"),
                    customer_id=customer_id,
                )
                if cost_violation and self._cost_guard.should_block:
                    self._lp._emit("cost.exceeded", {"violation": cost_violation})
                    if security.cost_guard and security.cost_guard.on_budget_exceeded:
                        security.cost_guard.on_budget_exceeded(cost_violation)
                    raise CostLimitError(cost_violation)

            # 2. PII detection + redaction
            pii_opts = security.pii
            if pii_opts and pii_opts.enabled is not False:
                messages = kwargs.get("messages", [])
                all_text = "\n".join(m.get("content", "") for m in messages if m.get("content"))

                # Scan tool definitions for PII in parameters
                tools = kwargs.get("tools")
                if tools:
                    tool_texts: list[str] = []
                    for tool in tools:
                        func = tool.get("function", {}) if isinstance(tool, dict) else getattr(tool, "function", None)
                        if func:
                            params = func.get("parameters", {}) if isinstance(func, dict) else getattr(func, "parameters", {})
                            if params:
                                tool_texts.append(json.dumps(params))
                    if tool_texts:
                        all_text = all_text + "\n" + "\n".join(tool_texts)

                input_pii_detections = detect_pii(all_text, PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None)

                # Merge with ML providers
                if pii_opts.providers:
                    provider_dets = []
                    for p in pii_opts.providers:
                        try:
                            provider_dets.append(p.detect(all_text))
                        except Exception:
                            pass
                    if provider_dets:
                        input_pii_detections = merge_detections(input_pii_detections, *provider_dets)

                if input_pii_detections:
                    self._lp._emit("pii.detected", {"detections": input_pii_detections, "direction": "input"})
                    if pii_opts.on_detect:
                        pii_opts.on_detect(input_pii_detections)

                redaction_strategy = pii_opts.redaction or "placeholder"
                if input_pii_detections and redaction_strategy != "none":
                    redaction_applied = True
                    redacted_messages = []
                    shared_counters: dict[str, int] = {}
                    for msg in messages:
                        result = redact_pii(
                            msg.get("content", ""),
                            RedactionOptions(
                                strategy=redaction_strategy,
                                types=pii_opts.types,
                                providers=pii_opts.providers,
                            ),
                            shared_counters=shared_counters,
                        )
                        redaction_mapping.update(result.mapping)
                        redacted_messages.append({**msg, "content": result.redacted_text})
                    effective_kwargs = {**kwargs, "messages": redacted_messages}
                    self._lp._emit("pii.redacted", {"strategy": redaction_strategy, "count": len(input_pii_detections)})

            # 2b. Secret detection (input)
            if security.secret_detection and security.secret_detection.enabled is not False:
                messages = effective_kwargs.get("messages", [])
                all_input = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
                input_secret_detections = detect_secrets(
                    all_input,
                    SecretDetectionOptions(
                        built_in_patterns=security.secret_detection.built_in_patterns,
                        custom_patterns=security.secret_detection.custom_patterns,
                    ),
                )
                if input_secret_detections:
                    self._lp._emit("secret.detected", {"detections": input_secret_detections, "direction": "input"})
                    if security.secret_detection.on_detect:
                        security.secret_detection.on_detect(input_secret_detections)
                    if security.secret_detection.action == "block":
                        types_found = ", ".join(d.type for d in input_secret_detections)
                        raise RuntimeError(f"Secrets detected in input: {types_found}")

            # 3. Injection detection
            inj_opts = security.injection
            if inj_opts and inj_opts.enabled is not False:
                messages = kwargs.get("messages", [])
                # Scan user messages AND tool/function results (untrusted external input)
                user_text = "\n".join(
                    m.get("content", "") for m in messages if m.get("role") in ("user", "tool", "function")
                )
                if user_text:
                    inj_detect_opts = InjectionOptions(
                        block_threshold=inj_opts.block_threshold or 0.7,
                    )
                    injection_result = detect_injection(user_text, inj_detect_opts)

                    if inj_opts.providers:
                        provider_results = []
                        for p in inj_opts.providers:
                            try:
                                provider_results.append(p.detect(user_text))
                            except Exception:
                                provider_results.append(InjectionAnalysis(0, [], "allow"))
                        injection_result = merge_injection_analyses(
                            [injection_result] + provider_results,
                            inj_detect_opts,
                        )

                    if injection_result.risk_score > 0:
                        self._lp._emit("injection.detected", {"analysis": injection_result})
                    if inj_opts.on_detect:
                        inj_opts.on_detect(injection_result)

                    if inj_opts.block_on_high_risk and injection_result.action == "block":
                        self._lp._emit("injection.blocked", {"analysis": injection_result})
                        raise PromptInjectionError(injection_result)

            # 3b. Jailbreak detection
            if security.jailbreak and security.jailbreak.enabled is not False:
                messages = kwargs.get("messages", [])
                user_text = "\n".join(
                    m.get("content", "") for m in messages if m.get("role") in ("user", "tool", "function")
                )
                if user_text:
                    jailbreak_result = detect_jailbreak(
                        user_text,
                        JailbreakOptions(
                            block_threshold=security.jailbreak.block_threshold,
                            warn_threshold=security.jailbreak.warn_threshold,
                        ),
                    )
                    if security.jailbreak.providers:
                        provider_results = []
                        for p in security.jailbreak.providers:
                            try:
                                provider_results.append(p.detect(user_text))
                            except Exception:
                                pass
                        if provider_results:
                            jailbreak_result = merge_jailbreak_analyses(
                                [jailbreak_result, *provider_results],
                                JailbreakOptions(block_threshold=security.jailbreak.block_threshold),
                            )
                    if jailbreak_result.risk_score > 0:
                        self._lp._emit("jailbreak.detected", {"analysis": jailbreak_result})
                        if security.jailbreak.on_detect:
                            security.jailbreak.on_detect(jailbreak_result)
                    if security.jailbreak.block_on_detection is not False and jailbreak_result.action == "block":
                        self._lp._emit("jailbreak.blocked", {"analysis": jailbreak_result})
                        raise JailbreakError(jailbreak_result)

            # 4. Content filter
            cf_opts = security.content_filter
            if cf_opts and cf_opts.enabled is not False:
                messages = kwargs.get("messages", [])
                all_input = "\n".join(m.get("content", "") for m in messages)
                input_content_violations = detect_content_violations(all_input, "input", cf_opts)

                if input_content_violations:
                    self._lp._emit("content.violated", {"violations": input_content_violations, "direction": "input"})

                if has_blocking_violation(input_content_violations, cf_opts):
                    if cf_opts.on_violation and input_content_violations:
                        cf_opts.on_violation(input_content_violations[0])
                    raise ContentViolationError(input_content_violations)

                if input_content_violations and cf_opts.on_violation:
                    for v in input_content_violations:
                        cf_opts.on_violation(v)

            # 5. Topic guard
            if security.topic_guard and security.topic_guard.enabled is not False:
                messages = effective_kwargs.get("messages", [])
                all_input = "\n".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
                topic_violation_result = check_topic_guard(
                    all_input,
                    TopicGuardOptions(
                        allowed_topics=security.topic_guard.allowed_topics,
                        blocked_topics=security.topic_guard.blocked_topics,
                    ),
                )
                if topic_violation_result:
                    self._lp._emit("topic.violated", {"violation": topic_violation_result})
                    if security.topic_guard.on_violation:
                        security.topic_guard.on_violation(topic_violation_result)
                    if security.topic_guard.action != "warn":
                        raise TopicViolationError(topic_violation_result)

        # ── STREAMING SHORTCUT ─────────────────────────────────────
        is_streaming = effective_kwargs.get("stream", False)
        if is_streaming:
            start = time.monotonic()
            raw_stream = await self._original.create(**effective_kwargs)

            # Use StreamGuardEngine if stream_guard is configured
            if security and security.stream_guard:
                from ._internal.streaming import _extract_openai_chunk_text
                engine = StreamGuardEngine(
                    stream_guard=security.stream_guard,
                    pii_types=security.pii.types if security.pii else None,
                    pii_providers=security.pii.providers if security.pii else None,
                    injection_block_threshold=(
                        security.injection.block_threshold
                        if security.injection else None
                    ),
                    extract_text=_extract_openai_chunk_text,
                )
                return engine.wrap(raw_stream)

            # Legacy: SecurityStream for post-hoc scanning
            pii_types = None
            pii_providers = None
            if security and security.pii and security.pii.enabled is not False:
                pii_types = security.pii.types
                pii_providers = security.pii.providers
            wrapped_stream = SecurityStream(
                raw_stream,
                pii_types=pii_types,
                providers=pii_providers,
            )
            return wrapped_stream

        # ── CALL ORIGINAL API ───────────────────────────────────────
        start = time.monotonic()
        result = await self._original.create(**effective_kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        # ── POST-CALL SECURITY PIPELINE ─────────────────────────────
        response_for_caller = result

        # Extract response text (needed for both security scanning and event capture)
        response_text = None
        choices = getattr(result, "choices", None)
        if choices is None and isinstance(result, dict):
            choices = result.get("choices")
        if choices and len(choices) > 0:
            choice = choices[0]
            msg = getattr(choice, "message", None)
            if msg is None and isinstance(choice, dict):
                msg = choice.get("message")
            if msg:
                response_text = getattr(msg, "content", None)
                if response_text is None and isinstance(msg, dict):
                    response_text = msg.get("content")

        if security:
            # Post-call: scan response for PII
            pii_opts = security.pii
            if pii_opts and pii_opts.scan_response and response_text:
                output_pii_detections = detect_pii(response_text, PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None)

            # Post-call: scan tool_calls in response for PII
            if pii_opts and pii_opts.enabled is not False:
                tool_call_pii = _extract_tool_call_pii(result, pii_opts)
                if tool_call_pii:
                    output_pii_detections = merge_detections(output_pii_detections, tool_call_pii)

            if output_pii_detections:
                self._lp._emit("pii.detected", {"detections": output_pii_detections, "direction": "output"})

            # Post-call: scan response for content violations
            cf_opts = security.content_filter
            if cf_opts and cf_opts.enabled is not False and response_text:
                output_content_violations = detect_content_violations(response_text, "output", cf_opts)
                if output_content_violations:
                    self._lp._emit("content.violated", {"violations": output_content_violations, "direction": "output"})

            # Post-call: output safety scan
            if security.output_safety and security.output_safety.enabled is not False and response_text:
                output_safety_threats = scan_output_safety(
                    response_text,
                    OutputSafetyOptions(categories=security.output_safety.categories),
                )
                if output_safety_threats:
                    self._lp._emit("output.unsafe", {"threats": output_safety_threats})
                    if security.output_safety.on_detect:
                        security.output_safety.on_detect(output_safety_threats)
                    if security.output_safety.action == "block":
                        cats = ", ".join(t.category for t in output_safety_threats)
                        raise RuntimeError(f"Unsafe output detected: {cats}")

            # Post-call: prompt leakage detection
            if security.prompt_leakage and security.prompt_leakage.enabled is not False and response_text:
                prompt_leakage_result = detect_prompt_leakage(
                    response_text,
                    PromptLeakageOptions(
                        system_prompt=security.prompt_leakage.system_prompt,
                        threshold=security.prompt_leakage.threshold,
                    ),
                )
                if prompt_leakage_result.leaked:
                    self._lp._emit("prompt.leaked", {"result": prompt_leakage_result})
                    if security.prompt_leakage.on_detect:
                        security.prompt_leakage.on_detect(prompt_leakage_result)
                    if security.prompt_leakage.block_on_leak:
                        raise RuntimeError(f"System prompt leakage detected (similarity: {prompt_leakage_result.similarity:.2f})")

            # Post-call: secret detection (output)
            if security.secret_detection and security.secret_detection.enabled is not False and security.secret_detection.scan_response is not False and response_text:
                output_secret_detections = detect_secrets(
                    response_text,
                    SecretDetectionOptions(
                        built_in_patterns=security.secret_detection.built_in_patterns,
                        custom_patterns=security.secret_detection.custom_patterns,
                    ),
                )
                if output_secret_detections:
                    self._lp._emit("secret.detected", {"detections": output_secret_detections, "direction": "output"})
                    if security.secret_detection.on_detect:
                        security.secret_detection.on_detect(output_secret_detections)

            # Post-call: output schema validation
            if security.output_schema and response_text:
                validation = validate_output_schema(response_text, security.output_schema)
                if not validation.valid:
                    self._lp._emit("schema.invalid", {"errors": validation.errors, "response_text": response_text})
                    if security.output_schema.block_on_invalid:
                        raise OutputSchemaError(validation.errors, response_text)

            # Post-call: de-redact response
            if redaction_mapping and response_text:
                de_redacted = de_redact(response_text, redaction_mapping)
                if de_redacted != response_text:
                    # Build a modified result with de-redacted content
                    if isinstance(result, dict):
                        new_choices = list(result.get("choices", []))
                        if new_choices:
                            c = dict(new_choices[0])
                            c["message"] = {**c.get("message", {}), "content": de_redacted}
                            new_choices[0] = c
                        response_for_caller = {**result, "choices": new_choices}
                    else:
                        # For object-style results, try to create a modified copy
                        # This is best-effort; object results may not be easily clonable
                        pass

            # Post-call: update cost guard
            if self._cost_guard:
                usage = getattr(result, "usage", None)
                if usage is None and isinstance(result, dict):
                    usage = result.get("usage")
                if usage:
                    if isinstance(usage, dict):
                        in_tok = usage.get("prompt_tokens", 0)
                        out_tok = usage.get("completion_tokens", 0)
                    else:
                        in_tok = getattr(usage, "prompt_tokens", 0)
                        out_tok = getattr(usage, "completion_tokens", 0)
                    actual_cost = calculate_event_cost("openai", kwargs.get("model", "unknown"), in_tok, out_tok)
                    customer_id = als_ctx.customer_id if als_ctx else None
                    self._cost_guard.record_cost(actual_cost, customer_id)

        # ── CAPTURE EVENT ───────────────────────────────────────────
        try:
            self._capture_event(
                kwargs, result, latency_ms, security,
                input_pii_detections, output_pii_detections,
                injection_result, cost_violation,
                input_content_violations, output_content_violations,
                redaction_applied,
                jailbreak_result=jailbreak_result,
                unicode_scan_result=unicode_scan_result,
                input_secret_detections=input_secret_detections,
                output_secret_detections=output_secret_detections,
                topic_violation_result=topic_violation_result,
                output_safety_threats=output_safety_threats,
                prompt_leakage_result=prompt_leakage_result,
                response_text=response_text,
            )
        except Exception:
            pass  # SDK must never throw

        return response_for_caller

    def _capture_event(
        self,
        params: dict,
        result: Any,
        latency_ms: float,
        security: Any = None,
        input_pii: Optional[list] = None,
        output_pii: Optional[list] = None,
        injection: Optional[InjectionAnalysis] = None,
        cost_violation: Any = None,
        input_violations: Optional[list] = None,
        output_violations: Optional[list] = None,
        redaction_applied: bool = False,
        jailbreak_result: Optional[JailbreakAnalysis] = None,
        unicode_scan_result: Optional[UnicodeScanResult] = None,
        input_secret_detections: Optional[list] = None,
        output_secret_detections: Optional[list] = None,
        topic_violation_result: Optional[TopicViolation] = None,
        output_safety_threats: Optional[list] = None,
        prompt_leakage_result: Optional[PromptLeakageResult] = None,
        response_text: Optional[str] = None,
    ) -> None:
        usage = getattr(result, "usage", None)
        if usage is None:
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

        als_ctx = _lp_context.get()

        customer_id = (als_ctx.customer_id if als_ctx else None)
        feature = (als_ctx.feature if als_ctx else None) or self._opts.feature

        if not customer_id and self._opts.customer:
            ctx_result = self._opts.customer()
            if asyncio.iscoroutine(ctx_result):
                pass
            elif hasattr(ctx_result, "id"):
                customer_id = ctx_result.id
                feature = getattr(ctx_result, "feature", None) or feature

        trace_id = (als_ctx.trace_id if als_ctx else None) or self._opts.trace_id
        span_name = (als_ctx.span_name if als_ctx else None) or self._opts.span_name
        metadata = als_ctx.metadata if als_ctx else None

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
            response_text=response_text or None,
            status_code=200,
            trace_id=trace_id,
            span_name=span_name,
            metadata=metadata,
        )

        # Enrich with security metadata
        if security:
            input_pii = input_pii or []
            output_pii = output_pii or []
            input_violations = input_violations or []
            output_violations = output_violations or []

            has_ml_pii = bool(security.pii and security.pii.providers)
            has_ml_inj = bool(security.injection and security.injection.providers)

            if input_pii or output_pii:
                event.pii_detections = PIIDetectionsPayload(
                    input_count=len(input_pii),
                    output_count=len(output_pii),
                    types=list(set(d.type for d in input_pii + output_pii)),
                    redaction_applied=redaction_applied,
                    detector_used="both" if has_ml_pii else "regex",
                    input_details=[
                        PIIDetailEntry(type=d.type, start=d.start, end=d.end, confidence=d.confidence)
                        for d in input_pii
                    ],
                    output_details=[
                        PIIDetailEntry(type=d.type, start=d.start, end=d.end, confidence=d.confidence)
                        for d in output_pii
                    ],
                )

            if injection:
                event.injection_risk = InjectionRiskPayload(
                    score=injection.risk_score,
                    triggered=injection.triggered,
                    action=injection.action,
                    detector_used="both" if has_ml_inj else "rules",
                )

            if input_violations or output_violations:
                event.content_violations = ContentViolationsPayload(
                    input_violations=[
                        {"category": v.category, "matched": v.matched, "severity": v.severity}
                        for v in input_violations
                    ],
                    output_violations=[
                        {"category": v.category, "matched": v.matched, "severity": v.severity}
                        for v in output_violations
                    ],
                )

            if self._cost_guard:
                cg_opts = security.cost_guard
                event.cost_guard = CostGuardPayload(
                    estimated_cost=cost_usd,
                    budget_remaining=max(
                        0.0,
                        (cg_opts.max_cost_per_hour or float("inf")) - self._cost_guard.get_current_hour_spend()
                    ) if cg_opts else 0.0,
                    limit_triggered=cost_violation.type if cost_violation else None,
                )

            if jailbreak_result and jailbreak_result.risk_score > 0:
                event.jailbreak_risk = JailbreakRiskPayload(
                    score=jailbreak_result.risk_score,
                    triggered=jailbreak_result.triggered,
                    action=jailbreak_result.action,
                    decoded_payloads=getattr(jailbreak_result, "decoded_payloads", None),
                )

            if unicode_scan_result and unicode_scan_result.found:
                event.unicode_threats = UnicodeThreatsPayload(
                    found=True,
                    threat_count=len(unicode_scan_result.threats),
                    threat_types=list(set(t.type for t in unicode_scan_result.threats)),
                    action=(security.unicode_sanitizer.action if security.unicode_sanitizer and security.unicode_sanitizer.action else "strip"),
                )

            in_secrets = input_secret_detections or []
            out_secrets = output_secret_detections or []
            if in_secrets or out_secrets:
                event.secret_detections = SecretDetectionsPayload(
                    input_count=len(in_secrets),
                    output_count=len(out_secrets),
                    types=list(set(d.type for d in in_secrets + out_secrets)),
                )

            if topic_violation_result:
                event.topic_violation = TopicViolationPayload(
                    type=topic_violation_result.type,
                    topic=topic_violation_result.topic,
                    matched_keywords=topic_violation_result.matched_keywords,
                    score=topic_violation_result.score,
                )

            safety_threats = output_safety_threats or []
            if safety_threats:
                event.output_safety = OutputSafetyPayload(
                    threat_count=len(safety_threats),
                    categories=list(set(t.category for t in safety_threats)),
                    threats=[
                        {"category": t.category, "matched": t.matched, "severity": t.severity}
                        for t in safety_threats
                    ],
                )

            if prompt_leakage_result and prompt_leakage_result.leaked:
                event.prompt_leakage = PromptLeakagePayload(
                    leaked=True,
                    similarity=prompt_leakage_result.similarity,
                    meta_response_detected=prompt_leakage_result.meta_response_detected,
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
