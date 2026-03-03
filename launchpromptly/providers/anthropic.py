"""Anthropic Claude provider adapter.

Wraps Anthropic client's ``client.messages.create()`` with the full
security pipeline (PII, injection, content filter, cost guard).
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from .._internal.cost import calculate_event_cost
from .._internal.fingerprint import fingerprint_messages
from .._internal.pii import detect_pii, merge_detections, PIIDetection, PIIDetectOptions
from .._internal.redaction import redact_pii, de_redact, RedactionOptions
from .._internal.injection import (
    detect_injection, merge_injection_analyses,
    InjectionAnalysis, InjectionOptions,
)
from .._internal.cost_guard import CostGuard
from .._internal.content_filter import (
    detect_content_violations, has_blocking_violation, ContentViolation,
)
from .._internal.event_types import (
    IngestEventPayload, PIIDetectionsPayload, InjectionRiskPayload,
    CostGuardPayload, ContentViolationsPayload,
)
from .._internal.model_policy import check_model_policy
from ..errors import (
    PromptInjectionError, CostLimitError,
    ContentViolationError, ModelPolicyError,
    OutputSchemaError,
)
from .._internal.schema_validator import validate_output_schema
from .._internal.streaming import SecurityStream, StreamGuardEngine, StreamSecurityReport
from .._internal.event_types import StreamGuardEventPayload


# ── Extraction Helpers ────────────────────────────────────────────────────────


def extract_content_block_text(content: Any) -> str:
    """Extract text from Anthropic message content (string or ContentBlock list)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in content
            if (isinstance(b, dict) and b.get("type") == "text")
            or (hasattr(b, "type") and b.type == "text")
        )
    return ""


def extract_anthropic_message_texts(params: Dict[str, Any]) -> Dict[str, str]:
    """Extract all text from Anthropic create params for security scanning.

    Returns dict with keys: all_text, user_text, system_text.
    """
    parts: List[str] = []
    user_parts: List[str] = []

    # System prompt
    system_text = ""
    system = params.get("system")
    if system:
        system_text = system if isinstance(system, str) else extract_content_block_text(system)
        parts.append(system_text)

    # Messages
    for msg in params.get("messages", []):
        text = extract_content_block_text(msg.get("content", ""))
        parts.append(text)
        # Anthropic tool_result blocks use role='user', so they're already captured here
        if msg.get("role") == "user":
            user_parts.append(text)

    return {
        "all_text": "\n".join(parts),
        "user_text": "\n".join(user_parts),
        "system_text": system_text,
    }


def extract_anthropic_response_text(result: Any) -> Optional[str]:
    """Extract response text from Anthropic response."""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if content is None:
        return None
    return extract_content_block_text(content)


def extract_anthropic_tool_calls(result: Any) -> List[Any]:
    """Extract tool_use blocks from Anthropic response."""
    content = getattr(result, "content", None)
    if content is None and isinstance(result, dict):
        content = result.get("content")
    if not content:
        return []
    return [
        b for b in content
        if (isinstance(b, dict) and b.get("type") == "tool_use")
        or (hasattr(b, "type") and b.type == "tool_use")
    ]


def extract_anthropic_stream_chunk(chunk: Any) -> Optional[str]:
    """Extract text from an Anthropic streaming chunk."""
    # content_block_delta with text_delta
    chunk_type = getattr(chunk, "type", None)
    if chunk_type is None and isinstance(chunk, dict):
        chunk_type = chunk.get("type")

    if chunk_type == "content_block_delta":
        delta = getattr(chunk, "delta", None)
        if delta is None and isinstance(chunk, dict):
            delta = chunk.get("delta")
        if delta:
            delta_type = getattr(delta, "type", None)
            if delta_type is None and isinstance(delta, dict):
                delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = getattr(delta, "text", None)
                if text is None and isinstance(delta, dict):
                    text = delta.get("text")
                return text

    # Fallback: delta.text
    delta = getattr(chunk, "delta", None)
    if delta is None and isinstance(chunk, dict):
        delta = chunk.get("delta")
    if delta:
        text = getattr(delta, "text", None)
        if text is None and isinstance(delta, dict):
            text = delta.get("text")
        if text:
            return text

    return None


def _redact_content(
    content: Any,
    strategy: str,
    types: Any = None,
    providers: Any = None,
    mapping: Optional[Dict[str, str]] = None,
) -> Any:
    """Apply redaction to Anthropic message content."""
    if isinstance(content, str):
        result = redact_pii(content, RedactionOptions(strategy=strategy, types=types, providers=providers))
        if mapping is not None:
            mapping.update(result.mapping)
        return result.redacted_text

    if isinstance(content, list):
        redacted = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                result = redact_pii(
                    block.get("text", ""),
                    RedactionOptions(strategy=strategy, types=types, providers=providers),
                )
                if mapping is not None:
                    mapping.update(result.mapping)
                redacted.append({**block, "text": result.redacted_text})
            else:
                redacted.append(block)
        return redacted

    return content


# ── Wrapped Client ────────────────────────────────────────────────────────────


class _WrappedAnthropicMessages:
    """Proxy for client.messages that intercepts create()."""

    def __init__(self, original: Any, lp: Any, opts: Any) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts
        security = opts.security
        self._cost_guard: Optional[CostGuard] = None
        if security and security.cost_guard:
            self._cost_guard = CostGuard(security.cost_guard)

    async def create(self, **kwargs: Any) -> Any:
        from ..client import _lp_context

        security = self._opts.security
        als_ctx = _lp_context.get()

        # ── PRE-CALL SECURITY ──────────────────────────────────
        input_pii_detections: List[PIIDetection] = []
        output_pii_detections: List[PIIDetection] = []
        injection_result: Optional[InjectionAnalysis] = None
        cost_violation = None
        input_content_violations: List[ContentViolation] = []
        output_content_violations: List[ContentViolation] = []
        redaction_mapping: Dict[str, str] = {}
        redaction_applied = False
        effective_kwargs = kwargs

        if security:
            # 0. Model policy enforcement (first check)
            if security.model_policy:
                violation = check_model_policy(kwargs, security.model_policy)
                if violation:
                    if security.model_policy.on_violation:
                        security.model_policy.on_violation(violation)
                    self._lp._emit("model.blocked", {"violation": violation})
                    raise ModelPolicyError(violation)

            # 1. Cost guard
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
                    if security.cost_guard and security.cost_guard.on_budget_exceeded:
                        security.cost_guard.on_budget_exceeded(cost_violation)
                    self._lp._emit("cost.exceeded", {"violation": cost_violation})
                    raise CostLimitError(cost_violation)

            # 3. PII detection + redaction
            pii_opts = security.pii
            if pii_opts and pii_opts.enabled is not False:
                texts = extract_anthropic_message_texts(kwargs)
                all_text = texts["all_text"]

                detect_opts = PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None
                input_pii_detections = detect_pii(all_text, detect_opts)

                if pii_opts.providers:
                    provider_dets = []
                    for p in pii_opts.providers:
                        try:
                            provider_dets.append(p.detect(all_text))
                        except Exception:
                            pass
                    if provider_dets:
                        input_pii_detections = merge_detections(input_pii_detections, *provider_dets)

                if input_pii_detections and pii_opts.on_detect:
                    pii_opts.on_detect(input_pii_detections)

                if input_pii_detections:
                    self._lp._emit("pii.detected", {"detections": input_pii_detections, "direction": "input"})

                strategy = pii_opts.redaction or "placeholder"
                if input_pii_detections and strategy != "none":
                    redaction_applied = True
                    messages = kwargs.get("messages", [])
                    redacted_messages = []
                    for msg in messages:
                        new_content = _redact_content(
                            msg.get("content", ""), strategy,
                            pii_opts.types, pii_opts.providers, redaction_mapping,
                        )
                        redacted_messages.append({**msg, "content": new_content})

                    effective_kwargs = {**kwargs, "messages": redacted_messages}

                    # Redact system prompt
                    system = kwargs.get("system")
                    if system:
                        redacted_system = _redact_content(
                            system, strategy,
                            pii_opts.types, pii_opts.providers, redaction_mapping,
                        )
                        effective_kwargs["system"] = redacted_system

                    self._lp._emit("pii.redacted", {"strategy": strategy, "count": len(input_pii_detections)})

            # 4. Injection detection
            inj_opts = security.injection
            if inj_opts and inj_opts.enabled is not False:
                texts = extract_anthropic_message_texts(kwargs)
                user_text = texts["user_text"]
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
                            [injection_result] + provider_results, inj_detect_opts,
                        )

                    if inj_opts.on_detect:
                        inj_opts.on_detect(injection_result)
                    if injection_result.risk_score > 0:
                        self._lp._emit("injection.detected", {"analysis": injection_result})
                    if inj_opts.block_on_high_risk and injection_result.action == "block":
                        self._lp._emit("injection.blocked", {"analysis": injection_result})
                        raise PromptInjectionError(injection_result)

            # 5. Content filter
            cf_opts = security.content_filter
            if cf_opts and cf_opts.enabled is not False:
                texts = extract_anthropic_message_texts(kwargs)
                input_content_violations = detect_content_violations(texts["all_text"], "input", cf_opts)
                if input_content_violations:
                    self._lp._emit("content.violated", {"violations": input_content_violations, "direction": "input"})
                if has_blocking_violation(input_content_violations, cf_opts):
                    if cf_opts.on_violation and input_content_violations:
                        cf_opts.on_violation(input_content_violations[0])
                    raise ContentViolationError(input_content_violations)
                if input_content_violations and cf_opts.on_violation:
                    for v in input_content_violations:
                        cf_opts.on_violation(v)

        # ── STREAMING ──────────────────────────────────────────
        if effective_kwargs.get("stream"):
            start = time.monotonic()
            raw_stream = await self._original.create(**effective_kwargs)

            # Use StreamGuardEngine if stream_guard is configured
            if security and security.stream_guard:
                engine = StreamGuardEngine(
                    stream_guard=security.stream_guard,
                    pii_types=security.pii.types if security.pii else None,
                    pii_providers=security.pii.providers if security.pii else None,
                    injection_block_threshold=(
                        security.injection.block_threshold
                        if security.injection else None
                    ),
                    extract_text=extract_anthropic_stream_chunk,
                )
                return engine.wrap(raw_stream)

            # Legacy: wrap with SecurityStream for post-hoc scanning
            pii_types = None
            pii_providers = None
            if security and security.pii and security.pii.enabled is not False:
                pii_types = security.pii.types
                pii_providers = security.pii.providers
            return SecurityStream(
                raw_stream,
                pii_types=pii_types,
                providers=pii_providers,
                extract_text=extract_anthropic_stream_chunk,
            )

        # ── CALL ORIGINAL API ──────────────────────────────────
        start = time.monotonic()
        result = await self._original.create(**effective_kwargs)
        latency_ms = (time.monotonic() - start) * 1000

        # ── POST-CALL SECURITY ─────────────────────────────────
        response_for_caller = result

        if security:
            response_text = extract_anthropic_response_text(result)

            # PII scan response
            pii_opts = security.pii
            if pii_opts and pii_opts.scan_response and response_text:
                detect_opts = PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None
                output_pii_detections = detect_pii(response_text, detect_opts)

            # Scan tool use inputs for PII
            if pii_opts and pii_opts.enabled is not False:
                tool_calls = extract_anthropic_tool_calls(result)
                if tool_calls:
                    tool_texts = []
                    for tc in tool_calls:
                        inp = tc.get("input") if isinstance(tc, dict) else getattr(tc, "input", None)
                        if inp:
                            tool_texts.append(json.dumps(inp) if not isinstance(inp, str) else inp)
                    if tool_texts:
                        detect_opts = PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None
                        tool_pii = detect_pii("\n".join(tool_texts), detect_opts)
                        output_pii_detections = merge_detections(output_pii_detections, tool_pii)

            if output_pii_detections:
                self._lp._emit("pii.detected", {"detections": output_pii_detections, "direction": "output"})

            # Content filter
            cf_opts = security.content_filter
            if cf_opts and cf_opts.enabled is not False and response_text:
                output_content_violations = detect_content_violations(response_text, "output", cf_opts)
                if output_content_violations:
                    self._lp._emit("content.violated", {"violations": output_content_violations, "direction": "output"})

            # Output schema validation
            if security.output_schema and response_text:
                validation = validate_output_schema(response_text, security.output_schema)
                if not validation.valid:
                    self._lp._emit("schema.invalid", {"errors": validation.errors, "response_text": response_text})
                    if security.output_schema.block_on_invalid:
                        raise OutputSchemaError(validation.errors, response_text)

            # De-redact response
            if redaction_mapping and response_text:
                de_redacted = de_redact(response_text, redaction_mapping)
                if de_redacted != response_text:
                    content = getattr(result, "content", None)
                    if content is None and isinstance(result, dict):
                        content = result.get("content")
                    if isinstance(content, list):
                        new_content = []
                        for block in content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                new_content.append({**block, "text": de_redact(block.get("text", ""), redaction_mapping)})
                            else:
                                new_content.append(block)
                        if isinstance(result, dict):
                            response_for_caller = {**result, "content": new_content}

            # Cost guard update
            if self._cost_guard:
                usage = getattr(result, "usage", None)
                if usage is None and isinstance(result, dict):
                    usage = result.get("usage")
                if usage:
                    in_tok = getattr(usage, "input_tokens", 0) if not isinstance(usage, dict) else usage.get("input_tokens", 0)
                    out_tok = getattr(usage, "output_tokens", 0) if not isinstance(usage, dict) else usage.get("output_tokens", 0)
                    actual_cost = calculate_event_cost("anthropic", kwargs.get("model", "unknown"), in_tok, out_tok)
                    customer_id = als_ctx.customer_id if als_ctx else None
                    self._cost_guard.record_cost(actual_cost, customer_id)

        # ── CAPTURE EVENT ──────────────────────────────────────
        try:
            self._capture_event(
                kwargs, result, latency_ms, security,
                input_pii_detections, output_pii_detections,
                injection_result, cost_violation,
                input_content_violations, output_content_violations,
                redaction_applied,
            )
        except Exception:
            pass

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
    ) -> None:
        from ..client import _lp_context

        usage = getattr(result, "usage", None)
        if usage is None and isinstance(result, dict):
            usage = result.get("usage")
        if usage is None:
            return

        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
        else:
            input_tokens = getattr(usage, "input_tokens", 0)
            output_tokens = getattr(usage, "output_tokens", 0)
        total_tokens = input_tokens + output_tokens

        model = params.get("model", "unknown")
        cost_usd = calculate_event_cost("anthropic", model, input_tokens, output_tokens)

        # Normalize messages for fingerprinting
        texts = extract_anthropic_message_texts(params)
        messages = params.get("messages", [])
        normalized = [
            {"role": m.get("role", "user"), "content": extract_content_block_text(m.get("content", ""))}
            for m in messages
        ]
        fingerprint = fingerprint_messages(normalized, texts["system_text"] or None)

        als_ctx = _lp_context.get()
        customer_id = als_ctx.customer_id if als_ctx else None
        feature = (als_ctx.feature if als_ctx else None) or self._opts.feature

        if not customer_id and self._opts.customer:
            try:
                ctx_result = self._opts.customer()
                if hasattr(ctx_result, "id"):
                    customer_id = ctx_result.id
                    feature = getattr(ctx_result, "feature", None) or feature
            except Exception:
                pass

        trace_id = (als_ctx.trace_id if als_ctx else None) or self._opts.trace_id
        span_name = (als_ctx.span_name if als_ctx else None) or self._opts.span_name
        metadata = als_ctx.metadata if als_ctx else None

        event = IngestEventPayload(
            provider="anthropic",
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
            prompt_preview=None if security else fingerprint.prompt_preview,
            status_code=200,
            trace_id=trace_id,
            span_name=span_name,
            metadata=metadata,
        )

        # Security metadata
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

        self._lp._batcher.enqueue(event)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


class _WrappedAnthropicClient:
    """Proxy for Anthropic client that intercepts messages.create()."""

    def __init__(self, original: Any, lp: Any, opts: Any) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts
        self.messages = _WrappedAnthropicMessages(original.messages, lp, opts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def wrap_anthropic_client(client: Any, lp: Any, opts: Any = None) -> Any:
    """Wrap an Anthropic client with the LaunchPromptly security pipeline.

    Usage::

        from anthropic import Anthropic

        client = Anthropic(api_key="...")
        wrapped = lp.wrap_anthropic(client, options=WrapOptions(
            security=SecurityOptions(pii=PIISecurityOptions(redaction="placeholder")),
        ))
        result = await wrapped.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": "Hello"}],
        )
    """
    from ..types import WrapOptions
    return _WrappedAnthropicClient(client, lp, opts or WrapOptions())
