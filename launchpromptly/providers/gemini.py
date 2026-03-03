"""Google Gemini provider adapter.

Wraps Gemini client's ``client.models.generateContent()`` with the full
security pipeline (PII, injection, content filter, cost guard).

Supports the ``@google/genai`` (Node) / ``google-genai`` (Python) SDK format.
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


def _extract_part_text(part: Any) -> Optional[str]:
    """Extract text from a single Gemini part."""
    if isinstance(part, dict):
        if "text" in part and isinstance(part["text"], str):
            return part["text"]
        if "functionCall" in part:
            return json.dumps(part["functionCall"].get("args", {}))
        if "functionResponse" in part:
            return json.dumps(part["functionResponse"].get("response", {}))
    else:
        text = getattr(part, "text", None)
        if text is not None:
            return text
        fc = getattr(part, "function_call", None) or getattr(part, "functionCall", None)
        if fc:
            args = getattr(fc, "args", None) or (fc.get("args") if isinstance(fc, dict) else None)
            return json.dumps(args or {})
    return None


def extract_gemini_content_text(content: Any) -> str:
    """Extract text from a Gemini content (dict with role + parts)."""
    parts = content.get("parts", []) if isinstance(content, dict) else getattr(content, "parts", [])
    texts = [_extract_part_text(p) for p in parts]
    return "\n".join(t for t in texts if t)


def _normalize_contents(contents: Any) -> List[Dict[str, Any]]:
    """Normalize Gemini contents to list of dicts. Handles string shorthand."""
    if isinstance(contents, str):
        return [{"role": "user", "parts": [{"text": contents}]}]
    if isinstance(contents, list):
        result = []
        for c in contents:
            if isinstance(c, dict):
                result.append(c)
            else:
                result.append({
                    "role": getattr(c, "role", "user"),
                    "parts": [{"text": _extract_part_text(p) or ""} for p in getattr(c, "parts", [])],
                })
        return result
    return []


def extract_gemini_message_texts(params: Dict[str, Any]) -> Dict[str, str]:
    """Extract all text from Gemini generateContent params.

    Returns dict with keys: all_text, user_text, system_text.
    """
    parts: List[str] = []
    user_parts: List[str] = []

    # System instruction
    system_text = ""
    config = params.get("config") or {}
    sys_instr = config.get("systemInstruction") or config.get("system_instruction")
    if sys_instr:
        if isinstance(sys_instr, str):
            system_text = sys_instr
        else:
            system_text = extract_gemini_content_text(sys_instr)
        parts.append(system_text)

    # Contents
    contents = _normalize_contents(params.get("contents", []))
    for content in contents:
        text = extract_gemini_content_text(content)
        parts.append(text)
        # Include user messages and function results (untrusted external input)
        role = content.get("role", "user")
        if role in ("user", "function", "tool"):
            user_parts.append(text)

    return {
        "all_text": "\n".join(parts),
        "user_text": "\n".join(user_parts),
        "system_text": system_text,
    }


def extract_gemini_response_text(result: Any) -> Optional[str]:
    """Extract response text from Gemini response."""
    # Direct text accessor
    text = getattr(result, "text", None)
    if text is None and isinstance(result, dict):
        text = result.get("text")
    if isinstance(text, str):
        return text

    # Standard candidates path
    candidates = getattr(result, "candidates", None)
    if candidates is None and isinstance(result, dict):
        candidates = result.get("candidates")
    if candidates and len(candidates) > 0:
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate, dict) else getattr(candidate, "content", None)
        if content:
            cparts = content.get("parts", []) if isinstance(content, dict) else getattr(content, "parts", [])
            texts = []
            for p in cparts:
                t = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                if isinstance(t, str):
                    texts.append(t)
            if texts:
                return "".join(texts)

    return None


def extract_gemini_function_calls(result: Any) -> List[Any]:
    """Extract function call parts from Gemini response."""
    candidates = getattr(result, "candidates", None)
    if candidates is None and isinstance(result, dict):
        candidates = result.get("candidates")
    if not candidates:
        return []
    candidate = candidates[0]
    content = candidate.get("content") if isinstance(candidate, dict) else getattr(candidate, "content", None)
    if not content:
        return []
    cparts = content.get("parts", []) if isinstance(content, dict) else getattr(content, "parts", [])
    return [
        p for p in cparts
        if (isinstance(p, dict) and "functionCall" in p)
        or hasattr(p, "function_call")
        or hasattr(p, "functionCall")
    ]


def extract_gemini_stream_chunk(chunk: Any) -> Optional[str]:
    """Extract text from a Gemini streaming chunk."""
    # Candidates format
    candidates = getattr(chunk, "candidates", None)
    if candidates is None and isinstance(chunk, dict):
        candidates = chunk.get("candidates")
    if candidates and len(candidates) > 0:
        candidate = candidates[0]
        content = candidate.get("content") if isinstance(candidate, dict) else getattr(candidate, "content", None)
        if content:
            cparts = content.get("parts", []) if isinstance(content, dict) else getattr(content, "parts", [])
            if cparts:
                t = cparts[0].get("text") if isinstance(cparts[0], dict) else getattr(cparts[0], "text", None)
                if isinstance(t, str):
                    return t

    # Direct text field
    text = getattr(chunk, "text", None)
    if text is None and isinstance(chunk, dict):
        text = chunk.get("text")
    if isinstance(text, str):
        return text

    return None


def _redact_gemini_content(
    content: Dict[str, Any],
    strategy: str,
    types: Any = None,
    providers: Any = None,
    mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Apply redaction to Gemini content parts."""
    parts = content.get("parts", [])
    redacted_parts = []
    for part in parts:
        if isinstance(part, dict) and "text" in part and isinstance(part["text"], str):
            result = redact_pii(
                part["text"],
                RedactionOptions(strategy=strategy, types=types, providers=providers),
            )
            if mapping is not None:
                mapping.update(result.mapping)
            redacted_parts.append({**part, "text": result.redacted_text})
        else:
            redacted_parts.append(part)
    return {**content, "parts": redacted_parts}


# ── Wrapped Client ────────────────────────────────────────────────────────────


class _WrappedGeminiModels:
    """Proxy for client.models that intercepts generateContent()."""

    def __init__(self, original: Any, lp: Any, opts: Any) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts
        security = opts.security
        self._cost_guard: Optional[CostGuard] = None
        if security and security.cost_guard:
            self._cost_guard = CostGuard(security.cost_guard)

    async def generate_content(self, **kwargs: Any) -> Any:
        return await self._do_generate(kwargs, streaming=False)

    async def generate_content_stream(self, **kwargs: Any) -> Any:
        return await self._do_generate(kwargs, streaming=True)

    # Also support camelCase (JS SDK convention)
    async def generateContent(self, *args: Any, **kwargs: Any) -> Any:
        # Support positional arg (single dict)
        if args and isinstance(args[0], dict):
            return await self._do_generate(args[0], streaming=False)
        return await self._do_generate(kwargs, streaming=False)

    async def generateContentStream(self, *args: Any, **kwargs: Any) -> Any:
        if args and isinstance(args[0], dict):
            return await self._do_generate(args[0], streaming=True)
        return await self._do_generate(kwargs, streaming=True)

    async def _do_generate(self, params: Dict[str, Any], streaming: bool) -> Any:
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
        effective_params = params

        if security:
            # 0. Model policy enforcement (first check)
            if security.model_policy:
                # Normalize Gemini params to the model-policy interface
                config = params.get("config") or {}
                policy_params = {
                    "model": params.get("model", ""),
                    "max_tokens": config.get("maxOutputTokens") or config.get("max_output_tokens"),
                    "temperature": config.get("temperature"),
                    "system": config.get("systemInstruction") or config.get("system_instruction"),
                }
                violation = check_model_policy(policy_params, security.model_policy)
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
                config = params.get("config") or {}
                cost_violation = self._cost_guard.check_pre_call(
                    model=params.get("model", "unknown"),
                    max_tokens=config.get("maxOutputTokens") or config.get("max_output_tokens"),
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
                texts = extract_gemini_message_texts(params)
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
                    contents = _normalize_contents(params.get("contents", []))
                    redacted_contents = [
                        _redact_gemini_content(c, strategy, pii_opts.types, pii_opts.providers, redaction_mapping)
                        for c in contents
                    ]
                    effective_params = {**params, "contents": redacted_contents}

                    # Redact system instruction
                    config = params.get("config") or {}
                    sys_instr = config.get("systemInstruction") or config.get("system_instruction")
                    if sys_instr:
                        if isinstance(sys_instr, str):
                            result = redact_pii(sys_instr, RedactionOptions(strategy=strategy, types=pii_opts.types))
                            redaction_mapping.update(result.mapping)
                            config_key = "systemInstruction" if "systemInstruction" in config else "system_instruction"
                            effective_params["config"] = {**config, config_key: result.redacted_text}
                        else:
                            redacted_sys = _redact_gemini_content(
                                sys_instr if isinstance(sys_instr, dict) else {"parts": [{"text": str(sys_instr)}]},
                                strategy, pii_opts.types, pii_opts.providers, redaction_mapping,
                            )
                            config_key = "systemInstruction" if "systemInstruction" in config else "system_instruction"
                            effective_params["config"] = {**config, config_key: redacted_sys}

                    self._lp._emit("pii.redacted", {"strategy": strategy, "count": len(input_pii_detections)})

            # 4. Injection detection
            inj_opts = security.injection
            if inj_opts and inj_opts.enabled is not False:
                texts = extract_gemini_message_texts(params)
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

                    if injection_result.risk_score > 0:
                        self._lp._emit("injection.detected", {"analysis": injection_result})

                    if inj_opts.on_detect:
                        inj_opts.on_detect(injection_result)
                    if inj_opts.block_on_high_risk and injection_result.action == "block":
                        self._lp._emit("injection.blocked", {"analysis": injection_result})
                        raise PromptInjectionError(injection_result)

            # 5. Content filter
            cf_opts = security.content_filter
            if cf_opts and cf_opts.enabled is not False:
                texts = extract_gemini_message_texts(params)
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
        if streaming:
            method = getattr(self._original, "generate_content_stream", None) or getattr(self._original, "generateContentStream", None)
            if method:
                start = time.monotonic()
                raw_stream = await method(**effective_params)

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
                        extract_text=extract_gemini_stream_chunk,
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
                    extract_text=extract_gemini_stream_chunk,
                )
            # Fallback: no streaming method found
            pass

        # ── CALL ORIGINAL API ──────────────────────────────────
        method = getattr(self._original, "generate_content", None) or getattr(self._original, "generateContent", None)
        start = time.monotonic()
        result = await method(**effective_params)
        latency_ms = (time.monotonic() - start) * 1000

        # ── POST-CALL SECURITY ─────────────────────────────────
        response_for_caller = result

        if security:
            response_text = extract_gemini_response_text(result)

            # PII scan response
            pii_opts = security.pii
            if pii_opts and pii_opts.scan_response and response_text:
                detect_opts = PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None
                output_pii_detections = detect_pii(response_text, detect_opts)

            # Scan function call args for PII
            if pii_opts and pii_opts.enabled is not False:
                fn_calls = extract_gemini_function_calls(result)
                if fn_calls:
                    fn_texts = []
                    for p in fn_calls:
                        fc = p.get("functionCall") if isinstance(p, dict) else getattr(p, "function_call", None)
                        if fc:
                            args = fc.get("args", {}) if isinstance(fc, dict) else getattr(fc, "args", {})
                            fn_texts.append(json.dumps(args))
                    if fn_texts:
                        detect_opts = PIIDetectOptions(types=pii_opts.types) if pii_opts.types else None
                        fn_pii = detect_pii("\n".join(fn_texts), detect_opts)
                        output_pii_detections = merge_detections(output_pii_detections, fn_pii)

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

            # De-redact
            if redaction_mapping and response_text:
                de_redacted = de_redact(response_text, redaction_mapping)
                if de_redacted != response_text and isinstance(result, dict):
                    candidates = result.get("candidates", [])
                    if candidates:
                        c = candidates[0]
                        content = c.get("content", {})
                        cparts = content.get("parts", [])
                        new_parts = []
                        for p in cparts:
                            if isinstance(p, dict) and "text" in p:
                                new_parts.append({**p, "text": de_redact(p["text"], redaction_mapping)})
                            else:
                                new_parts.append(p)
                        new_content = {**content, "parts": new_parts}
                        new_candidates = [{**c, "content": new_content}] + candidates[1:]
                        response_for_caller = {**result, "candidates": new_candidates}

            # Cost guard update
            if self._cost_guard:
                usage = getattr(result, "usage_metadata", None) or getattr(result, "usageMetadata", None)
                if usage is None and isinstance(result, dict):
                    usage = result.get("usageMetadata") or result.get("usage_metadata")
                if usage:
                    if isinstance(usage, dict):
                        in_tok = usage.get("promptTokenCount", 0)
                        out_tok = usage.get("candidatesTokenCount", 0)
                    else:
                        in_tok = getattr(usage, "prompt_token_count", 0) or getattr(usage, "promptTokenCount", 0)
                        out_tok = getattr(usage, "candidates_token_count", 0) or getattr(usage, "candidatesTokenCount", 0)
                    actual_cost = calculate_event_cost("gemini", params.get("model", "unknown"), in_tok, out_tok)
                    customer_id = als_ctx.customer_id if als_ctx else None
                    self._cost_guard.record_cost(actual_cost, customer_id)

        # ── CAPTURE EVENT ──────────────────────────────────────
        try:
            self._capture_event(
                params, result, latency_ms, security,
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

        # Extract usage
        usage = getattr(result, "usage_metadata", None) or getattr(result, "usageMetadata", None)
        if usage is None and isinstance(result, dict):
            usage = result.get("usageMetadata") or result.get("usage_metadata")
        if usage is None:
            return

        if isinstance(usage, dict):
            input_tokens = usage.get("promptTokenCount", 0)
            output_tokens = usage.get("candidatesTokenCount", 0)
            total_tokens = usage.get("totalTokenCount", 0)
        else:
            input_tokens = getattr(usage, "prompt_token_count", 0) or getattr(usage, "promptTokenCount", 0)
            output_tokens = getattr(usage, "candidates_token_count", 0) or getattr(usage, "candidatesTokenCount", 0)
            total_tokens = getattr(usage, "total_token_count", 0) or getattr(usage, "totalTokenCount", 0)

        model = params.get("model", "unknown")
        cost_usd = calculate_event_cost("gemini", model, input_tokens, output_tokens)

        texts = extract_gemini_message_texts(params)
        contents = _normalize_contents(params.get("contents", []))
        normalized = [
            {
                "role": "assistant" if c.get("role") == "model" else c.get("role", "user"),
                "content": extract_gemini_content_text(c),
            }
            for c in contents
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
            provider="gemini",
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


class _WrappedGeminiClient:
    """Proxy for Google Gemini client that intercepts models.generateContent()."""

    def __init__(self, original: Any, lp: Any, opts: Any) -> None:
        self._original = original
        self._lp = lp
        self._opts = opts
        self.models = _WrappedGeminiModels(original.models, lp, opts)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._original, name)


def wrap_gemini_client(client: Any, lp: Any, opts: Any = None) -> Any:
    """Wrap a Google Gemini client with the LaunchPromptly security pipeline.

    Usage::

        from google import genai

        client = genai.Client(api_key="...")
        wrapped = lp.wrap_gemini(client, options=WrapOptions(
            security=SecurityOptions(pii=PIISecurityOptions(redaction="placeholder")),
        ))
        result = await wrapped.models.generate_content(
            model="gemini-2.0-flash",
            contents=[{"role": "user", "parts": [{"text": "Hello"}]}],
        )
    """
    from ..types import WrapOptions
    return _WrappedGeminiClient(client, lp, opts or WrapOptions())
