"""Red Team runner — executes attacks against a wrapped LLM client."""
from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Optional

from .types import (
    AttackPayload,
    AttackResult,
    GuardrailEventCapture,
    RedTeamOptions,
    RedTeamProgress,
    RedTeamReport,
)
from .attacks import get_built_in_attacks, inject_system_prompt
from .analyzer import analyze_attack_result, AnalysisInput
from .reporter import generate_report


# ── Helpers ──────────────────────────────────────────────────────────────────


def _extract_response_text(result: Any) -> Optional[str]:
    """Extract response text from various provider response formats."""
    if result is None:
        return None

    # OpenAI format
    if hasattr(result, "choices"):
        choices = result.choices
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg and hasattr(msg, "content"):
                return msg.content
    elif isinstance(result, dict):
        choices = result.get("choices", [])
        if choices:
            msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            content = msg.get("content")
            if isinstance(content, str):
                return content

    # Anthropic format
    content_blocks = getattr(result, "content", None)
    if isinstance(content_blocks, list):
        texts = []
        for block in content_blocks:
            if getattr(block, "type", None) == "text":
                texts.append(getattr(block, "text", ""))
        if texts:
            return "".join(texts)

    # Gemini format
    if hasattr(result, "text") and callable(result.text):
        try:
            return result.text()
        except Exception:
            pass

    return None


async def _call_wrapped_client(
    client: Any,
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
) -> Any:
    """Detect provider type and call the appropriate chat method."""
    import inspect

    # OpenAI-style: client.chat.completions.create()
    chat = getattr(client, "chat", None)
    if chat is not None:
        completions = getattr(chat, "completions", None)
        if completions is not None:
            create_fn = getattr(completions, "create", None)
            if create_fn is not None:
                params: Dict[str, Any] = {"messages": messages}
                params["model"] = model or "gpt-4o-mini"
                result = create_fn(**params)
                if inspect.isawaitable(result):
                    result = await result
                return result

    # Anthropic-style: client.messages.create()
    msgs_attr = getattr(client, "messages", None)
    if msgs_attr is not None:
        create_fn = getattr(msgs_attr, "create", None)
        if create_fn is not None:
            system = "\n".join(m["content"] for m in messages if m.get("role") == "system")
            non_system = [m for m in messages if m.get("role") != "system"]
            params = {"messages": non_system, "max_tokens": 256}
            if system:
                params["system"] = system
            params["model"] = model or "claude-sonnet-4-20250514"
            result = create_fn(**params)
            if inspect.isawaitable(result):
                result = await result
            return result

    raise ValueError("Unsupported client format. Expected OpenAI, Anthropic, or Gemini client.")


# ── Runner ───────────────────────────────────────────────────────────────────


async def run_red_team(
    wrapped_client: Any,
    lp_instance: Any,
    options: Optional[RedTeamOptions] = None,
) -> RedTeamReport:
    """Run red team attacks against a wrapped LLM client."""
    opts = options or RedTeamOptions()

    # 1. Build attack list
    attacks: List[AttackPayload] = [
        *get_built_in_attacks(opts.categories),
        *(opts.custom_attacks or []),
    ]

    # 2. Inject system prompt
    if opts.system_prompt:
        attacks = inject_system_prompt(attacks, opts.system_prompt)

    # 3. Shuffle and limit
    random.shuffle(attacks)
    attacks = attacks[:opts.max_attacks]

    # 4. Dry run
    if opts.dry_run:
        dry_results = [
            AttackResult(
                attack=a,
                outcome="inconclusive",
                guardrail_events=[],
                latency_ms=0,
                analysis_reason="Dry run — no LLM calls made",
            )
            for a in attacks
        ]
        return generate_report(dry_results, 0, 0)

    # 5. Set up event interception
    original_emit = lp_instance._emit
    captured_events: List[GuardrailEventCapture] = []

    def intercepted_emit(event_type: str, data: dict) -> None:
        captured_events.append(GuardrailEventCapture(
            type=event_type, data=data, timestamp=time.time(),
        ))
        original_emit(event_type, data)

    lp_instance._emit = intercepted_emit

    results: List[AttackResult] = []
    start_time = time.monotonic()
    estimated_cost = 0.0
    sem = asyncio.Semaphore(opts.concurrency)

    async def run_attack(attack: AttackPayload, idx: int) -> None:
        nonlocal estimated_cost
        async with sem:
            # Isolate events for this attack
            my_events: List[GuardrailEventCapture] = []
            nonlocal captured_events
            prev_events = captured_events
            captured_events = my_events

            attack_start = time.monotonic()
            response_text: Optional[str] = None
            error: Optional[Exception] = None

            try:
                result = await _call_wrapped_client(
                    wrapped_client, attack.messages, opts.model,
                )
                response_text = _extract_response_text(result)
            except Exception as e:
                error = e

            latency_ms = (time.monotonic() - attack_start) * 1000
            estimated_cost += 0.0003

            analysis = analyze_attack_result(AnalysisInput(
                attack=attack,
                guardrail_events=my_events,
                response_text=response_text,
                error=error,
            ))

            attack_result = AttackResult(
                attack=attack,
                outcome=analysis.outcome,
                response_preview=response_text[:500] if response_text else None,
                guardrail_events=my_events,
                error=str(error) if error else None,
                latency_ms=latency_ms,
                analysis_reason=analysis.reason,
            )
            results.append(attack_result)
            captured_events = prev_events

            if opts.on_progress:
                opts.on_progress(RedTeamProgress(
                    completed=len(results),
                    total=len(attacks),
                    current_attack=attack.name,
                    current_category=attack.category,
                ))

            if opts.delay_ms > 0 and idx < len(attacks) - 1:
                await asyncio.sleep(opts.delay_ms / 1000)

    # 6. Execute attacks
    tasks = [run_attack(attack, idx) for idx, attack in enumerate(attacks)]
    await asyncio.gather(*tasks)

    # 7. Restore original _emit
    lp_instance._emit = original_emit

    # 8. Generate report
    total_duration = (time.monotonic() - start_time) * 1000
    return generate_report(results, total_duration, estimated_cost)
