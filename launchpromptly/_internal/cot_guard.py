"""Chain-of-thought auditing module -- scans reasoning/thinking blocks
for injection, system prompt leakage, and goal drift.
Zero dependencies. Stateless, pure functions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional

from .injection import detect_injection
from .prompt_leakage import PromptLeakageOptions, detect_prompt_leakage

# ── Types ────────────────────────────────────────────────────────────────────


@dataclass
class ChainOfThoughtGuardOptions:
    enabled: Optional[bool] = None
    scan_reasoning_blocks: Optional[bool] = None
    """Scan reasoning blocks for injection patterns. Default: True."""
    injection_detection: Optional[bool] = None
    """Detect injection in chain-of-thought text. Default: True."""
    system_prompt_leak_detection: Optional[bool] = None
    """Detect system prompt leakage in reasoning."""
    system_prompt: Optional[str] = None
    """System prompt text for leak detection."""
    goal_drift_detection: Optional[bool] = None
    """Detect reasoning diverging from original task."""
    task_description: Optional[str] = None
    """Original task description. Falls back to first user message."""
    goal_drift_threshold: Optional[float] = None
    """Similarity threshold for goal drift. Default: 0.3"""
    action: Optional[Literal["block", "warn", "flag"]] = None
    """Action on violation. Default: 'warn'."""
    on_violation: Optional[Callable[[ChainOfThoughtViolation], None]] = None
    """Callback on violation."""
    embedding_provider: Optional[Any] = None
    """Optional ML embedding provider for semantic goal drift detection.
    Uses cosine similarity instead of Jaccard word overlap."""
    nli_session: Optional[Any] = None
    """Optional NLI session for coherence scoring between reasoning steps.
    Detects injected/contradictory reasoning steps."""


@dataclass
class ChainOfThoughtViolation:
    type: Literal["cot_injection", "cot_system_leak", "cot_goal_drift"]
    reasoning_snippet: str
    risk_score: float
    details: str


@dataclass
class ChainOfThoughtScanResult:
    violations: List[ChainOfThoughtViolation] = field(default_factory=list)
    blocked: bool = False
    reasoning_text: str = ""


# ── Reasoning extraction ─────────────────────────────────────────────────────

_REASONING_BLOCK_PATTERNS = [
    re.compile(r"<thinking>([\s\S]*?)</thinking>", re.IGNORECASE),
    re.compile(r"<scratchpad>([\s\S]*?)</scratchpad>", re.IGNORECASE),
    re.compile(r"<reasoning>([\s\S]*?)</reasoning>", re.IGNORECASE),
    re.compile(r"<internal_monologue>([\s\S]*?)</internal_monologue>", re.IGNORECASE),
]


def extract_reasoning_text(response: Any) -> str:
    """Extract reasoning text from an LLM response object."""
    if response is None:
        return ""

    parts: List[str] = []

    # OpenAI o-series: reasoning_content on the message
    try:
        message = response.get("choices", [{}])[0].get("message", {})
        if message.get("reasoning_content"):
            parts.append(message["reasoning_content"])
    except (AttributeError, IndexError, TypeError):
        message = {}

    # Anthropic: thinking content blocks
    try:
        content = response.get("content", None)
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "thinking" and block.get("thinking"):
                    parts.append(block["thinking"])
    except (AttributeError, TypeError):
        pass

    # Tag-based extraction from response text
    text = ""
    if isinstance(message, dict):
        text = message.get("content", "") or ""
    if not text and isinstance(response, str):
        text = response

    if isinstance(text, str):
        for rx in _REASONING_BLOCK_PATTERNS:
            for match in rx.finditer(text):
                parts.append(match.group(1).strip())

    return "\n".join(parts).strip()


# ── Helpers ──────────────────────────────────────────────────────────────────

_TOKEN_SPLIT = re.compile(r"[\s,.!?;:()\[\]{}'\"]+")


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = {t for t in _TOKEN_SPLIT.split(a.lower()) if len(t) > 2}
    tokens_b = {t for t in _TOKEN_SPLIT.split(b.lower()) if len(t) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    return intersection / (len(tokens_a) + len(tokens_b) - intersection)


def _truncate(s: str, max_len: int = 200) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


# ── Public API ───────────────────────────────────────────────────────────────


def scan_chain_of_thought(
    reasoning_text: str,
    options: ChainOfThoughtGuardOptions,
) -> ChainOfThoughtScanResult:
    """Scan extracted reasoning text for violations."""
    violations: List[ChainOfThoughtViolation] = []

    if not reasoning_text:
        return ChainOfThoughtScanResult(violations=[], blocked=False, reasoning_text="")

    # Injection detection
    if options.injection_detection is not False:
        analysis = detect_injection(reasoning_text)
        if analysis.risk_score >= 0.5:
            violations.append(
                ChainOfThoughtViolation(
                    type="cot_injection",
                    reasoning_snippet=_truncate(reasoning_text),
                    risk_score=analysis.risk_score,
                    details=f"Injection detected in reasoning: {', '.join(analysis.triggered)}",
                )
            )

    # System prompt leak detection
    if options.system_prompt_leak_detection and options.system_prompt:
        leak_result = detect_prompt_leakage(
            reasoning_text,
            PromptLeakageOptions(system_prompt=options.system_prompt, threshold=0.4),
        )
        if leak_result.leaked:
            violations.append(
                ChainOfThoughtViolation(
                    type="cot_system_leak",
                    reasoning_snippet=_truncate(reasoning_text),
                    risk_score=leak_result.similarity,
                    details=f"System prompt leak in reasoning (score: {leak_result.similarity:.2f})",
                )
            )

    # Goal drift detection -- semantic (embedding) or lexical (Jaccard)
    if options.goal_drift_detection and options.task_description:
        tokens = [t for t in _TOKEN_SPLIT.split(reasoning_text.lower()) if len(t) > 2]
        if len(tokens) >= 10:
            if options.embedding_provider is not None:
                # Semantic goal drift via embeddings
                task_emb = options.embedding_provider.embed(options.task_description)
                reasoning_emb = options.embedding_provider.embed(reasoning_text)
                similarity = float(options.embedding_provider.cosine(task_emb, reasoning_emb))
            else:
                similarity = _jaccard_similarity(reasoning_text, options.task_description)

            threshold = options.goal_drift_threshold if options.goal_drift_threshold is not None else 0.3
            if similarity < threshold:
                violations.append(
                    ChainOfThoughtViolation(
                        type="cot_goal_drift",
                        reasoning_snippet=_truncate(reasoning_text),
                        risk_score=1 - similarity,
                        details=f"Reasoning diverged from task (similarity: {similarity:.2f}, threshold: {threshold})",
                    )
                )

    # Coherence scoring via NLI -- detect injected reasoning steps
    if options.nli_session is not None and len(reasoning_text) > 100:
        steps = _split_reasoning_steps(reasoning_text)
        if len(steps) >= 2:
            for i in range(min(len(steps) - 1, 5)):
                results = options.nli_session.classify_pair(steps[i], steps[i + 1], top_k=None)
                contradiction_score = _get_contradiction_score(results)
                if contradiction_score > 0.7:
                    violations.append(
                        ChainOfThoughtViolation(
                            type="cot_injection",
                            reasoning_snippet=_truncate(steps[i + 1]),
                            risk_score=contradiction_score,
                            details=f"Incoherent reasoning step {i + 2}: contradicts previous step (score: {contradiction_score:.2f})",
                        )
                    )
                    break  # One coherence violation is enough

    action = options.action or "warn"
    return ChainOfThoughtScanResult(
        violations=violations,
        blocked=action == "block" and len(violations) > 0,
        reasoning_text=reasoning_text,
    )


def _split_reasoning_steps(text: str) -> list[str]:
    """Split reasoning text into logical steps (by paragraph or numbered list)."""
    # Try numbered steps
    numbered = [s.strip() for s in re.split(r"(?:^|\n)\s*(?:\d+[.)]\s|Step\s+\d+[.:]\s)", text, flags=re.IGNORECASE) if len(s.strip()) > 20]
    if len(numbered) >= 2:
        return numbered

    # Fall back to paragraphs
    paragraphs = [s.strip() for s in text.split("\n\n") if len(s.strip()) > 20]
    if len(paragraphs) >= 2:
        return paragraphs

    # Fall back to sentences (max 5)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
    return sentences[:5]


def _get_contradiction_score(results: list[dict]) -> float:
    """Extract contradiction score from NLI output labels."""
    for r in results:
        label = r.get("label", "").upper()
        if label in ("CONTRADICTION", "LABEL_0"):
            return float(r["score"])
    for r in results:
        label = r.get("label", "").upper()
        if label in ("ENTAILMENT", "LABEL_2"):
            return 1.0 - float(r["score"])
    return 0.0
