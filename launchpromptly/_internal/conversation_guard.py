"""Multi-step conversation guard -- stateful class that tracks
conversation state across LLM calls for agentic workflows.
Zero dependencies.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Literal, Optional

from .pii import PIIDetection

# ── Types ────────────────────────────────────────────────────────────────────


@dataclass
class ConversationGuardOptions:
    max_turns: Optional[int] = None
    """Maximum turns before blocking."""
    topic_drift_detection: Optional[bool] = None
    """Detect topic drift across turns."""
    topic_drift_threshold: Optional[float] = None
    """Threshold for topic drift. Default: 0.3"""
    cross_turn_pii_tracking: Optional[bool] = None
    """Track PII spread across turns."""
    accumulating_risk: Optional[bool] = None
    """Accumulate risk scores across turns."""
    risk_threshold: Optional[float] = None
    """Cumulative risk threshold. Default: 2.0"""
    max_consecutive_similar_responses: Optional[int] = None
    """Max consecutive similar responses for loop detection. Default: 3"""
    max_total_tool_calls: Optional[int] = None
    """Max total tool calls across conversation."""
    action: Optional[Literal["block", "warn", "flag"]] = None
    """Action on violation. Default: 'block'."""
    on_violation: Optional[Callable[[ConversationGuardViolation], None]] = None
    """Callback on violation."""


@dataclass
class ConversationGuardViolation:
    type: Literal[
        "max_turns",
        "topic_drift",
        "cross_turn_pii",
        "risk_threshold",
        "agent_loop",
        "tool_call_limit",
    ]
    current_turn: int
    details: str
    cumulative_risk_score: Optional[float] = None


@dataclass
class TurnRecord:
    turn_number: int
    timestamp: float
    user_message_hash: str
    response_hash: str
    response_summary: str
    pii_types_detected: List[str] = field(default_factory=list)
    pii_values_hashed: List[str] = field(default_factory=list)
    tool_call_count: int = 0
    risk_contribution: float = 0.0


@dataclass
class RecordTurnInput:
    user_message: str
    response_text: str
    tool_call_count: int
    pii_detections: Optional[List[PIIDetection]] = None
    injection_risk_score: Optional[float] = None
    jailbreak_risk_score: Optional[float] = None


@dataclass
class ConversationSummary:
    turns: int
    cumulative_risk_score: float
    total_tool_calls: int
    unique_pii_types: List[str] = field(default_factory=list)
    pii_spread_detected: bool = False


# ── Helpers ──────────────────────────────────────────────────────────────────

_TOKEN_SPLIT = re.compile(r"[\s,.!?;:()\[\]{}'\"]+")
_MAX_HISTORY = 100


def _fnv1a(s: str) -> str:
    """FNV-1a hash for fast string hashing (non-cryptographic)."""
    h = 2166136261
    for ch in s:
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    # Convert to base-36 string
    if h == 0:
        return "0"
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"
    result = []
    while h:
        result.append(digits[h % 36])
        h //= 36
    return "".join(reversed(result))


def _jaccard_similarity(a: str, b: str) -> float:
    tokens_a = {t for t in _TOKEN_SPLIT.split(a.lower()) if len(t) > 2}
    tokens_b = {t for t in _TOKEN_SPLIT.split(b.lower()) if len(t) > 2}
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a & tokens_b)
    return intersection / (len(tokens_a) + len(tokens_b) - intersection)


# ── ConversationGuard ────────────────────────────────────────────────────────


class ConversationGuard:
    """Stateful guard that tracks conversation state across LLM calls."""

    def __init__(self, options: ConversationGuardOptions) -> None:
        self._options = options
        self._turns: List[TurnRecord] = []
        self._cumulative_risk = 0.0
        self._total_tools = 0
        self._pii_values_seen: set[str] = set()
        self._consecutive_similar = 0
        self._last_response_hash = ""
        self._baseline_message = ""
        self._pii_spread = False

    def check_pre_call(self) -> Optional[ConversationGuardViolation]:
        """Pre-call check: verify turn limits before making the LLM call."""
        if self._options.max_turns is not None and len(self._turns) >= self._options.max_turns:
            return ConversationGuardViolation(
                type="max_turns",
                current_turn=len(self._turns),
                details=f"Conversation reached {self._options.max_turns} turn limit",
            )
        if (
            self._options.max_total_tool_calls is not None
            and self._total_tools >= self._options.max_total_tool_calls
        ):
            return ConversationGuardViolation(
                type="tool_call_limit",
                current_turn=len(self._turns),
                details=f"Total tool calls ({self._total_tools}) reached limit ({self._options.max_total_tool_calls})",
            )
        return None

    def record_turn(self, input: RecordTurnInput) -> List[ConversationGuardViolation]:
        """Post-call: record the turn and check for violations."""
        violations: List[ConversationGuardViolation] = []
        turn_number = len(self._turns) + 1

        # Set baseline from first user message
        if not self._baseline_message and input.user_message:
            self._baseline_message = input.user_message

        # Hash response for loop detection
        resp_hash = _fnv1a(input.response_text[:500])
        if resp_hash == self._last_response_hash:
            self._consecutive_similar += 1
        else:
            self._consecutive_similar = 1
        self._last_response_hash = resp_hash

        # Agent loop detection
        max_similar = self._options.max_consecutive_similar_responses or 3
        if self._consecutive_similar >= max_similar:
            violations.append(
                ConversationGuardViolation(
                    type="agent_loop",
                    current_turn=turn_number,
                    details=f"{self._consecutive_similar} consecutive similar responses detected",
                )
            )

        # Tool call tracking
        self._total_tools += input.tool_call_count
        if (
            self._options.max_total_tool_calls is not None
            and self._total_tools > self._options.max_total_tool_calls
        ):
            violations.append(
                ConversationGuardViolation(
                    type="tool_call_limit",
                    current_turn=turn_number,
                    details=f"Total tool calls ({self._total_tools}) exceed limit ({self._options.max_total_tool_calls})",
                )
            )

        # Risk accumulation
        risk_contribution = (
            (input.injection_risk_score or 0) * 0.5
            + (input.jailbreak_risk_score or 0) * 0.3
            + (0.2 if input.tool_call_count > 5 else 0)
        )
        self._cumulative_risk += risk_contribution

        if self._options.accumulating_risk:
            threshold = self._options.risk_threshold if self._options.risk_threshold is not None else 2.0
            if self._cumulative_risk >= threshold:
                violations.append(
                    ConversationGuardViolation(
                        type="risk_threshold",
                        current_turn=turn_number,
                        details=f"Cumulative risk ({self._cumulative_risk:.2f}) exceeds threshold ({threshold})",
                        cumulative_risk_score=self._cumulative_risk,
                    )
                )

        # Cross-turn PII tracking
        pii_types: List[str] = []
        pii_hashes: List[str] = []
        if input.pii_detections:
            for d in input.pii_detections:
                pii_types.append(d.type)
                h = _fnv1a(d.value)
                pii_hashes.append(h)

                if self._options.cross_turn_pii_tracking and h in self._pii_values_seen:
                    self._pii_spread = True
                    violations.append(
                        ConversationGuardViolation(
                            type="cross_turn_pii",
                            current_turn=turn_number,
                            details=f'PII type "{d.type}" detected in previous turn appeared again in turn {turn_number}',
                        )
                    )
                self._pii_values_seen.add(h)

        # Topic drift
        if self._options.topic_drift_detection and self._baseline_message and turn_number > 1:
            user_tokens = [t for t in _TOKEN_SPLIT.split(input.user_message.lower()) if len(t) > 2]
            if len(user_tokens) >= 10:
                similarity = _jaccard_similarity(input.user_message, self._baseline_message)
                threshold = self._options.topic_drift_threshold if self._options.topic_drift_threshold is not None else 0.3
                if similarity < threshold:
                    violations.append(
                        ConversationGuardViolation(
                            type="topic_drift",
                            current_turn=turn_number,
                            details=f"Topic drift detected (similarity: {similarity:.2f}, threshold: {threshold})",
                        )
                    )

        # Record turn
        self._turns.append(
            TurnRecord(
                turn_number=turn_number,
                timestamp=time.time(),
                user_message_hash=_fnv1a(input.user_message),
                response_hash=resp_hash,
                response_summary=input.response_text[:200],
                pii_types_detected=pii_types,
                pii_values_hashed=pii_hashes,
                tool_call_count=input.tool_call_count,
                risk_contribution=risk_contribution,
            )
        )

        # Prune old turns
        if len(self._turns) > _MAX_HISTORY:
            self._turns = self._turns[-_MAX_HISTORY:]

        return violations

    @property
    def turn_count(self) -> int:
        return len(self._turns)

    @property
    def risk_score(self) -> float:
        return self._cumulative_risk

    @property
    def tool_calls(self) -> int:
        return self._total_tools

    def reset(self) -> None:
        self._turns.clear()
        self._cumulative_risk = 0.0
        self._total_tools = 0
        self._pii_values_seen.clear()
        self._consecutive_similar = 0
        self._last_response_hash = ""
        self._baseline_message = ""
        self._pii_spread = False

    def get_summary(self) -> ConversationSummary:
        all_pii_types: set[str] = set()
        for turn in self._turns:
            for t in turn.pii_types_detected:
                all_pii_types.add(t)
        return ConversationSummary(
            turns=len(self._turns),
            cumulative_risk_score=self._cumulative_risk,
            total_tool_calls=self._total_tools,
            unique_pii_types=sorted(all_pii_types),
            pii_spread_detected=self._pii_spread,
        )
