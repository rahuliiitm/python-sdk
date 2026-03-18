"""Types for the Red Team Engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Pattern


# ── Attack Categories ────────────────────────────────────────────────────────

AttackCategory = Literal[
    "injection",
    "jailbreak",
    "pii_extraction",
    "prompt_leakage",
    "content_bypass",
    "encoding_evasion",
    "multi_turn",
    "tool_abuse",
]

# ── Attack Payloads ──────────────────────────────────────────────────────────


@dataclass
class AttackPayload:
    """A single red team attack."""

    id: str
    category: AttackCategory
    name: str
    messages: List[Dict[str, str]]
    expected_outcome: Literal["blocked", "redacted", "warned", "refused"]
    severity: Literal["critical", "high", "medium", "low"]
    description: str
    reference: Optional[str] = None
    success_indicators: Optional[List[Pattern]] = None  # type: ignore[type-arg]


# ── Attack Results ───────────────────────────────────────────────────────────

AttackOutcome = Literal[
    "blocked", "redacted", "refused", "bypassed", "error", "inconclusive"
]


@dataclass
class GuardrailEventCapture:
    type: str
    data: Dict[str, Any]
    timestamp: float


@dataclass
class AttackResult:
    """Result of a single attack execution."""

    attack: AttackPayload
    outcome: AttackOutcome
    guardrail_events: List[GuardrailEventCapture]
    latency_ms: float
    analysis_reason: str
    response_preview: Optional[str] = None
    error: Optional[str] = None


# ── Red Team Options ─────────────────────────────────────────────────────────


@dataclass
class RedTeamProgress:
    completed: int
    total: int
    current_attack: str
    current_category: AttackCategory


@dataclass
class RedTeamOptions:
    """Configuration for a red team run."""

    categories: Optional[List[AttackCategory]] = None
    max_attacks: int = 50
    concurrency: int = 3
    delay_ms: float = 500
    system_prompt: Optional[str] = None
    custom_attacks: Optional[List[AttackPayload]] = None
    on_progress: Optional[Callable[[RedTeamProgress], None]] = None
    model: Optional[str] = None
    dry_run: bool = False
    contextual_attacks: Optional[bool] = None  # Default: True if system_prompt provided


# ── Report ───────────────────────────────────────────────────────────────────


@dataclass
class CategoryScore:
    category: AttackCategory
    score: int
    total: int
    blocked: int
    refused: int
    bypassed: int
    errors: int
    inconclusive: int


@dataclass
class Vulnerability:
    severity: Literal["critical", "high", "medium", "low"]
    category: AttackCategory
    attack_name: str
    attack_id: str
    description: str
    remediation: str
    response_preview: Optional[str] = None


@dataclass
class RedTeamReport:
    """Full red team security report."""

    security_score: int
    categories: List[CategoryScore]
    attacks: List[AttackResult]
    vulnerabilities: List[Vulnerability]
    total_attacks: int
    total_duration_ms: float
    estimated_cost_usd: float
    timestamp: str
