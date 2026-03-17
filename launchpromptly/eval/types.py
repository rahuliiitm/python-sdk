"""Types for the eval CLI."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

GuardrailName = Literal[
    "injection", "jailbreak", "pii", "content", "unicode", "secrets", "tool_guard"
]


@dataclass
class TestCase:  # noqa: N801 — naming matches Node SDK
    """A single eval test case."""

    __test__ = False  # prevent pytest collection

    prompt: str
    expected: Literal["blocked", "allowed", "detected"]
    pii_types: Optional[List[str]] = None
    label: Optional[str] = None


@dataclass
class EvalSuite:
    guardrail: GuardrailName
    cases: List[TestCase] = field(default_factory=list)


@dataclass
class EvalConfig:
    suites: List[EvalSuite]
    name: Optional[str] = None
    threshold: Optional[float] = None


@dataclass
class TestCaseResult:
    prompt: str
    expected: str
    actual: str
    passed: bool
    latency_ms: float
    label: Optional[str] = None
    details: Optional[str] = None


@dataclass
class SuiteMetrics:
    guardrail: str
    total: int
    passed: int
    failed: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    pass_rate: float
    avg_latency_ms: float


@dataclass
class SuiteResult:
    guardrail: str
    cases: List[TestCaseResult]
    metrics: SuiteMetrics


@dataclass
class EvalReport:
    name: str
    timestamp: str
    suites: List[SuiteResult]
    overall_total: int
    overall_passed: int
    overall_failed: int
    overall_pass_rate: float
    overall_avg_latency_ms: float
