"""Evaluation engine — runs test cases against guardrails, collects metrics."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

from .types import (
    EvalConfig,
    EvalReport,
    EvalSuite,
    SuiteMetrics,
    SuiteResult,
    TestCase,
    TestCaseResult,
)


def _run_case(guardrail: str, tc: TestCase) -> Tuple[str, Optional[str]]:
    """Run a single test case against its guardrail. Returns (actual, details)."""
    if guardrail == "injection":
        from .._internal.injection import detect_injection

        result = detect_injection(tc.prompt)
        if result.risk_score >= 0.5:
            return "blocked", f"risk={result.risk_score}"
        return "allowed", None

    if guardrail == "jailbreak":
        from .._internal.jailbreak import detect_jailbreak

        result = detect_jailbreak(tc.prompt)
        if result.risk_score >= 0.5:
            return "blocked", f"risk={result.risk_score}"
        return "allowed", None

    if guardrail == "pii":
        from .._internal.pii import detect_pii

        detections = detect_pii(tc.prompt)
        if detections:
            types = sorted({d.type for d in detections})
            return "detected", ",".join(types)
        return "allowed", None

    if guardrail == "content":
        from .._internal.content_filter import detect_content_violations, has_blocking_violation

        violations = detect_content_violations(tc.prompt, "input")
        if has_blocking_violation(violations):
            return "blocked", ",".join(v.category for v in violations)
        return "allowed", None

    if guardrail == "unicode":
        from .._internal.unicode_sanitizer import scan_unicode

        result = scan_unicode(tc.prompt)
        if result.threats:
            return "detected", ",".join(t.type for t in result.threats)
        return "allowed", None

    if guardrail == "secrets":
        from .._internal.secret_detection import detect_secrets

        detections = detect_secrets(tc.prompt)
        if detections:
            return "detected", ",".join(d.type for d in detections)
        return "allowed", None

    if guardrail == "tool_guard":
        from .._internal.tool_guard import ToolCallInfo, ToolGuardOptions, check_tool_calls

        try:
            parsed = json.loads(tc.prompt)
            tool_call = ToolCallInfo(
                name=parsed.get("tool", ""),
                arguments=parsed["args"] if isinstance(parsed.get("args"), str) else json.dumps(parsed.get("args", {})),
            )
            result = check_tool_calls(
                [tool_call],
                ToolGuardOptions(dangerous_arg_detection=True, action="block"),
            )
            if result.blocked:
                return "blocked", ",".join(v.type for v in result.violations)
            return "allowed", None
        except (json.JSONDecodeError, KeyError):
            return "allowed", "invalid JSON"

    return "allowed", f"unknown guardrail: {guardrail}"


def _compute_metrics(guardrail: str, results: list[TestCaseResult]) -> SuiteMetrics:
    """Compute precision, recall, F1, and pass rate from test case results."""
    tp = fp = tn = fn = 0

    for r in results:
        expected_positive = r.expected in ("blocked", "detected")
        actual_positive = r.actual in ("blocked", "detected")

        if expected_positive and actual_positive:
            tp += 1
        elif not expected_positive and actual_positive:
            fp += 1
        elif not expected_positive and not actual_positive:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    passed = sum(1 for r in results if r.passed)
    total_latency = sum(r.latency_ms for r in results)

    return SuiteMetrics(
        guardrail=guardrail,
        total=len(results),
        passed=passed,
        failed=len(results) - passed,
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        pass_rate=round(passed / len(results), 3) if results else 1.0,
        avg_latency_ms=round(total_latency / len(results), 2) if results else 0.0,
    )


def _run_suite(suite: EvalSuite) -> SuiteResult:
    """Run a suite of test cases against a guardrail."""
    results: list[TestCaseResult] = []

    for tc in suite.cases:
        start = time.perf_counter()
        actual, details = _run_case(suite.guardrail, tc)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        results.append(
            TestCaseResult(
                prompt=tc.prompt,
                expected=tc.expected,
                actual=actual,
                passed=(actual == tc.expected),
                latency_ms=latency_ms,
                label=tc.label,
                details=details,
            )
        )

    return SuiteResult(
        guardrail=suite.guardrail,
        cases=results,
        metrics=_compute_metrics(suite.guardrail, results),
    )


def run_eval(config: EvalConfig) -> EvalReport:
    """Run the evaluation engine."""
    suite_results = [_run_suite(suite) for suite in config.suites]

    total_cases = sum(r.metrics.total for r in suite_results)
    total_passed = sum(r.metrics.passed for r in suite_results)
    total_latency = sum(r.metrics.avg_latency_ms * r.metrics.total for r in suite_results)

    return EvalReport(
        name=config.name or "LaunchPromptly Eval",
        timestamp=datetime.now(timezone.utc).isoformat(),
        suites=suite_results,
        overall_total=total_cases,
        overall_passed=total_passed,
        overall_failed=total_cases - total_passed,
        overall_pass_rate=round(total_passed / total_cases, 3) if total_cases > 0 else 1.0,
        overall_avg_latency_ms=round(total_latency / total_cases, 2) if total_cases > 0 else 0.0,
    )
