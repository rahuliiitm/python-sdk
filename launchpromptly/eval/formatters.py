"""Output formatters for eval results."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from .types import EvalReport


def format_json(report: EvalReport) -> str:
    """Format report as JSON string."""
    data = asdict(report)
    # Restructure overall for consistency with Node SDK
    data["overall"] = {
        "total": data.pop("overall_total"),
        "passed": data.pop("overall_passed"),
        "failed": data.pop("overall_failed"),
        "passRate": data.pop("overall_pass_rate"),
        "avgLatencyMs": data.pop("overall_avg_latency_ms"),
    }
    return json.dumps(data, indent=2)


def format_csv(report: EvalReport) -> str:
    """Format report as CSV."""
    lines = ["guardrail,prompt,expected,actual,pass,latency_ms,label"]
    for suite in report.suites:
        for c in suite.cases:
            prompt = c.prompt.replace('"', '""')
            lines.append(
                f'{suite.guardrail},"{prompt}",{c.expected},{c.actual},{c.passed},{c.latency_ms},{c.label or ""}'
            )
    return "\n".join(lines)


def format_markdown(report: EvalReport) -> str:
    """Format report as markdown table."""
    lines: list[str] = []

    lines.append(f"# {report.name}")
    lines.append(f"_{report.timestamp}_\n")

    # Summary
    lines.append("## Summary")
    lines.append(f"- **Total:** {report.overall_total}")
    lines.append(f"- **Passed:** {report.overall_passed}")
    lines.append(f"- **Failed:** {report.overall_failed}")
    lines.append(f"- **Pass Rate:** {report.overall_pass_rate * 100:.1f}%")
    lines.append(f"- **Avg Latency:** {report.overall_avg_latency_ms}ms\n")

    # Per-suite metrics
    lines.append("## Guardrail Metrics\n")
    lines.append("| Guardrail | Total | Pass | Fail | Precision | Recall | F1 | Pass Rate |")
    lines.append("|-----------|-------|------|------|-----------|--------|----|-----------|")

    for suite in report.suites:
        m = suite.metrics
        lines.append(
            f"| {m.guardrail} | {m.total} | {m.passed} | {m.failed} "
            f"| {m.precision} | {m.recall} | {m.f1} | {m.pass_rate * 100:.1f}% |"
        )

    lines.append("")

    # Failed cases
    failed_cases = []
    for s in report.suites:
        for c in s.cases:
            if not c.passed:
                failed_cases.append((s.guardrail, c))

    if failed_cases:
        lines.append(f"## Failed Cases ({len(failed_cases)})\n")
        lines.append("| Guardrail | Label | Expected | Actual | Prompt (truncated) |")
        lines.append("|-----------|-------|----------|--------|-------------------|")
        for guardrail, c in failed_cases:
            prompt = c.prompt[:50] + "..." if len(c.prompt) > 50 else c.prompt
            lines.append(f"| {guardrail} | {c.label or '-'} | {c.expected} | {c.actual} | {prompt} |")
    else:
        lines.append("All tests passed!")

    return "\n".join(lines)
