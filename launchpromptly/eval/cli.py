"""CLI for running guardrail evaluations.

Usage:
    python -m launchpromptly eval
    python -m launchpromptly eval --filter injection,jailbreak
    python -m launchpromptly eval --format json
    python -m launchpromptly eval --threshold 0.95
"""

from __future__ import annotations

import sys
from typing import List, Optional

from .corpus import load_corpus
from .formatters import format_csv, format_json, format_markdown
from .runner import run_eval
from .types import EvalConfig


def _parse_args(argv: List[str]) -> dict:
    """Parse CLI arguments."""
    result = {
        "filter": None,
        "format": "markdown",
        "threshold": None,
        "config": None,
        "help": False,
    }

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ("--help", "-h"):
            result["help"] = True
        elif arg == "--filter" and i + 1 < len(argv):
            i += 1
            result["filter"] = [s.strip() for s in argv[i].split(",")]
        elif arg == "--format" and i + 1 < len(argv):
            i += 1
            fmt = argv[i].lower()
            if fmt in ("json", "csv", "markdown"):
                result["format"] = fmt
        elif arg == "--threshold" and i + 1 < len(argv):
            i += 1
            result["threshold"] = float(argv[i])
        elif arg == "--config" and i + 1 < len(argv):
            i += 1
            result["config"] = argv[i]
        i += 1

    return result


def eval_main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for the eval CLI."""
    if argv is None:
        argv = sys.argv[1:]

    args = _parse_args(argv)

    if args["help"]:
        print(
            """
launchpromptly eval — Run guardrail test suites

Usage:
  python -m launchpromptly eval [options]

Options:
  --filter <guardrails>   Comma-separated guardrails (injection,jailbreak,pii,content,unicode,secrets,tool_guard)
  --format <format>       Output format: markdown (default), json, csv
  --threshold <number>    Minimum pass rate (0-1). Exit code 1 if below threshold
  --config <path>         Path to YAML/JSON config file
  -h, --help              Show this help
"""
        )
        return 0

    # Load test suites
    suites = load_corpus(args["filter"])

    if not suites:
        print("No test suites found. Check --filter values.", file=sys.stderr)
        return 1

    config = EvalConfig(
        name="LaunchPromptly Eval",
        suites=suites,
        threshold=args["threshold"],
    )

    # Run evaluation
    report = run_eval(config)

    # Output results
    fmt = args["format"]
    if fmt == "json":
        print(format_json(report))
    elif fmt == "csv":
        print(format_csv(report))
    else:
        print(format_markdown(report))

    # Check threshold
    if args["threshold"] is not None and report.overall_pass_rate < args["threshold"]:
        print(
            f"\nFAILED: Pass rate {report.overall_pass_rate * 100:.1f}% "
            f"below threshold {args['threshold'] * 100:.1f}%",
            file=sys.stderr,
        )
        return 1

    return 0
