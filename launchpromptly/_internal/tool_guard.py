"""Tool-use validation module -- guards which tools an agent can call,
validates argument safety, and scans tool results.
Zero dependencies. Stateless, pure functions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from .pii import detect_pii
from .injection import detect_injection
from .secret_detection import detect_secrets
from .schema_validator import validate_schema

# ── Types ────────────────────────────────────────────────────────────────────


@dataclass
class ToolGuardOptions:
    enabled: Optional[bool] = None
    allowed_tools: Optional[List[str]] = None
    """Only these tool names are allowed. Mutually exclusive with blocked_tools."""
    blocked_tools: Optional[List[str]] = None
    """These tool names are blocked."""
    parameter_schemas: Optional[Dict[str, Dict[str, Any]]] = None
    """Per-tool JSON schema for argument validation. Key = tool name."""
    dangerous_arg_detection: Optional[bool] = None
    """Detect SQL injection, path traversal, shell injection, SSRF in args."""
    dangerous_tool_patterns: Optional[List[str]] = None
    """Regex patterns for inherently dangerous tools."""
    max_tool_calls_per_turn: Optional[int] = None
    """Max tool calls allowed per single LLM response."""
    scan_tool_results: Optional[bool] = None
    """Scan tool result text for PII/injection/secrets."""
    action: Optional[Literal["block", "warn", "flag"]] = None
    """Action on violation. Default: 'block'."""
    on_violation: Optional[Callable[[ToolGuardViolation], None]] = None
    """Callback on any violation."""


@dataclass
class ToolGuardViolation:
    type: Literal[
        "blocked_tool",
        "unlisted_tool",
        "invalid_args",
        "dangerous_args",
        "dangerous_tool",
        "frequency_limit",
        "turn_limit",
        "tool_result_threat",
    ]
    tool_name: str
    details: str
    arg_snippet: Optional[str] = None
    """Offending argument snippet (truncated to 200 chars)."""


@dataclass
class ToolGuardCheckResult:
    violations: List[ToolGuardViolation] = field(default_factory=list)
    blocked: bool = False


@dataclass
class ToolCallInfo:
    name: str
    arguments: Any  # str or dict


@dataclass
class ToolResultThreat:
    type: Literal["pii", "injection", "secret"]
    pii_type: Optional[str] = None
    details: str = ""


@dataclass
class ToolResultScanReport:
    tool_name: str
    threats: List[ToolResultThreat] = field(default_factory=list)
    clean: bool = True


# ── Built-in dangerous patterns ──────────────────────────────────────────────

_DEFAULT_DANGEROUS_TOOL_PATTERNS = [
    re.compile(r"^(file_write|write_file|create_file)", re.IGNORECASE),
    re.compile(r"^(exec|execute|run_command|shell|bash|cmd)", re.IGNORECASE),
    re.compile(r"^(http_request|fetch|curl|wget|network)", re.IGNORECASE),
    re.compile(r"^(delete|remove|drop|truncate)", re.IGNORECASE),
    re.compile(r"^(eval|code_interpreter|python_exec)", re.IGNORECASE),
    re.compile(r"^(send_email|send_message|post_to)", re.IGNORECASE),
]

_DANGEROUS_ARG_CATEGORIES = [
    {
        "category": "sql_injection",
        "patterns": [
            re.compile(r"\bUNION\s+SELECT\b", re.IGNORECASE),
            re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
            re.compile(r";\s*DELETE\s+FROM\b", re.IGNORECASE),
            re.compile(r"\bOR\s+1\s*=\s*1\b", re.IGNORECASE),
            re.compile(r"\bOR\s+'[^']*'\s*=\s*'[^']*'", re.IGNORECASE),
            re.compile(r"--\s*$", re.MULTILINE),
            re.compile(r";\s*INSERT\s+INTO\b", re.IGNORECASE),
            re.compile(r"\bSELECT\s+.*\bFROM\s+.*\bWHERE\b.*\bOR\b", re.IGNORECASE),
        ],
    },
    {
        "category": "path_traversal",
        "patterns": [
            re.compile(r"\.\./"),
            re.compile(r"\.\.\\"),
            re.compile(r"%2e%2e", re.IGNORECASE),
            re.compile(r"%252e%252e", re.IGNORECASE),
        ],
    },
    {
        "category": "shell_injection",
        "patterns": [
            re.compile(r"\$\([^)]+\)"),
            re.compile(r"`[^`]+`"),
            re.compile(r";\s*(?:rm|cat|curl|wget|nc|bash|sh|chmod|chown)\b", re.IGNORECASE),
            re.compile(r"\|\s*(?:bash|sh|zsh)\b", re.IGNORECASE),
            re.compile(r"&&\s*(?:rm|curl|wget)\b", re.IGNORECASE),
        ],
    },
    {
        "category": "ssrf",
        "patterns": [
            re.compile(r"169\.254\.169\.254"),
            re.compile(r"(?:https?://)?localhost(?::\d+)?", re.IGNORECASE),
            re.compile(r"(?:https?://)?127\.0\.0\.1(?::\d+)?"),
            re.compile(r"(?:https?://)?0\.0\.0\.0(?::\d+)?"),
            re.compile(r"(?:https?://)\[?::1\]?(?::\d+)?"),
            re.compile(r"(?:https?://)?10\.\d+\.\d+\.\d+"),
            re.compile(r"(?:https?://)?172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+"),
            re.compile(r"(?:https?://)?192\.168\.\d+\.\d+"),
        ],
    },
]

# ── Helpers ──────────────────────────────────────────────────────────────────


def _truncate(s: str, max_len: int = 200) -> str:
    return s[:max_len] + "..." if len(s) > max_len else s


def _stringify_args(args: Any) -> str:
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args)
    except (TypeError, ValueError):
        return str(args)


def _matches_pattern(name: str, pattern: str) -> bool:
    if "*" in pattern:
        regex = re.compile("^" + pattern.replace("*", ".*") + "$", re.IGNORECASE)
        return bool(regex.match(name))
    return name.lower() == pattern.lower()


def _is_dangerous_tool(
    name: str, custom_patterns: Optional[List[str]] = None
) -> tuple[bool, Optional[str]]:
    if custom_patterns is not None:
        for p in custom_patterns:
            if _matches_pattern(name, p):
                return True, p
        return False, None

    for rx in _DEFAULT_DANGEROUS_TOOL_PATTERNS:
        if rx.search(name):
            return True, rx.pattern
    return False, None


# ── Public API ───────────────────────────────────────────────────────────────


def check_tool_calls(
    tool_calls: List[ToolCallInfo],
    options: ToolGuardOptions,
    context: Optional[Dict[str, Any]] = None,
) -> ToolGuardCheckResult:
    """Validate tool calls against tool guard configuration.
    Returns violations found. Pure function, no state.
    """
    violations: List[ToolGuardViolation] = []

    # Turn limit check
    if (
        options.max_tool_calls_per_turn is not None
        and len(tool_calls) > options.max_tool_calls_per_turn
    ):
        violations.append(
            ToolGuardViolation(
                type="turn_limit",
                tool_name="*",
                details=f"{len(tool_calls)} tool calls exceed max_tool_calls_per_turn ({options.max_tool_calls_per_turn})",
            )
        )

    for call in tool_calls:
        name = call.name
        args_str = _stringify_args(call.arguments)

        # Whitelist check
        if options.allowed_tools is not None:
            allowed = any(_matches_pattern(name, p) for p in options.allowed_tools)
            if not allowed:
                violations.append(
                    ToolGuardViolation(
                        type="unlisted_tool",
                        tool_name=name,
                        details=f'Tool "{name}" is not in allowed_tools',
                    )
                )
                continue

        # Blacklist check
        if options.blocked_tools and len(options.blocked_tools) > 0:
            blocked = any(_matches_pattern(name, p) for p in options.blocked_tools)
            if blocked:
                violations.append(
                    ToolGuardViolation(
                        type="blocked_tool",
                        tool_name=name,
                        details=f'Tool "{name}" is in blocked_tools',
                    )
                )
                continue

        # Dangerous tool pattern check
        dangerous, pattern = _is_dangerous_tool(name, options.dangerous_tool_patterns)
        if dangerous:
            violations.append(
                ToolGuardViolation(
                    type="dangerous_tool",
                    tool_name=name,
                    details=f'Tool "{name}" matches dangerous pattern: {pattern}',
                )
            )

        # Parameter schema validation
        if options.parameter_schemas and name in options.parameter_schemas:
            schema = options.parameter_schemas[name]
            try:
                args_obj = (
                    json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments
                )
            except (json.JSONDecodeError, TypeError):
                violations.append(
                    ToolGuardViolation(
                        type="invalid_args",
                        tool_name=name,
                        details=f'Malformed JSON arguments for tool "{name}"',
                        arg_snippet=_truncate(args_str),
                    )
                )
                continue

            errors = validate_schema(args_obj, schema)
            if errors:
                violations.append(
                    ToolGuardViolation(
                        type="invalid_args",
                        tool_name=name,
                        details=f"Schema validation failed: {'; '.join(e.message for e in errors)}",
                        arg_snippet=_truncate(args_str),
                    )
                )

        # Dangerous arg detection
        if options.dangerous_arg_detection:
            dangerous_args = detect_dangerous_args(args_str)
            for d in dangerous_args:
                violations.append(
                    ToolGuardViolation(
                        type="dangerous_args",
                        tool_name=name,
                        details=f"Dangerous {d['category']} pattern in arguments",
                        arg_snippet=_truncate(d["matched"]),
                    )
                )

    action = options.action or "block"
    return ToolGuardCheckResult(
        violations=violations,
        blocked=action == "block" and len(violations) > 0,
    )


def detect_dangerous_args(args: str) -> List[Dict[str, str]]:
    """Detect dangerous patterns in tool call argument text."""
    results: List[Dict[str, str]] = []
    for cat in _DANGEROUS_ARG_CATEGORIES:
        for rx in cat["patterns"]:
            match = rx.search(args)
            if match:
                results.append({"category": cat["category"], "matched": match.group(0)})
                break  # One match per category is enough
    return results


def scan_tool_result(
    tool_name: str,
    result_text: str,
    options: Optional[Dict[str, Any]] = None,
) -> ToolResultScanReport:
    """Scan tool result text for PII, injection patterns, and secrets."""
    threats: List[ToolResultThreat] = []

    # PII scan
    pii_detections = detect_pii(result_text)
    for d in pii_detections:
        threats.append(
            ToolResultThreat(
                type="pii",
                pii_type=d.type,
                details=f'{d.type} detected in result from "{tool_name}"',
            )
        )

    # Injection scan
    inj_result = detect_injection(result_text)
    threshold = (options or {}).get("injection_threshold", 0.5)
    if inj_result.risk_score >= threshold:
        threats.append(
            ToolResultThreat(
                type="injection",
                details=f'Injection risk {inj_result.risk_score:.2f} in result from "{tool_name}" ({", ".join(inj_result.triggered)})',
            )
        )

    # Secret scan
    secrets = detect_secrets(result_text)
    for s in secrets:
        threats.append(
            ToolResultThreat(
                type="secret",
                details=f'{s.type} detected in result from "{tool_name}"',
            )
        )

    return ToolResultScanReport(
        tool_name=tool_name,
        threats=threats,
        clean=len(threats) == 0,
    )
