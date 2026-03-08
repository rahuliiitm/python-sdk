"""
Output safety scanning module -- detects dangerous executable content in LLM output.
Different from content filter (which catches hate/violence/toxic content).
This catches operationally dangerous patterns: destructive commands, SQL injection,
suspicious URLs, and dangerous code constructs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Set

OutputSafetyCategory = Literal[
    "dangerous_commands",
    "sql_injection",
    "suspicious_urls",
    "dangerous_code",
]

OutputSafetySeverity = Literal["warn", "block"]


@dataclass
class OutputSafetyOptions:
    """Options for output safety scanning."""

    categories: Optional[List[OutputSafetyCategory]] = None  # defaults to all four


@dataclass
class OutputSafetyThreat:
    category: OutputSafetyCategory
    matched: str
    severity: OutputSafetySeverity
    context: str  # +-50 characters surrounding the match, clamped to text bounds


_MAX_SCAN_LENGTH = 1_000_000  # 1 MB


# -- Category rules ------------------------------------------------------------

@dataclass
class _CategoryRule:
    category: OutputSafetyCategory
    patterns: List[re.Pattern[str]]
    severity: OutputSafetySeverity


_CATEGORIES: List[_CategoryRule] = [
    _CategoryRule(
        category="dangerous_commands",
        severity="block",
        patterns=[
            re.compile(r"\brm\s+-rf\b", re.IGNORECASE),
            re.compile(r"\bdel\s+/f\s+/s", re.IGNORECASE),
            re.compile(r"\bformat\s+c:", re.IGNORECASE),
            re.compile(r"\bDROP\s+TABLE\b", re.IGNORECASE),
            re.compile(r"\bDELETE\s+FROM\b", re.IGNORECASE),
            re.compile(r"\bTRUNCATE\s+TABLE\b", re.IGNORECASE),
            re.compile(r"\bshutdown\s+-h\b", re.IGNORECASE),
            re.compile(r"\bmkfs\.", re.IGNORECASE),
            re.compile(r"\bdd\s+if=/dev/zero", re.IGNORECASE),
            re.compile(r"\bchmod\s+-R\s+777\s+/", re.IGNORECASE),
        ],
    ),
    _CategoryRule(
        category="sql_injection",
        severity="warn",
        patterns=[
            re.compile(r"['\"];\s*DROP\b", re.IGNORECASE),
            re.compile(r"\bOR\s+1\s*=\s*1\b", re.IGNORECASE),
            re.compile(r"\bUNION\s+SELECT\b", re.IGNORECASE),
            re.compile(r"\bINTO\s+OUTFILE\b", re.IGNORECASE),
            re.compile(r"\bLOAD_FILE\s*\(", re.IGNORECASE),
            re.compile(r"\bxp_cmdshell\b", re.IGNORECASE),
        ],
    ),
    _CategoryRule(
        category="suspicious_urls",
        severity="warn",
        patterns=[
            # IP-based URLs (not localhost / 127.0.0.1 / 0.0.0.0)
            re.compile(r"https?://(?!127\.0\.0\.1\b|0\.0\.0\.0\b|localhost\b)\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"),
            re.compile(r"https?://[^\s]*\.onion\b", re.IGNORECASE),
            re.compile(r"\bdata:[^;]+;base64,", re.IGNORECASE),
            re.compile(r"\bjavascript:", re.IGNORECASE),
        ],
    ),
    _CategoryRule(
        category="dangerous_code",
        severity="warn",
        patterns=[
            re.compile(r"\beval\s*\(", re.IGNORECASE),
            re.compile(r"\bexec\s*\(", re.IGNORECASE),
            re.compile(r"\bos\.system\s*\(", re.IGNORECASE),
            re.compile(r"\bsubprocess\.call\s*\(", re.IGNORECASE),
            re.compile(r"\b__import__\s*\(", re.IGNORECASE),
            re.compile(r"\bchild_process\.exec\s*\(", re.IGNORECASE),
            re.compile(r"\bnew\s+Function\s*\(", re.IGNORECASE),
        ],
    ),
]


# -- Detection -----------------------------------------------------------------

def _extract_context(text: str, match_start: int, match_end: int) -> str:
    """Extract context string: +-50 characters around the match, clamped to text bounds."""
    ctx_start = max(0, match_start - 50)
    ctx_end = min(len(text), match_end + 50)
    return text[ctx_start:ctx_end]


def scan_output_safety(
    text: str,
    options: Optional[OutputSafetyOptions] = None,
) -> List[OutputSafetyThreat]:
    """Scan LLM output text for operationally dangerous content.
    Returns all detected threats sorted by their position in the text.
    Text longer than 1 MB is truncated before scanning.
    """
    if not text:
        return []

    # Cap input length to prevent DoS
    scan_text = text[:_MAX_SCAN_LENGTH] if len(text) > _MAX_SCAN_LENGTH else text

    allowed_categories: Optional[Set[str]] = None
    if options and options.categories:
        allowed_categories = set(options.categories)

    threats: List[tuple[OutputSafetyThreat, int]] = []

    for rule in _CATEGORIES:
        if allowed_categories is not None and rule.category not in allowed_categories:
            continue

        for pattern in rule.patterns:
            for match in pattern.finditer(scan_text):
                threats.append((
                    OutputSafetyThreat(
                        category=rule.category,
                        matched=match.group(0),
                        severity=rule.severity,
                        context=_extract_context(
                            scan_text,
                            match.start(),
                            match.end(),
                        ),
                    ),
                    match.start(),
                ))

    # Sort by position in text
    threats.sort(key=lambda t: t[1])

    return [t[0] for t in threats]
