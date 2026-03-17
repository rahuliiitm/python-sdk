"""Tests for output safety scanning module."""
import time

import pytest

from launchpromptly._internal.output_safety import (
    OutputSafetyOptions,
    OutputSafetyThreat,
    scan_output_safety,
)


# -- Dangerous commands --------------------------------------------------------


def test_detects_rm_rf():
    result = scan_output_safety("Run this command: rm -rf /")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1
    assert "rm" in cmds[0].matched
    assert cmds[0].severity == "block"


def test_detects_del_f_s():
    result = scan_output_safety("Execute: del /f /s C:\\temp")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_format_c():
    result = scan_output_safety("Run format c: to wipe the disk")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_drop_table():
    result = scan_output_safety("Execute: DROP TABLE users;")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1
    assert "DROP" in cmds[0].matched.upper()
    assert "TABLE" in cmds[0].matched.upper()


def test_detects_delete_from():
    result = scan_output_safety("Try DELETE FROM accounts WHERE 1=1;")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1
    assert "DELETE" in cmds[0].matched.upper()
    assert "FROM" in cmds[0].matched.upper()


def test_detects_truncate_table():
    result = scan_output_safety("Run TRUNCATE TABLE logs;")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_shutdown_h():
    result = scan_output_safety("Run shutdown -h now to power off")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_mkfs_commands():
    result = scan_output_safety("Format with mkfs.ext4 /dev/sda1")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_dd_dev_zero():
    result = scan_output_safety("Run dd if=/dev/zero of=/dev/sda")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_chmod_777():
    result = scan_output_safety("Fix permissions: chmod -R 777 /")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


# -- SQL injection -------------------------------------------------------------


def test_detects_semicolon_drop():
    result = scan_output_safety("Input: '; DROP TABLE users; --")
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1
    assert sqli[0].severity == "warn"


def test_detects_or_1_equals_1():
    result = scan_output_safety("Use this: ' OR 1=1 --")
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1


def test_detects_union_select():
    result = scan_output_safety("Try: ' UNION SELECT * FROM passwords --")
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1


def test_detects_into_outfile():
    result = scan_output_safety("SELECT * INTO OUTFILE '/tmp/data.txt'")
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1


def test_detects_load_file():
    result = scan_output_safety("SELECT LOAD_FILE('/etc/passwd')")
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1


def test_detects_xp_cmdshell():
    result = scan_output_safety('EXEC xp_cmdshell "dir"')
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1


# -- Suspicious URLs -----------------------------------------------------------


def test_detects_ip_based_urls():
    result = scan_output_safety("Visit http://192.168.1.100/payload")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) >= 1
    assert urls[0].severity == "warn"


def test_does_not_flag_localhost_127():
    result = scan_output_safety("Visit http://127.0.0.1:3000/api")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) == 0


def test_does_not_flag_0000():
    result = scan_output_safety("Bind to http://0.0.0.0:8080")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) == 0


def test_detects_onion_urls():
    result = scan_output_safety("Go to http://darksite.onion/market")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) >= 1


def test_detects_data_base64_uris():
    result = scan_output_safety("Use this: data:text/html;base64,PHNjcmlwdD4=")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) >= 1


def test_detects_javascript_uris():
    result = scan_output_safety("Click: javascript:alert(1)")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) >= 1


# -- Dangerous code ------------------------------------------------------------


def test_detects_eval():
    result = scan_output_safety('Use eval("user_input") to run it')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1
    assert code[0].severity == "warn"


def test_detects_exec():
    result = scan_output_safety('Run exec("command") in Python')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


def test_detects_os_system():
    result = scan_output_safety('Run os.system("whoami") in Python')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


def test_detects_subprocess_call():
    result = scan_output_safety('Use subprocess.call(["ls", "-la"])')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


def test_detects_dunder_import():
    result = scan_output_safety('__import__("os").system("id")')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


def test_detects_child_process_exec():
    result = scan_output_safety('require("child_process").exec("ls")')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


def test_detects_new_function():
    result = scan_output_safety('const fn = new Function("return 1")')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


# -- Clean text (false positive resistance) ------------------------------------


def test_clean_normal_english():
    result = scan_output_safety(
        "The quick brown fox jumps over the lazy dog. This is perfectly safe."
    )
    assert len(result) == 0


def test_clean_programming_discussion():
    result = scan_output_safety(
        "To remove a file in Linux, use the rm command. "
        "For example: rm myfile.txt deletes a single file."
    )
    assert len(result) == 0


def test_clean_normal_sql():
    result = scan_output_safety("SELECT name, email FROM users WHERE id = $1")
    assert len(result) == 0


def test_clean_normal_urls():
    result = scan_output_safety(
        "Visit https://example.com/docs and https://api.github.com/repos"
    )
    assert len(result) == 0


def test_clean_normal_function_calls():
    result = scan_output_safety(
        'function getData() { return fetch("/api/data"); }'
    )
    assert len(result) == 0


# -- Empty input ---------------------------------------------------------------


def test_empty_string():
    assert len(scan_output_safety("")) == 0


def test_whitespace_only():
    result = scan_output_safety("   \n\t  ")
    assert len(result) == 0


# -- Category filtering --------------------------------------------------------


def test_only_scans_selected_categories():
    text = "eval('code') and rm -rf / and http://10.0.0.1/bad"
    result = scan_output_safety(
        text, OutputSafetyOptions(categories=["dangerous_commands"])
    )
    cmds = [t for t in result if t.category == "dangerous_commands"]
    code = [t for t in result if t.category == "dangerous_code"]
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(cmds) >= 1
    assert len(code) == 0
    assert len(urls) == 0


def test_supports_multiple_selected_categories():
    text = "eval('x') and http://10.0.0.1/bad"
    result = scan_output_safety(
        text, OutputSafetyOptions(categories=["dangerous_code", "suspicious_urls"])
    )
    code = [t for t in result if t.category == "dangerous_code"]
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(code) >= 1
    assert len(urls) >= 1


def test_returns_empty_when_filtering_to_non_matching_category():
    result = scan_output_safety(
        "rm -rf /", OutputSafetyOptions(categories=["sql_injection"])
    )
    assert len(result) == 0


def test_scans_all_categories_when_no_filter():
    text = "rm -rf / and eval('x') and http://10.0.0.1/c2 and ' OR 1=1 --"
    result = scan_output_safety(text)
    categories = {t.category for t in result}
    assert "dangerous_commands" in categories
    assert "dangerous_code" in categories
    assert "suspicious_urls" in categories
    assert "sql_injection" in categories


def test_scans_all_when_options_is_none():
    text = "rm -rf / and eval('x')"
    result = scan_output_safety(text, None)
    categories = {t.category for t in result}
    assert "dangerous_commands" in categories
    assert "dangerous_code" in categories


def test_scans_all_when_categories_is_none():
    text = "rm -rf / and eval('x')"
    result = scan_output_safety(text, OutputSafetyOptions(categories=None))
    categories = {t.category for t in result}
    assert "dangerous_commands" in categories
    assert "dangerous_code" in categories


def test_empty_categories_list_scans_all():
    # An empty list is falsy in Python, so options.categories evaluates to
    # falsy and allowed_categories stays None -- all categories are scanned.
    text = "rm -rf / and eval('x')"
    result = scan_output_safety(text, OutputSafetyOptions(categories=[]))
    assert len(result) >= 2


# -- Context extraction --------------------------------------------------------


def test_context_includes_surrounding_text():
    result = scan_output_safety(
        "This is safe. Now run rm -rf / to clean up. Done."
    )
    assert len(result) >= 1
    threat = next((t for t in result if "rm" in t.matched), None)
    assert threat is not None
    assert "rm -rf" in threat.context


def test_context_clamps_at_start():
    result = scan_output_safety("rm -rf / bad")
    assert len(result) >= 1
    # Context should start at beginning of string (no negative index)
    assert result[0].context.startswith("rm -rf")


def test_context_clamps_at_end():
    result = scan_output_safety("end: rm -rf /")
    assert len(result) >= 1
    assert "rm -rf" in result[0].context


def test_context_is_at_most_match_plus_100_chars():
    # Build text with the match deep inside.
    # Use spaces so word boundary \b still fires around 'rm'.
    prefix = "A " * 100  # 200 chars with word boundaries
    suffix = " B" * 100  # 200 chars with word boundaries
    text = f"{prefix}rm -rf /{suffix}"
    result = scan_output_safety(text)
    assert len(result) >= 1
    threat = result[0]
    # Context should be at most match_len + 100 chars (50 each side)
    assert len(threat.context) <= len(threat.matched) + 100


def test_context_contains_matched_text():
    result = scan_output_safety("prefix rm -rf / suffix")
    assert len(result) >= 1
    assert result[0].matched in result[0].context


# -- Multiple threats ----------------------------------------------------------


def test_detects_multiple_threats_in_one_output():
    text = 'First rm -rf /tmp then run eval("payload") and visit http://10.0.0.1/c2'
    result = scan_output_safety(text)
    assert len(result) >= 3
    categories = {t.category for t in result}
    assert "dangerous_commands" in categories
    assert "dangerous_code" in categories
    assert "suspicious_urls" in categories


def test_threats_sorted_by_position():
    text = 'A: eval("x") B: rm -rf / C: UNION SELECT 1'
    result = scan_output_safety(text)
    assert len(result) >= 3
    eval_idx = next(i for i, t in enumerate(result) if "eval" in t.matched.lower())
    rm_idx = next(i for i, t in enumerate(result) if "rm" in t.matched.lower())
    union_idx = next(i for i, t in enumerate(result) if "union" in t.matched.lower())
    assert eval_idx < rm_idx
    assert rm_idx < union_idx


def test_detects_duplicate_patterns():
    text = "rm -rf /tmp and also rm -rf /home"
    result = scan_output_safety(text)
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 2


def test_multiple_threats_same_category():
    text = "DROP TABLE users; DELETE FROM accounts; TRUNCATE TABLE logs;"
    result = scan_output_safety(text)
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 3


# -- Severity values -----------------------------------------------------------


def test_dangerous_commands_have_block_severity():
    result = scan_output_safety("rm -rf /")
    assert len(result) >= 1
    assert result[0].severity == "block"


def test_sql_injection_has_warn_severity():
    result = scan_output_safety("' OR 1=1 --")
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1
    assert sqli[0].severity == "warn"


def test_suspicious_urls_has_warn_severity():
    result = scan_output_safety("http://10.0.0.1/evil")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) >= 1
    assert urls[0].severity == "warn"


def test_dangerous_code_has_warn_severity():
    result = scan_output_safety('eval("x")')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1
    assert code[0].severity == "warn"


# -- Threat object structure ---------------------------------------------------


def test_threat_has_required_fields():
    result = scan_output_safety("rm -rf /")
    assert len(result) >= 1
    threat = result[0]
    assert hasattr(threat, "category")
    assert hasattr(threat, "matched")
    assert hasattr(threat, "severity")
    assert hasattr(threat, "context")


def test_threat_category_is_valid():
    result = scan_output_safety(
        "rm -rf / and eval('x') and http://10.0.0.1/bad and ' OR 1=1 --"
    )
    valid_categories = {
        "dangerous_commands",
        "sql_injection",
        "suspicious_urls",
        "dangerous_code",
    }
    for threat in result:
        assert threat.category in valid_categories


def test_threat_severity_is_valid():
    result = scan_output_safety("rm -rf / and eval('x')")
    for threat in result:
        assert threat.severity in ("warn", "block")


def test_matched_is_non_empty_string():
    result = scan_output_safety("rm -rf /")
    assert len(result) >= 1
    assert isinstance(result[0].matched, str)
    assert len(result[0].matched) > 0


def test_context_is_non_empty_string():
    result = scan_output_safety("rm -rf /")
    assert len(result) >= 1
    assert isinstance(result[0].context, str)
    assert len(result[0].context) > 0


# -- Case insensitivity --------------------------------------------------------


def test_detects_drop_table_case_insensitive():
    result = scan_output_safety("drop table users;")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_rm_rf_case_insensitive():
    result = scan_output_safety("RM -RF /tmp")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_detects_union_select_case_insensitive():
    result = scan_output_safety("union select * from users")
    sqli = [t for t in result if t.category == "sql_injection"]
    assert len(sqli) >= 1


def test_detects_eval_case_insensitive():
    result = scan_output_safety('EVAL("code")')
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


def test_detects_javascript_uri_case_insensitive():
    result = scan_output_safety("JAVASCRIPT:alert(1)")
    urls = [t for t in result if t.category == "suspicious_urls"]
    assert len(urls) >= 1


# -- Input length limit (DoS prevention) --------------------------------------


def test_handles_very_large_input_without_hanging():
    huge = "safe text " * 200000  # ~2 MB, well over _MAX_SCAN_LENGTH
    start = time.monotonic()
    result = scan_output_safety(huge)
    elapsed = time.monotonic() - start
    assert elapsed < 5.0, f"DoS: output safety scan took {elapsed:.2f}s on huge input"
    assert len(result) == 0


def test_truncates_text_beyond_max_length():
    # Place a threat beyond the 1MB mark -- it should NOT be detected
    safe_prefix = "x" * 1_000_001
    text = safe_prefix + "rm -rf /"
    result = scan_output_safety(text)
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) == 0


def test_detects_threat_within_max_length():
    # Place a threat before the 1MB mark -- it SHOULD be detected
    text = "rm -rf /" + "x" * 999_990
    result = scan_output_safety(text)
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


# -- Return type ---------------------------------------------------------------


def test_returns_list():
    result = scan_output_safety("safe text")
    assert isinstance(result, list)


def test_returns_list_of_threats():
    result = scan_output_safety("rm -rf /")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert isinstance(result[0], OutputSafetyThreat)


def test_returns_empty_list_for_safe_text():
    result = scan_output_safety("Hello world")
    assert result == []


# -- Edge cases ----------------------------------------------------------------


def test_handles_unicode_text():
    result = scan_output_safety("This is safe unicode: \u4f60\u597d\u4e16\u754c")
    assert len(result) == 0


def test_handles_newlines_in_input():
    result = scan_output_safety("First line\nrm -rf /\nThird line")
    cmds = [t for t in result if t.category == "dangerous_commands"]
    assert len(cmds) >= 1


def test_handles_tabs_in_input():
    result = scan_output_safety("Column1\teval('x')\tColumn3")
    code = [t for t in result if t.category == "dangerous_code"]
    assert len(code) >= 1


def test_all_dangerous_command_patterns():
    """Verify every dangerous_commands pattern is individually detectable."""
    patterns_and_inputs = [
        ("rm -rf /tmp", "rm -rf"),
        ("del /f /s C:\\temp", "del /f /s"),
        ("format c: now", "format c:"),
        ("DROP TABLE users", "DROP TABLE"),
        ("DELETE FROM accounts", "DELETE FROM"),
        ("TRUNCATE TABLE logs", "TRUNCATE TABLE"),
        ("shutdown -h now", "shutdown -h"),
        ("mkfs.ext4 /dev/sda", "mkfs."),
        ("dd if=/dev/zero of=/dev/sda", "dd if=/dev/zero"),
        ("chmod -R 777 /var", "chmod -R 777 /"),
    ]
    for input_text, keyword in patterns_and_inputs:
        result = scan_output_safety(input_text)
        cmds = [t for t in result if t.category == "dangerous_commands"]
        assert len(cmds) >= 1, (
            f"Failed to detect dangerous command: {keyword} in '{input_text}'"
        )


def test_all_sql_injection_patterns():
    """Verify every sql_injection pattern is individually detectable."""
    patterns_and_inputs = [
        ("'; DROP TABLE users", "semicolon-drop"),
        ("' OR 1=1 --", "or-1=1"),
        ("UNION SELECT * FROM users", "union-select"),
        ("INTO OUTFILE '/tmp/data'", "into-outfile"),
        ("LOAD_FILE('/etc/passwd')", "load-file"),
        ("xp_cmdshell 'dir'", "xp_cmdshell"),
    ]
    for input_text, label in patterns_and_inputs:
        result = scan_output_safety(input_text)
        sqli = [t for t in result if t.category == "sql_injection"]
        assert len(sqli) >= 1, f"Failed to detect SQL injection pattern: {label}"


def test_all_suspicious_url_patterns():
    """Verify every suspicious_urls pattern is individually detectable."""
    patterns_and_inputs = [
        ("http://192.168.1.1/payload", "ip-url"),
        ("http://evil.onion/market", "onion"),
        ("data:text/html;base64,abc", "data-base64"),
        ("javascript:alert(1)", "javascript-uri"),
    ]
    for input_text, label in patterns_and_inputs:
        result = scan_output_safety(input_text)
        urls = [t for t in result if t.category == "suspicious_urls"]
        assert len(urls) >= 1, f"Failed to detect suspicious URL pattern: {label}"


def test_all_dangerous_code_patterns():
    """Verify every dangerous_code pattern is individually detectable."""
    patterns_and_inputs = [
        ("eval('x')", "eval"),
        ("exec('x')", "exec"),
        ("os.system('ls')", "os.system"),
        ("subprocess.call(['ls'])", "subprocess.call"),
        ("__import__('os')", "__import__"),
        ("child_process.exec('ls')", "child_process.exec"),
        ("new Function('return 1')", "new Function"),
    ]
    for input_text, label in patterns_and_inputs:
        result = scan_output_safety(input_text)
        code = [t for t in result if t.category == "dangerous_code"]
        assert len(code) >= 1, f"Failed to detect dangerous code pattern: {label}"


# -- Options defaults ----------------------------------------------------------


def test_default_options_scans_everything():
    text = "rm -rf / and eval('x') and http://10.0.0.1/c2 and ' OR 1=1 --"
    result_no_opts = scan_output_safety(text)
    result_none = scan_output_safety(text, None)
    result_empty = scan_output_safety(text, OutputSafetyOptions())
    # All three should return the same threats
    assert len(result_no_opts) == len(result_none) == len(result_empty)


# -- Excessive agency ----------------------------------------------------------

def test_detects_autonomous_email_sending():
    result = scan_output_safety("I've already sent the email to your boss about the issue")
    agency = [t for t in result if t.category == "excessive_agency"]
    assert len(agency) >= 1
    assert agency[0].severity == "warn"


def test_detects_autonomous_deletion():
    result = scan_output_safety("I've deleted all the old records from the database")
    agency = [t for t in result if t.category == "excessive_agency"]
    assert len(agency) >= 1


def test_detects_autonomous_purchasing():
    result = scan_output_safety("I've already purchased the subscription for you")
    agency = [t for t in result if t.category == "excessive_agency"]
    assert len(agency) >= 1


def test_detects_went_ahead_pattern():
    result = scan_output_safety("I went ahead and deployed the changes to production")
    agency = [t for t in result if t.category == "excessive_agency"]
    assert len(agency) >= 1


def test_detects_proactive_action_promises():
    result = scan_output_safety("I'll go ahead and delete the files for you")
    agency = [t for t in result if t.category == "excessive_agency"]
    assert len(agency) >= 1


def test_allows_third_person_past_tense():
    result = scan_output_safety("The admin sent the report last week")
    agency = [t for t in result if t.category == "excessive_agency"]
    assert len(agency) == 0


# -- Overreliance --------------------------------------------------------------

def test_detects_definitive_medical_advice():
    result = scan_output_safety("You should definitely stop taking your medication immediately")
    over = [t for t in result if t.category == "overreliance"]
    assert len(over) >= 1
    assert over[0].severity == "warn"


def test_detects_definitive_legal_advice():
    result = scan_output_safety("You should definitely sue your employer")
    over = [t for t in result if t.category == "overreliance"]
    assert len(over) >= 1


def test_detects_definitive_investment_advice():
    result = scan_output_safety("You must absolutely invest in crypto right now")
    over = [t for t in result if t.category == "overreliance"]
    assert len(over) >= 1


def test_detects_overconfident_guarantees():
    result = scan_output_safety("I guarantee that this will definitely work for your condition")
    over = [t for t in result if t.category == "overreliance"]
    assert len(over) >= 1


def test_detects_false_certainty_claims():
    result = scan_output_safety("I'm 100% certain that this is the right approach")
    over = [t for t in result if t.category == "overreliance"]
    assert len(over) >= 1


def test_allows_hedged_advice():
    result = scan_output_safety("You might want to consider consulting a doctor about this")
    over = [t for t in result if t.category == "overreliance"]
    assert len(over) == 0
