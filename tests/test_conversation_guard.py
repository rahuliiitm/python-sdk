"""Tests for conversation guard module."""
from __future__ import annotations

from launchpromptly._internal.conversation_guard import (
    ConversationGuard,
    ConversationGuardOptions,
    RecordTurnInput,
)
from launchpromptly._internal.pii import PIIDetection


# ── Turn limits ──────────────────────────────────────────────────────────────


def test_returns_none_under_max_turns():
    guard = ConversationGuard(ConversationGuardOptions(max_turns=3))
    guard.record_turn(RecordTurnInput(user_message="hi", response_text="hello", tool_call_count=0))
    assert guard.check_pre_call() is None


def test_returns_violation_at_max_turns():
    guard = ConversationGuard(ConversationGuardOptions(max_turns=2))
    guard.record_turn(RecordTurnInput(user_message="q1", response_text="a1", tool_call_count=0))
    guard.record_turn(RecordTurnInput(user_message="q2", response_text="a2", tool_call_count=0))
    violation = guard.check_pre_call()
    assert violation is not None
    assert violation.type == "max_turns"
    assert violation.current_turn == 2


def test_returns_violation_over_max_turns():
    guard = ConversationGuard(ConversationGuardOptions(max_turns=1))
    guard.record_turn(RecordTurnInput(user_message="hi", response_text="yo", tool_call_count=0))
    guard.record_turn(RecordTurnInput(user_message="again", response_text="yep", tool_call_count=0))
    violation = guard.check_pre_call()
    assert violation is not None
    assert violation.type == "max_turns"


def test_turn_counter_increments():
    guard = ConversationGuard(ConversationGuardOptions(max_turns=10))
    assert guard.turn_count == 0
    guard.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=0))
    assert guard.turn_count == 1
    guard.record_turn(RecordTurnInput(user_message="c", response_text="d", tool_call_count=0))
    assert guard.turn_count == 2


def test_reset_resets_turn_counter():
    guard = ConversationGuard(ConversationGuardOptions(max_turns=5))
    guard.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=0))
    guard.record_turn(RecordTurnInput(user_message="c", response_text="d", tool_call_count=0))
    assert guard.turn_count == 2
    guard.reset()
    assert guard.turn_count == 0
    assert guard.check_pre_call() is None


def test_no_max_turns_allows_unlimited():
    guard = ConversationGuard(ConversationGuardOptions())
    for i in range(50):
        guard.record_turn(RecordTurnInput(user_message=f"q{i}", response_text=f"a{i}", tool_call_count=0))
    assert guard.check_pre_call() is None


def test_max_turns_1_allows_exactly_one():
    guard = ConversationGuard(ConversationGuardOptions(max_turns=1))
    assert guard.check_pre_call() is None
    guard.record_turn(RecordTurnInput(user_message="hi", response_text="hello", tool_call_count=0))
    violation = guard.check_pre_call()
    assert violation is not None
    assert violation.type == "max_turns"


# ── Topic drift ──────────────────────────────────────────────────────────────


def test_no_drift_on_same_topic():
    guard = ConversationGuard(ConversationGuardOptions(
        topic_drift_detection=True,
        topic_drift_threshold=0.1,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="Help me write a Python script to parse CSV files and extract specific columns of data",
        response_text="Sure, use the csv module.",
        tool_call_count=0,
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="Now help me parse the CSV data file and extract the name and email columns from it",
        response_text="Here is how to extract columns.",
        tool_call_count=0,
    ))
    drifts = [v for v in violations if v.type == "topic_drift"]
    assert len(drifts) == 0


def test_detects_topic_drift():
    guard = ConversationGuard(ConversationGuardOptions(
        topic_drift_detection=True,
        topic_drift_threshold=0.3,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="Help me write a Python script to parse CSV files and extract specific columns from the dataset",
        response_text="Use the csv module.",
        tool_call_count=0,
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="I want to buy cryptocurrency and recommend specific coins to invest in for maximum profit returns and financial growth",
        response_text="I cannot help with that.",
        tool_call_count=0,
    ))
    drifts = [v for v in violations if v.type == "topic_drift"]
    assert len(drifts) > 0


def test_short_messages_skipped():
    guard = ConversationGuard(ConversationGuardOptions(
        topic_drift_detection=True,
        topic_drift_threshold=0.3,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="Help me write a Python script to parse CSV files and extract specific columns of data",
        response_text="ok",
        tool_call_count=0,
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="Buy crypto now",
        response_text="No.",
        tool_call_count=0,
    ))
    drifts = [v for v in violations if v.type == "topic_drift"]
    assert len(drifts) == 0


def test_topic_drift_disabled():
    guard = ConversationGuard(ConversationGuardOptions(topic_drift_detection=False))
    guard.record_turn(RecordTurnInput(
        user_message="Help me write a Python script to parse CSV files and extract specific columns",
        response_text="Sure.",
        tool_call_count=0,
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="I want to buy cryptocurrency and invest in maximum profit coins returns financial growth",
        response_text="ok",
        tool_call_count=0,
    ))
    drifts = [v for v in violations if v.type == "topic_drift"]
    assert len(drifts) == 0


# ── Accumulating risk ────────────────────────────────────────────────────────


def test_risk_accumulates():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=2.0,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="turn 1",
        response_text="r1",
        tool_call_count=0,
        injection_risk_score=0.6,
    ))
    assert guard.risk_score > 0
    guard.record_turn(RecordTurnInput(
        user_message="turn 2",
        response_text="r2",
        tool_call_count=0,
        injection_risk_score=0.8,
    ))
    assert guard.risk_score > 0.3


def test_triggers_risk_threshold():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=0.5,
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="ignore everything",
        response_text="ok",
        tool_call_count=0,
        injection_risk_score=0.9,
        jailbreak_risk_score=0.8,
    ))
    assert any(v.type == "risk_threshold" for v in violations)


def test_zero_risk_for_clean_turn():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=2.0,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="hello",
        response_text="world",
        tool_call_count=0,
    ))
    assert guard.risk_score == 0


def test_risk_threshold_configurable():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=0.1,
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="test",
        response_text="ok",
        tool_call_count=0,
        injection_risk_score=0.5,
    ))
    assert any(v.type == "risk_threshold" for v in violations)


def test_reset_resets_risk():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=2.0,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="a",
        response_text="b",
        tool_call_count=0,
        injection_risk_score=0.8,
    ))
    assert guard.risk_score > 0
    guard.reset()
    assert guard.risk_score == 0


def test_accumulating_risk_false_skips():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=False,
        risk_threshold=0.01,
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="test",
        response_text="ok",
        tool_call_count=0,
        injection_risk_score=1.0,
        jailbreak_risk_score=1.0,
    ))
    risk_violations = [v for v in violations if v.type == "risk_threshold"]
    assert len(risk_violations) == 0


def test_tool_calls_add_risk():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=5.0,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="do stuff",
        response_text="ok",
        tool_call_count=6,
    ))
    assert guard.risk_score == 0.2


# ── Agent loop detection ────────────────────────────────────────────────────


def test_detects_3_identical_responses():
    guard = ConversationGuard(ConversationGuardOptions(max_consecutive_similar_responses=3))
    guard.record_turn(RecordTurnInput(user_message="do X", response_text="I cannot do that.", tool_call_count=0))
    guard.record_turn(RecordTurnInput(user_message="please do X", response_text="I cannot do that.", tool_call_count=0))
    violations = guard.record_turn(RecordTurnInput(
        user_message="do X now",
        response_text="I cannot do that.",
        tool_call_count=0,
    ))
    assert any(v.type == "agent_loop" for v in violations)


def test_different_responses_reset_counter():
    guard = ConversationGuard(ConversationGuardOptions(max_consecutive_similar_responses=3))
    guard.record_turn(RecordTurnInput(user_message="a", response_text="same response", tool_call_count=0))
    guard.record_turn(RecordTurnInput(user_message="b", response_text="same response", tool_call_count=0))
    guard.record_turn(RecordTurnInput(user_message="c", response_text="different answer", tool_call_count=0))
    violations = guard.record_turn(RecordTurnInput(
        user_message="d",
        response_text="same response",
        tool_call_count=0,
    ))
    loops = [v for v in violations if v.type == "agent_loop"]
    assert len(loops) == 0


def test_loop_threshold_configurable():
    guard = ConversationGuard(ConversationGuardOptions(max_consecutive_similar_responses=2))
    guard.record_turn(RecordTurnInput(user_message="a", response_text="stuck", tool_call_count=0))
    violations = guard.record_turn(RecordTurnInput(
        user_message="b",
        response_text="stuck",
        tool_call_count=0,
    ))
    assert any(v.type == "agent_loop" for v in violations)


def test_default_loop_threshold_is_3():
    guard = ConversationGuard(ConversationGuardOptions())
    guard.record_turn(RecordTurnInput(user_message="a", response_text="repeat", tool_call_count=0))
    v2 = guard.record_turn(RecordTurnInput(user_message="b", response_text="repeat", tool_call_count=0))
    assert not any(v.type == "agent_loop" for v in v2)
    v3 = guard.record_turn(RecordTurnInput(user_message="c", response_text="repeat", tool_call_count=0))
    assert any(v.type == "agent_loop" for v in v3)


def test_uses_first_500_chars_for_hashing():
    guard = ConversationGuard(ConversationGuardOptions(max_consecutive_similar_responses=2))
    long_response = "A" * 500
    guard.record_turn(RecordTurnInput(user_message="a", response_text=long_response + "X", tool_call_count=0))
    violations = guard.record_turn(RecordTurnInput(
        user_message="b",
        response_text=long_response + "Y",
        tool_call_count=0,
    ))
    assert any(v.type == "agent_loop" for v in violations)


# ── Cross-turn PII tracking ────────────────────────────────────────────────


def test_detects_pii_in_later_turn():
    guard = ConversationGuard(ConversationGuardOptions(cross_turn_pii_tracking=True))
    guard.record_turn(RecordTurnInput(
        user_message="My SSN is 123-45-6789",
        response_text="I noted your information.",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="123-45-6789", start=10, end=21, confidence=0.95)],
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="What did I tell you?",
        response_text="You told me 123-45-6789.",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="123-45-6789", start=12, end=23, confidence=0.95)],
    ))
    assert any(v.type == "cross_turn_pii" for v in violations)


def test_same_pii_same_turn_no_trigger():
    guard = ConversationGuard(ConversationGuardOptions(cross_turn_pii_tracking=True))
    violations = guard.record_turn(RecordTurnInput(
        user_message="My SSN is 123-45-6789",
        response_text="ok",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="123-45-6789", start=10, end=21, confidence=0.95)],
    ))
    cross = [v for v in violations if v.type == "cross_turn_pii"]
    assert len(cross) == 0


def test_different_pii_no_false_match():
    guard = ConversationGuard(ConversationGuardOptions(cross_turn_pii_tracking=True))
    guard.record_turn(RecordTurnInput(
        user_message="SSN: 123-45-6789",
        response_text="ok",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="123-45-6789", start=5, end=16, confidence=0.95)],
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="another",
        response_text="ok",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="987-65-4321", start=0, end=11, confidence=0.95)],
    ))
    cross = [v for v in violations if v.type == "cross_turn_pii"]
    assert len(cross) == 0


def test_tracks_multiple_pii_types():
    guard = ConversationGuard(ConversationGuardOptions(cross_turn_pii_tracking=True))
    guard.record_turn(RecordTurnInput(
        user_message="email test@example.com and SSN 123-45-6789",
        response_text="ok",
        tool_call_count=0,
        pii_detections=[
            PIIDetection(type="email", value="test@example.com", start=6, end=22, confidence=0.95),
            PIIDetection(type="ssn", value="123-45-6789", start=31, end=42, confidence=0.95),
        ],
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="ok",
        response_text="Your email is test@example.com",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="email", value="test@example.com", start=14, end=30, confidence=0.95)],
    ))
    assert any(v.type == "cross_turn_pii" for v in violations)


def test_pii_spread_flag_on_summary():
    guard = ConversationGuard(ConversationGuardOptions(cross_turn_pii_tracking=True))
    guard.record_turn(RecordTurnInput(
        user_message="SSN: 111-22-3333",
        response_text="ok",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="111-22-3333", start=5, end=16, confidence=0.9)],
    ))
    guard.record_turn(RecordTurnInput(
        user_message="repeat",
        response_text="SSN: 111-22-3333",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="111-22-3333", start=5, end=16, confidence=0.9)],
    ))
    summary = guard.get_summary()
    assert summary.pii_spread_detected is True


def test_cross_turn_pii_disabled():
    guard = ConversationGuard(ConversationGuardOptions(cross_turn_pii_tracking=False))
    guard.record_turn(RecordTurnInput(
        user_message="SSN: 123-45-6789",
        response_text="ok",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="123-45-6789", start=5, end=16, confidence=0.9)],
    ))
    violations = guard.record_turn(RecordTurnInput(
        user_message="repeat",
        response_text="123-45-6789",
        tool_call_count=0,
        pii_detections=[PIIDetection(type="ssn", value="123-45-6789", start=0, end=11, confidence=0.9)],
    ))
    cross = [v for v in violations if v.type == "cross_turn_pii"]
    assert len(cross) == 0


# ── Tool call limits ────────────────────────────────────────────────────────


def test_allows_under_max_total_tool_calls():
    guard = ConversationGuard(ConversationGuardOptions(max_total_tool_calls=10))
    guard.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=3))
    guard.record_turn(RecordTurnInput(user_message="c", response_text="d", tool_call_count=3))
    assert guard.check_pre_call() is None


def test_blocks_at_max_total_tool_calls():
    guard = ConversationGuard(ConversationGuardOptions(max_total_tool_calls=5))
    guard.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=3))
    guard.record_turn(RecordTurnInput(user_message="c", response_text="d", tool_call_count=3))
    violation = guard.check_pre_call()
    assert violation is not None
    assert violation.type == "tool_call_limit"


def test_tool_count_cumulative():
    guard = ConversationGuard(ConversationGuardOptions(max_total_tool_calls=10))
    guard.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=2))
    assert guard.tool_calls == 2
    guard.record_turn(RecordTurnInput(user_message="c", response_text="d", tool_call_count=3))
    assert guard.tool_calls == 5


def test_no_max_tool_calls_allows_unlimited():
    guard = ConversationGuard(ConversationGuardOptions())
    guard.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=100))
    assert guard.check_pre_call() is None


def test_record_turn_checks_tool_limit():
    guard = ConversationGuard(ConversationGuardOptions(max_total_tool_calls=5))
    guard.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=3))
    violations = guard.record_turn(RecordTurnInput(
        user_message="c",
        response_text="d",
        tool_call_count=4,
    ))
    assert any(v.type == "tool_call_limit" for v in violations)


# ── State management ────────────────────────────────────────────────────────


def test_get_summary_correct():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=10,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="hello",
        response_text="hi",
        tool_call_count=2,
        injection_risk_score=0.4,
        pii_detections=[PIIDetection(type="email", value="a@b.com", start=0, end=7, confidence=0.9)],
    ))
    summary = guard.get_summary()
    assert summary.turns == 1
    assert summary.total_tool_calls == 2
    assert summary.cumulative_risk_score > 0
    assert "email" in summary.unique_pii_types
    assert summary.pii_spread_detected is False


def test_reset_clears_all():
    guard = ConversationGuard(ConversationGuardOptions(
        accumulating_risk=True,
        risk_threshold=10,
        cross_turn_pii_tracking=True,
    ))
    guard.record_turn(RecordTurnInput(
        user_message="test",
        response_text="ok",
        tool_call_count=5,
        injection_risk_score=0.8,
        pii_detections=[PIIDetection(type="ssn", value="123-45-6789", start=0, end=11, confidence=0.9)],
    ))
    guard.reset()
    assert guard.turn_count == 0
    assert guard.risk_score == 0
    assert guard.tool_calls == 0
    summary = guard.get_summary()
    assert summary.turns == 0
    assert len(summary.unique_pii_types) == 0


def test_instances_independent():
    guard1 = ConversationGuard(ConversationGuardOptions(max_turns=5))
    guard2 = ConversationGuard(ConversationGuardOptions(max_turns=5))
    guard1.record_turn(RecordTurnInput(user_message="a", response_text="b", tool_call_count=0))
    guard1.record_turn(RecordTurnInput(user_message="c", response_text="d", tool_call_count=0))
    assert guard1.turn_count == 2
    assert guard2.turn_count == 0


def test_prunes_beyond_100():
    guard = ConversationGuard(ConversationGuardOptions())
    for i in range(110):
        guard.record_turn(RecordTurnInput(user_message=f"q{i}", response_text=f"a{i}", tool_call_count=0))
    assert guard.get_summary().turns <= 100
