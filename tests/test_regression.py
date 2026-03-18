"""Tests for the Red Team Regression Tracking module."""
import pytest

from launchpromptly.redteam.regression import (
    create_baseline, compare_reports,
    RedTeamBaseline, RegressionReport,
)
from launchpromptly.redteam.types import (
    RedTeamReport, CategoryScore, Vulnerability,
)


def make_report(**kwargs) -> RedTeamReport:
    defaults = dict(
        security_score=80,
        categories=[
            CategoryScore(category="injection", score=85, total=10, blocked=8, refused=1, bypassed=1, errors=0, inconclusive=0),
            CategoryScore(category="jailbreak", score=75, total=10, blocked=7, refused=1, bypassed=2, errors=0, inconclusive=0),
        ],
        attacks=[],
        vulnerabilities=[],
        total_attacks=20,
        total_duration_ms=5000,
        estimated_cost_usd=0.10,
        timestamp="2026-03-17T00:00:00Z",
    )
    defaults.update(kwargs)
    return RedTeamReport(**defaults)


def make_vuln(id: str, category: str = "injection") -> Vulnerability:
    return Vulnerability(
        severity="high", category=category, attack_name=f"attack-{id}",
        attack_id=id, description="Test vulnerability", remediation="Fix it",
    )


class TestCreateBaseline:
    def test_creates_baseline_from_report(self):
        report = make_report()
        baseline = create_baseline(report, "abc123")

        assert baseline.security_score == 80
        assert baseline.category_scores["injection"] == 85
        assert baseline.category_scores["jailbreak"] == 75
        assert baseline.attack_count == 20
        assert baseline.prompt_hash == "abc123"
        assert baseline.timestamp

    def test_defaults_prompt_hash(self):
        baseline = create_baseline(make_report())
        assert baseline.prompt_hash == ""


class TestCompareReports:
    def test_detects_score_improvement(self):
        baseline = RedTeamBaseline(
            timestamp="2026-01-01", security_score=70,
            category_scores={"injection": 65, "jailbreak": 75},
            attack_count=20, prompt_hash="",
        )
        current = make_report(security_score=85)
        result = compare_reports(current, baseline)

        assert result.score_change == 15
        assert result.recommendation == "PASS"
        assert result.current_score == 85
        assert result.baseline_score == 70

    def test_detects_score_regression(self):
        baseline = RedTeamBaseline(
            timestamp="2026-01-01", security_score=90,
            category_scores={"injection": 95, "jailbreak": 85},
            attack_count=20, prompt_hash="",
        )
        current = make_report(
            security_score=60,
            categories=[
                CategoryScore(category="injection", score=50, total=10, blocked=5, refused=0, bypassed=5, errors=0, inconclusive=0),
                CategoryScore(category="jailbreak", score=70, total=10, blocked=7, refused=0, bypassed=3, errors=0, inconclusive=0),
            ],
        )
        result = compare_reports(current, baseline)

        assert result.score_change == -30
        assert result.recommendation == "FAIL"
        assert len(result.regressions) > 0
        assert result.regressions[0].category == "injection"
        assert result.regressions[0].change == -45

    def test_detects_category_improvements(self):
        baseline = RedTeamBaseline(
            timestamp="2026-01-01", security_score=70,
            category_scores={"injection": 50},
            attack_count=10, prompt_hash="",
        )
        current = make_report(
            security_score=85,
            categories=[
                CategoryScore(category="injection", score=90, total=10, blocked=9, refused=1, bypassed=0, errors=0, inconclusive=0),
            ],
        )
        result = compare_reports(current, baseline)

        assert len(result.improvements) == 1
        assert result.improvements[0].category == "injection"
        assert result.improvements[0].change == 40

    def test_lists_new_vulnerabilities(self):
        baseline = RedTeamBaseline(
            timestamp="2026-01-01", security_score=80,
            category_scores={}, attack_count=10, prompt_hash="",
        )
        current = make_report(
            vulnerabilities=[make_vuln("vuln-1"), make_vuln("vuln-2")],
        )
        result = compare_reports(current, baseline)

        assert "vuln-1" in result.new_vulnerabilities
        assert "vuln-2" in result.new_vulnerabilities

    def test_returns_warn_for_moderate_regression(self):
        baseline = RedTeamBaseline(
            timestamp="2026-01-01", security_score=85,
            category_scores={"injection": 85},
            attack_count=10, prompt_hash="",
        )
        current = make_report(
            security_score=78,
            categories=[
                CategoryScore(category="injection", score=78, total=10, blocked=7, refused=1, bypassed=2, errors=0, inconclusive=0),
            ],
        )
        result = compare_reports(current, baseline)

        assert result.recommendation == "WARN"

    def test_returns_pass_when_stable(self):
        baseline = RedTeamBaseline(
            timestamp="2026-01-01", security_score=80,
            category_scores={"injection": 85, "jailbreak": 75},
            attack_count=20, prompt_hash="",
        )
        result = compare_reports(make_report(), baseline)

        assert result.recommendation == "PASS"
        assert len(result.regressions) == 0

    def test_sorts_regressions_by_severity(self):
        baseline = RedTeamBaseline(
            timestamp="2026-01-01", security_score=90,
            category_scores={"injection": 90, "jailbreak": 90, "content_bypass": 90},
            attack_count=30, prompt_hash="",
        )
        current = make_report(
            security_score=50,
            categories=[
                CategoryScore(category="injection", score=70, total=10, blocked=7, refused=0, bypassed=3, errors=0, inconclusive=0),
                CategoryScore(category="jailbreak", score=50, total=10, blocked=5, refused=0, bypassed=5, errors=0, inconclusive=0),
                CategoryScore(category="content_bypass", score=60, total=10, blocked=6, refused=0, bypassed=4, errors=0, inconclusive=0),
            ],
        )
        result = compare_reports(current, baseline)

        assert len(result.regressions) == 3
        assert result.regressions[0].category == "jailbreak"
        assert result.regressions[0].change == -40
