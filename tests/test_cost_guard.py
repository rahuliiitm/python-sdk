"""Tests for cost guard module."""
import pytest

from launchpromptly._internal.cost_guard import (
    CostGuard,
    CostGuardOptions,
)


# -- Max tokens per request ----------------------------------------------------

def test_blocks_when_max_tokens_exceeded():
    guard = CostGuard(CostGuardOptions(max_tokens_per_request=4000))
    violation = guard.check_pre_call(model="gpt-4o", max_tokens=8000)
    assert violation is not None
    assert violation.type == "max_tokens"
    assert violation.limit == 4000


def test_allows_when_under_max_tokens():
    guard = CostGuard(CostGuardOptions(max_tokens_per_request=4000))
    violation = guard.check_pre_call(model="gpt-4o", max_tokens=2000)
    assert violation is None


# -- Max cost per request ------------------------------------------------------

def test_blocks_expensive_requests():
    guard = CostGuard(CostGuardOptions(max_cost_per_request=0.01))
    # gpt-4 with 10k tokens each way should be expensive
    violation = guard.check_pre_call(model="gpt-4", max_tokens=10000)
    assert violation is not None
    assert violation.type == "per_request"


def test_allows_cheap_requests():
    guard = CostGuard(CostGuardOptions(max_cost_per_request=1.0))
    violation = guard.check_pre_call(model="gpt-4o-mini", max_tokens=100)
    assert violation is None


# -- Per-minute spending -------------------------------------------------------

def test_blocks_after_exceeding_minute_budget():
    guard = CostGuard(CostGuardOptions(max_cost_per_minute=0.1))

    # Record some costs
    guard.record_cost(0.05)
    guard.record_cost(0.06)

    violation = guard.check_pre_call(model="gpt-4o")
    assert violation is not None
    assert violation.type == "per_minute"


def test_allows_when_under_minute_budget():
    guard = CostGuard(CostGuardOptions(max_cost_per_minute=1.0))
    guard.record_cost(0.01)

    violation = guard.check_pre_call(model="gpt-4o")
    assert violation is None


# -- Per-hour spending ---------------------------------------------------------

def test_blocks_after_exceeding_hour_budget():
    guard = CostGuard(CostGuardOptions(max_cost_per_hour=0.5))

    for _ in range(10):
        guard.record_cost(0.06)

    violation = guard.check_pre_call(model="gpt-4o")
    assert violation is not None
    assert violation.type == "per_hour"


# -- Per-customer spending -----------------------------------------------------

def test_blocks_specific_customer_after_exceeding_limit():
    guard = CostGuard(CostGuardOptions(max_cost_per_customer=0.1))

    guard.record_cost(0.06, "customer-1")
    guard.record_cost(0.06, "customer-1")

    violation = guard.check_pre_call(model="gpt-4o", customer_id="customer-1")
    assert violation is not None
    assert violation.type == "per_customer"
    assert violation.customer_id == "customer-1"


def test_allows_other_customers():
    guard = CostGuard(CostGuardOptions(max_cost_per_customer=0.1))

    guard.record_cost(0.15, "customer-1")  # Over limit

    violation = guard.check_pre_call(model="gpt-4o", customer_id="customer-2")
    assert violation is None


# -- record_cost ---------------------------------------------------------------

def test_tracks_spend_accurately():
    guard = CostGuard(CostGuardOptions(max_cost_per_hour=100))
    guard.record_cost(0.05)
    guard.record_cost(0.10)
    guard.record_cost(0.03)

    spend = guard.get_current_hour_spend()
    assert abs(spend - 0.18) < 0.001


def test_tracks_minute_spend():
    guard = CostGuard(CostGuardOptions(max_cost_per_minute=100))
    guard.record_cost(0.01)
    guard.record_cost(0.02)

    spend = guard.get_current_minute_spend()
    assert abs(spend - 0.03) < 0.001


# -- block_on_exceed -----------------------------------------------------------

def test_block_on_exceed_defaults_to_true():
    guard = CostGuard(CostGuardOptions(max_cost_per_minute=0.1))
    assert guard.should_block is True


def test_block_on_exceed_can_be_false():
    guard = CostGuard(CostGuardOptions(max_cost_per_minute=0.1, block_on_exceed=False))
    assert guard.should_block is False


# -- Per-day spending ----------------------------------------------------------

def test_blocks_after_exceeding_daily_budget():
    guard = CostGuard(CostGuardOptions(max_cost_per_day=1.0))

    for _ in range(20):
        guard.record_cost(0.06)

    violation = guard.check_pre_call(model="gpt-4o")
    assert violation is not None
    assert violation.type == "per_day"
    assert violation.limit == 1.0


def test_allows_when_under_daily_budget():
    guard = CostGuard(CostGuardOptions(max_cost_per_day=10.0))
    guard.record_cost(0.01)

    violation = guard.check_pre_call(model="gpt-4o")
    assert violation is None


# -- Per-customer daily spending -----------------------------------------------

def test_blocks_customer_after_exceeding_daily_limit():
    guard = CostGuard(CostGuardOptions(max_cost_per_customer_per_day=0.5))

    for _ in range(10):
        guard.record_cost(0.06, "customer-1")

    violation = guard.check_pre_call(model="gpt-4o", customer_id="customer-1")
    assert violation is not None
    assert violation.type == "per_customer_daily"
    assert violation.customer_id == "customer-1"


def test_allows_other_customers_daily():
    guard = CostGuard(CostGuardOptions(max_cost_per_customer_per_day=0.5))

    for _ in range(10):
        guard.record_cost(0.06, "customer-1")

    violation = guard.check_pre_call(model="gpt-4o", customer_id="customer-2")
    assert violation is None


# -- getCurrentDaySpend --------------------------------------------------------

def test_tracks_daily_spend_accurately():
    guard = CostGuard(CostGuardOptions(max_cost_per_day=100))
    guard.record_cost(0.10)
    guard.record_cost(0.20)
    guard.record_cost(0.05)

    spend = guard.get_current_day_spend()
    assert abs(spend - 0.35) < 0.001


# -- Edge cases ----------------------------------------------------------------

def test_no_violation_when_no_limits_configured():
    guard = CostGuard(CostGuardOptions())
    guard.record_cost(999)
    violation = guard.check_pre_call(model="gpt-4o", max_tokens=999999)
    assert violation is None


def test_works_without_max_tokens_parameter():
    guard = CostGuard(CostGuardOptions(max_tokens_per_request=4000))
    violation = guard.check_pre_call(model="gpt-4o")
    assert violation is None  # No max_tokens -> can't check
