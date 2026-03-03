"""
Cost guard module -- in-memory sliding window rate limiting for LLM spend.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, List, Literal, Optional

from .cost import calculate_event_cost

BudgetViolationType = Literal[
    "per_request",
    "per_minute",
    "per_hour",
    "per_day",
    "per_customer",
    "per_customer_daily",
    "max_tokens",
]


@dataclass
class CostGuardOptions:
    max_cost_per_request: Optional[float] = None
    max_cost_per_minute: Optional[float] = None
    max_cost_per_hour: Optional[float] = None
    max_cost_per_day: Optional[float] = None
    max_cost_per_customer: Optional[float] = None
    max_cost_per_customer_per_day: Optional[float] = None
    max_tokens_per_request: Optional[int] = None
    on_budget_exceeded: Optional[Callable[[BudgetViolation], None]] = None
    block_on_exceed: bool = True


@dataclass
class BudgetViolation:
    type: BudgetViolationType
    current_spend: float
    limit: float
    customer_id: Optional[str] = None


@dataclass
class _CostEntry:
    cost_usd: float
    timestamp_ms: float
    customer_id: Optional[str] = None


class CostGuard:
    """In-memory cost guard with sliding window tracking.

    Resets on SDK restart (no persistence).
    """

    def __init__(self, options: CostGuardOptions) -> None:
        self._options = options
        self._entries: List[_CostEntry] = []
        self._block_on_exceed = options.block_on_exceed

    def check_pre_call(
        self,
        *,
        model: str,
        max_tokens: Optional[int] = None,
        customer_id: Optional[str] = None,
    ) -> Optional[BudgetViolation]:
        """Pre-call check: estimate cost from model + max_tokens and check budgets.

        Returns a BudgetViolation if any limit is exceeded, None otherwise.
        """
        now = time.time() * 1000  # ms

        # Check max tokens per request
        if (
            self._options.max_tokens_per_request is not None
            and max_tokens is not None
            and max_tokens > self._options.max_tokens_per_request
        ):
            return BudgetViolation(
                type="max_tokens",
                current_spend=float(max_tokens),
                limit=float(self._options.max_tokens_per_request),
                customer_id=customer_id,
            )

        # Estimate cost (assume worst case: maxTokens for both input and output)
        if self._options.max_cost_per_request is not None and max_tokens is not None:
            estimated_cost = calculate_event_cost(
                "openai",
                model,
                max_tokens,
                max_tokens,
            )
            if estimated_cost > self._options.max_cost_per_request:
                return BudgetViolation(
                    type="per_request",
                    current_spend=estimated_cost,
                    limit=self._options.max_cost_per_request,
                    customer_id=customer_id,
                )

        # Check per-minute spend
        if self._options.max_cost_per_minute is not None:
            minute_spend = self.get_spend_in_window(now - 60_000, now)
            if minute_spend >= self._options.max_cost_per_minute:
                return BudgetViolation(
                    type="per_minute",
                    current_spend=minute_spend,
                    limit=self._options.max_cost_per_minute,
                    customer_id=customer_id,
                )

        # Check per-hour spend
        if self._options.max_cost_per_hour is not None:
            hour_spend = self.get_spend_in_window(now - 3_600_000, now)
            if hour_spend >= self._options.max_cost_per_hour:
                return BudgetViolation(
                    type="per_hour",
                    current_spend=hour_spend,
                    limit=self._options.max_cost_per_hour,
                    customer_id=customer_id,
                )

        # Check per-day spend (24h rolling window)
        if self._options.max_cost_per_day is not None:
            day_spend = self.get_spend_in_window(now - 86_400_000, now)
            if day_spend >= self._options.max_cost_per_day:
                return BudgetViolation(
                    type="per_day",
                    current_spend=day_spend,
                    limit=self._options.max_cost_per_day,
                    customer_id=customer_id,
                )

        # Check per-customer spend (per hour)
        if self._options.max_cost_per_customer is not None and customer_id is not None:
            customer_spend = self.get_spend_in_window(
                now - 3_600_000,
                now,
                customer_id=customer_id,
            )
            if customer_spend >= self._options.max_cost_per_customer:
                return BudgetViolation(
                    type="per_customer",
                    current_spend=customer_spend,
                    limit=self._options.max_cost_per_customer,
                    customer_id=customer_id,
                )

        # Check per-customer daily spend (24h rolling window)
        if self._options.max_cost_per_customer_per_day is not None and customer_id is not None:
            customer_day_spend = self.get_spend_in_window(
                now - 86_400_000,
                now,
                customer_id=customer_id,
            )
            if customer_day_spend >= self._options.max_cost_per_customer_per_day:
                return BudgetViolation(
                    type="per_customer_daily",
                    current_spend=customer_day_spend,
                    limit=self._options.max_cost_per_customer_per_day,
                    customer_id=customer_id,
                )

        return None

    def record_cost(self, cost_usd: float, customer_id: Optional[str] = None) -> None:
        """Post-call: record actual cost from API response."""
        self._entries.append(
            _CostEntry(
                cost_usd=cost_usd,
                timestamp_ms=time.time() * 1000,
                customer_id=customer_id,
            )
        )

        # Prune entries older than 1 hour to prevent memory growth
        self._prune_old_entries()

    def get_spend_in_window(
        self,
        from_ms: float,
        to_ms: float,
        customer_id: Optional[str] = None,
    ) -> float:
        """Get total spend in a time window, optionally filtered by customer."""
        total = 0.0
        for entry in self._entries:
            if entry.timestamp_ms < from_ms or entry.timestamp_ms > to_ms:
                continue
            if customer_id is not None and entry.customer_id != customer_id:
                continue
            total += entry.cost_usd
        return total

    def get_current_minute_spend(self) -> float:
        """Get current minute spend."""
        now = time.time() * 1000
        return self.get_spend_in_window(now - 60_000, now)

    def get_current_hour_spend(self) -> float:
        """Get current hour spend."""
        now = time.time() * 1000
        return self.get_spend_in_window(now - 3_600_000, now)

    def get_current_day_spend(self) -> float:
        """Get current day spend (24h rolling window)."""
        now = time.time() * 1000
        return self.get_spend_in_window(now - 86_400_000, now)

    @property
    def should_block(self) -> bool:
        """Whether to block on budget exceeded."""
        return self._block_on_exceed

    @property
    def config(self) -> CostGuardOptions:
        """The configured options."""
        return self._options

    def _prune_old_entries(self) -> None:
        cutoff = time.time() * 1000 - 86_400_000  # 24 hours
        while self._entries and self._entries[0].timestamp_ms < cutoff:
            self._entries.pop(0)
