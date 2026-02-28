from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ._internal.compliance import ComplianceViolation
    from ._internal.content_filter import ContentViolation
    from ._internal.cost_guard import BudgetViolation
    from ._internal.injection import InjectionAnalysis


class PromptNotFoundError(Exception):
    """Raised when a prompt slug does not exist (404)."""

    def __init__(self, slug: str) -> None:
        self.slug = slug
        super().__init__(f'Prompt "{slug}" not found')


class PromptInjectionError(Exception):
    """Raised when prompt injection is detected above the block threshold."""

    def __init__(self, analysis: InjectionAnalysis) -> None:
        self.analysis = analysis
        super().__init__(
            f"Prompt injection detected (risk: {analysis.risk_score}, "
            f"categories: {', '.join(analysis.triggered)})"
        )


class CostLimitError(Exception):
    """Raised when a cost/budget limit is exceeded."""

    def __init__(self, violation: BudgetViolation) -> None:
        self.violation = violation
        super().__init__(
            f"Cost limit exceeded: {violation.type} -- "
            f"current: ${violation.current_spend:.4f}, limit: ${violation.limit:.4f}"
        )


class ContentViolationError(Exception):
    """Raised when content policy violations are detected."""

    def __init__(self, violations: List[ContentViolation]) -> None:
        self.violations = violations
        categories = ", ".join(sorted({v.category for v in violations}))
        super().__init__(f"Content policy violation: {categories}")


class ComplianceError(Exception):
    """Raised when compliance checks fail."""

    def __init__(self, violations: List[ComplianceViolation]) -> None:
        self.violations = violations
        messages = "; ".join(v.message for v in violations)
        super().__init__(f"Compliance check failed: {messages}")
