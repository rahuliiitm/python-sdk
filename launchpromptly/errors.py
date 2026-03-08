from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ._internal.content_filter import ContentViolation
    from ._internal.cost_guard import BudgetViolation
    from ._internal.injection import InjectionAnalysis
    from ._internal.jailbreak import JailbreakAnalysis
    from ._internal.model_policy import ModelPolicyViolation
    from ._internal.schema_validator import SchemaValidationError
    from ._internal.topic_guard import TopicViolation
    from .types import StreamViolation


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


class ModelPolicyError(Exception):
    """Raised when a model policy violation is detected."""

    def __init__(self, violation: ModelPolicyViolation) -> None:
        self.violation = violation
        super().__init__(f"Model policy violation: {violation.message}")


class OutputSchemaError(Exception):
    """Raised when the LLM response fails output schema validation."""

    def __init__(
        self,
        validation_errors: List[SchemaValidationError],
        response_text: str,
    ) -> None:
        self.validation_errors = validation_errors
        self.response_text = response_text
        summary = "; ".join(e.message for e in validation_errors[:3])
        super().__init__(f"Output schema validation failed: {summary}")


class JailbreakError(Exception):
    """Raised when jailbreak attempt is detected above the block threshold."""

    def __init__(self, analysis: JailbreakAnalysis) -> None:
        self.analysis = analysis
        super().__init__(
            f"Jailbreak detected (risk: {analysis.risk_score}, "
            f"categories: {', '.join(analysis.triggered)})"
        )


class TopicViolationError(Exception):
    """Raised when a topic guard violation is detected."""

    def __init__(self, violation: TopicViolation) -> None:
        self.violation = violation
        topic_info = f" -- {violation.topic}" if violation.topic else ""
        super().__init__(f"Topic violation: {violation.type}{topic_info}")


class StreamAbortError(Exception):
    """Raised when a streaming response is aborted due to a mid-stream violation."""

    def __init__(self, violation: StreamViolation, partial_response: str) -> None:
        self.violation = violation
        self.partial_response = partial_response
        self.approximate_tokens = len(partial_response) // 4
        super().__init__(
            f"Stream aborted: {violation.type} violation detected at offset {violation.offset}"
        )
