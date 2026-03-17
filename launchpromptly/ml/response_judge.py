"""
ML-based response judge using NLI cross-encoder + sentence embeddings.

Uses Natural Language Inference to semantically verify whether LLM responses
comply with extracted constraints. Much more accurate than keyword overlap
for detecting paraphrased topic violations, subtle role deviations, etc.

Two models:
- Embeddings (all-MiniLM-L6-v2, 3-5ms) -- semantic topic matching
- NLI cross-encoder (ms-marco-MiniLM-L-6-v2, 5-10ms) -- entailment checking

Requires::

    pip install onnxruntime tokenizers
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

from .onnx_runtime import OnnxSession
from .embedding_provider import MLEmbeddingProvider

_DEFAULT_NLI_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_MAX_RESPONSE_LEN = 512

# Severity weights per violation type (same as heuristic judge)
_VIOLATION_WEIGHTS: dict[str, float] = {
    "topic_violation": 0.25,
    "role_deviation": 0.15,
    "forbidden_action": 0.30,
    "format_violation": 0.15,
    "grounding_violation": 0.20,
    "persona_break": 0.10,
}


class MLResponseJudge:
    """ML-based response judge using NLI cross-encoder + embeddings.

    Implements the ``ResponseJudgeProvider`` protocol -- drop-in alongside
    the heuristic judge. Results are merged via ``merge_judgments()``.

    Example::

        from launchpromptly.ml import MLResponseJudge
        judge = MLResponseJudge.create()
        # Register via security options:
        security={"response_judge": {"providers": [judge]}}
    """

    name = "ml-nli-judge"

    def __init__(
        self,
        nli_session: OnnxSession,
        embedding: MLEmbeddingProvider,
        max_constraints: int = 10,
    ) -> None:
        self._nli_session = nli_session
        self._embedding = embedding
        self._max_constraints = max_constraints

    @classmethod
    def create(
        cls,
        *,
        nli_model: Optional[str] = None,
        max_constraints: int = 10,
        cache_dir: Optional[Path] = None,
        embedding_provider: Optional[MLEmbeddingProvider] = None,
    ) -> "MLResponseJudge":
        """Create an MLResponseJudge by loading the NLI + embedding models."""
        model = nli_model or _DEFAULT_NLI_MODEL
        nli_session = OnnxSession.create(model, max_length=_MAX_RESPONSE_LEN, cache_dir=cache_dir)

        embedding = embedding_provider or MLEmbeddingProvider.create(cache_dir=cache_dir)

        return cls(nli_session, embedding, max_constraints)

    @classmethod
    def _create_for_test(
        cls,
        nli_session: OnnxSession,
        embedding: MLEmbeddingProvider,
        max_constraints: int = 10,
    ) -> "MLResponseJudge":
        """For testing: create with pre-built sessions."""
        return cls(nli_session, embedding, max_constraints)

    @property
    def embedding_provider(self) -> MLEmbeddingProvider:
        """Get the shared embedding provider (for reuse by other guards)."""
        return self._embedding

    def judge(self, response_text: str, profile: Any) -> Any:
        """Judge whether an LLM response violates the constraints from the ContextProfile."""
        from .._internal.context_engine import Constraint, ContextProfile
        from .._internal.response_judge import BoundaryViolation, ResponseJudgment

        if not response_text or profile is None:
            return ResponseJudgment()

        violations: list[BoundaryViolation] = []
        truncated = response_text[: _MAX_RESPONSE_LEN * 4]

        # 1. Semantic topic checking via embeddings
        self._check_topic_violations(truncated, profile, violations)

        # 2. NLI constraint checking
        self._check_constraint_violations(truncated, profile, violations)

        # 3. Grounding check via embeddings
        self._check_grounding_violation(truncated, profile, violations)

        # Compute compliance score
        penalty = 0.0
        for v in violations:
            penalty += v.confidence * _VIOLATION_WEIGHTS.get(v.type, 0.15)
        compliance_score = max(0.0, round(1.0 - penalty, 2))
        severity: str
        if compliance_score >= 0.7:
            severity = "low"
        elif compliance_score >= 0.4:
            severity = "medium"
        else:
            severity = "high"

        return ResponseJudgment(
            violated=len(violations) > 0,
            compliance_score=compliance_score,
            violations=violations,
            severity=severity,  # type: ignore[arg-type]
        )

    # -- Private checking methods ---------------------------------------------

    def _check_topic_violations(
        self, response_text: str, profile: Any, violations: list,
    ) -> None:
        from .._internal.context_engine import Constraint
        from .._internal.response_judge import BoundaryViolation

        response_emb = self._embedding.embed(response_text)

        # Check restricted topics
        for topic in profile.restricted_topics:
            topic_emb = self._embedding.embed(topic)
            sim = self._embedding.cosine(response_emb, topic_emb)
            if sim > 0.65:
                constraint = Constraint(
                    type="topic_boundary",
                    description=f"Restricted topic: {topic}",
                    keywords=topic.lower().split(),
                    source=topic,
                    confidence=0.8,
                )
                violations.append(BoundaryViolation(
                    type="topic_violation",
                    constraint=constraint,
                    confidence=min(sim * 1.2, 1.0),
                    evidence=f'Response semantically similar to restricted topic "{topic}" (similarity: {sim:.2f})',
                ))

        # Check allowed topics
        if profile.allowed_topics:
            topic_embeddings = self._embedding.embed_batch(profile.allowed_topics)
            max_sim = max(
                self._embedding.cosine(response_emb, te)
                for te in topic_embeddings
            )
            if max_sim < 0.35:
                constraint = Constraint(
                    type="topic_boundary",
                    description=f"Allowed topics: {', '.join(profile.allowed_topics)}",
                    keywords=[w for t in profile.allowed_topics for w in t.lower().split()],
                    source="allowed topics",
                    confidence=0.7,
                )
                violations.append(BoundaryViolation(
                    type="topic_violation",
                    constraint=constraint,
                    confidence=min((1.0 - max_sim) * 0.8, 0.95),
                    evidence=f"Response not semantically related to any allowed topic (max similarity: {max_sim:.2f})",
                ))

    def _check_constraint_violations(
        self, response_text: str, profile: Any, violations: list,
    ) -> None:
        from .._internal.response_judge import BoundaryViolation

        priority_weights: dict[str, int] = {
            "action_restriction": 3,
            "negative_instruction": 2,
            "role_constraint": 1,
        }
        prioritized = sorted(
            profile.constraints,
            key=lambda c: priority_weights.get(c.type, 0),
            reverse=True,
        )
        constraints_to_check = prioritized[: self._max_constraints]

        for constraint in constraints_to_check:
            if constraint.type == "topic_boundary":
                continue

            premise = constraint.description
            hypothesis = response_text[:500]

            results = self._nli_session.classify_pair(premise, hypothesis, top_k=None)
            contradiction_score = self._get_contradiction_score(results)

            if contradiction_score > 0.6:
                violation_type = self._constraint_to_violation_type(constraint.type)
                violations.append(BoundaryViolation(
                    type=violation_type,
                    constraint=constraint,
                    confidence=round(contradiction_score, 2),
                    evidence=f'NLI: response contradicts constraint "{premise}" (score: {contradiction_score:.2f})',
                ))

    def _check_grounding_violation(
        self, response_text: str, profile: Any, violations: list,
    ) -> None:
        from .._internal.context_engine import Constraint
        from .._internal.response_judge import BoundaryViolation

        if profile.grounding_mode == "any":
            return

        response_emb = self._embedding.embed(response_text)
        prompt_emb = self._embedding.embed(profile.raw_system_prompt)
        sim = self._embedding.cosine(response_emb, prompt_emb)

        if sim < 0.25:
            constraint = Constraint(
                type="knowledge_boundary",
                description=f"Grounding mode: {profile.grounding_mode}",
                keywords=[],
                source="grounding",
                confidence=0.75,
            )
            violations.append(BoundaryViolation(
                type="grounding_violation",
                constraint=constraint,
                confidence=min((1.0 - sim) * 0.7, 0.9),
                evidence=f"Response semantically distant from system prompt (similarity: {sim:.2f})",
            ))

    # -- Helpers --------------------------------------------------------------

    @staticmethod
    def _get_contradiction_score(results: list[dict[str, Any]]) -> float:
        for r in results:
            label = r.get("label", "").upper()
            if label in ("CONTRADICTION", "LABEL_0"):
                return float(r["score"])
        for r in results:
            label = r.get("label", "").upper()
            if label in ("ENTAILMENT", "LABEL_2"):
                return 1.0 - float(r["score"])
        return 0.0

    @staticmethod
    def _constraint_to_violation_type(constraint_type: str) -> str:
        mapping = {
            "action_restriction": "forbidden_action",
            "negative_instruction": "forbidden_action",
            "role_constraint": "role_deviation",
            "knowledge_boundary": "grounding_violation",
            "output_format": "format_violation",
            "persona_rule": "persona_break",
        }
        return mapping.get(constraint_type, "topic_violation")
