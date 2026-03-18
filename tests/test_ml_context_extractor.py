"""Tests for MLContextExtractor and extract_context_with_providers."""
import math

import numpy as np
import pytest

from launchpromptly._internal.context_engine import (
    ContextProfile,
    clear_context_cache,
    extract_context_with_providers,
)


# ── Mock Embedding Provider ─────────────────────────────────────────────────


class MockEmbeddingProvider:
    """Mock embedding provider where text → vector mapping is controlled.

    Each text is mapped to a unit vector in a specific dimension via text_to_index.
    Texts not in the map get a uniform vector (won't match any template strongly).
    """

    name = "mock-embedding"
    model_name = "test-model"

    def __init__(self, text_to_index: dict[str, int], dimension: int = 10):
        self._mapping = text_to_index
        self._dim = dimension

    def _make_vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._dim, dtype=np.float32)
        idx = self._mapping.get(text)
        if idx is not None and idx < self._dim:
            vec[idx] = 1.0
        else:
            vec[:] = 1.0 / math.sqrt(self._dim)
        return vec

    def embed(self, text: str) -> np.ndarray:
        return self._make_vector(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self._make_vector(t) for t in texts]

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = float(np.dot(a, b))
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        denom = norm_a * norm_b
        return 0.0 if denom == 0 else dot / denom


# ── Template text constants (must match _CONSTRAINT_TEMPLATE_SPECS) ──────────

TEMPLATE_ROLE = "The assistant must play a specific role, character, or professional identity"
TEMPLATE_ALLOWED = "The assistant is restricted to discussing only certain permitted topics"
TEMPLATE_RESTRICTED = "The assistant is prohibited from discussing certain topics"
TEMPLATE_ACTION = "The assistant must never perform certain forbidden actions or behaviors"
TEMPLATE_FORMAT = "The response must always be in a specific output format like JSON or markdown"
TEMPLATE_KNOWLEDGE = "The assistant must only use information from provided documents and not external knowledge"
TEMPLATE_PERSONA = "The assistant must maintain a specific personality trait, tone, or communication style"


def template_map() -> dict[str, int]:
    return {
        TEMPLATE_ROLE: 0,
        TEMPLATE_ALLOWED: 1,
        TEMPLATE_RESTRICTED: 2,
        TEMPLATE_ACTION: 3,
        TEMPLATE_FORMAT: 4,
        TEMPLATE_KNOWLEDGE: 5,
        TEMPLATE_PERSONA: 6,
    }


def _make_extractor(extra_mappings: dict[str, int] | None = None, threshold: float = 0.45):
    from launchpromptly.ml.context_extractor import MLContextExtractor

    mapping = template_map()
    if extra_mappings:
        mapping.update(extra_mappings)
    emb = MockEmbeddingProvider(mapping)
    return MLContextExtractor._create_for_test(emb, threshold)


# ── MLContextExtractor Tests ────────────────────────────────────────────────


class TestMLContextExtractor:
    def test_name(self):
        extractor = _make_extractor()
        assert extractor.name == "ml-context-extractor"

    def test_embedding_provider(self):
        extractor = _make_extractor()
        assert extractor.embedding_provider is not None

    def test_empty_string(self):
        extractor = _make_extractor()
        profile = extractor.extract("")
        assert profile.role is None
        assert profile.constraints == []
        assert profile.allowed_topics == []

    def test_whitespace(self):
        extractor = _make_extractor()
        profile = extractor.extract("   \n\t  ")
        assert profile.role is None
        assert profile.constraints == []

    def test_role_classification(self):
        extractor = _make_extractor({
            "You are a cooking assistant for our restaurant.": 0,
        })
        profile = extractor.extract("You are a cooking assistant for our restaurant.")
        assert profile.role is not None
        assert any(c.type == "role_constraint" for c in profile.constraints)

    def test_allowed_topic(self):
        extractor = _make_extractor({
            "Only help users with restaurant menu questions.": 1,
        })
        profile = extractor.extract("Only help users with restaurant menu questions.")
        assert len(profile.allowed_topics) > 0
        assert any(
            c.type == "topic_boundary" and c.description.startswith("Allowed")
            for c in profile.constraints
        )

    def test_restricted_topic_with_negative(self):
        extractor = _make_extractor({
            "Don't discuss competitor restaurants.": 2,
        })
        profile = extractor.extract("Don't discuss competitor restaurants.")
        assert len(profile.restricted_topics) > 0
        assert any(
            c.type == "topic_boundary" and c.description.startswith("Restricted")
            for c in profile.constraints
        )

    def test_forbidden_action(self):
        extractor = _make_extractor({
            "Never reveal internal pricing or recipes to users.": 3,
        })
        profile = extractor.extract("Never reveal internal pricing or recipes to users.")
        assert len(profile.forbidden_actions) > 0
        assert any(c.type == "action_restriction" for c in profile.constraints)

    def test_output_format(self):
        extractor = _make_extractor({
            "Respond in bullet points for all menu recommendations.": 4,
        })
        profile = extractor.extract("Respond in bullet points for all menu recommendations.")
        assert profile.output_format is not None
        assert any(c.type == "output_format" for c in profile.constraints)

    def test_knowledge_boundary(self):
        extractor = _make_extractor({
            "Only answer from the provided documents and menu data.": 5,
        })
        profile = extractor.extract("Only answer from the provided documents and menu data.")
        assert profile.grounding_mode == "documents_only"
        assert any(c.type == "knowledge_boundary" for c in profile.constraints)

    def test_persona_rule(self):
        extractor = _make_extractor({
            "Keep conversations family-friendly and professional.": 6,
        })
        profile = extractor.extract("Keep conversations family-friendly and professional.")
        assert any(c.type == "persona_rule" for c in profile.constraints)

    def test_multi_sentence(self):
        extractor = _make_extractor({
            "You are a helpful cooking assistant.": 0,
            "Never discuss politics or religion.": 2,
        })
        profile = extractor.extract(
            "You are a helpful cooking assistant. Never discuss politics or religion."
        )
        assert profile.role is not None
        assert len(profile.restricted_topics) > 0
        assert len(profile.constraints) >= 2

    def test_below_threshold(self):
        extractor = _make_extractor()
        profile = extractor.extract("This is a random sentence about nothing.")
        assert profile.constraints == []
        assert profile.role is None

    def test_entity_always_none(self):
        extractor = _make_extractor({
            "You work for Acme Corp as their assistant.": 0,
        })
        profile = extractor.extract("You work for Acme Corp as their assistant.")
        assert profile.entity is None

    def test_constraint_confidence_range(self):
        extractor = _make_extractor({
            "You are a travel guide.": 0,
        })
        profile = extractor.extract("You are a travel guide.")
        for c in profile.constraints:
            assert 0 < c.confidence <= 0.9

    def test_short_forbidden_action_skipped(self):
        extractor = _make_extractor({
            "Don't go.": 3,
        })
        profile = extractor.extract("Don't go.")
        assert profile.forbidden_actions == []

    def test_custom_threshold(self):
        extractor = _make_extractor(threshold=0.99)
        profile = extractor.extract("You are a cooking assistant.")
        assert profile.constraints == []

    def test_raw_system_prompt(self):
        extractor = _make_extractor()
        prompt = "You are a helpful assistant."
        profile = extractor.extract(prompt)
        assert profile.raw_system_prompt == prompt


# ── extract_context_with_providers Tests ─────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_context_cache()
    yield
    clear_context_cache()


class _MockProvider:
    """Simple mock ContextExtractorProvider."""

    def __init__(self, profile: ContextProfile):
        self._profile = profile
        self.name = "mock-provider"

    def extract(self, system_prompt: str) -> ContextProfile:
        return self._profile


class TestExtractContextWithProviders:
    def test_no_providers(self):
        profile = extract_context_with_providers("You are a cooking assistant.", [])
        assert profile.role == "cooking assistant"
        assert len(profile.constraints) > 0

    def test_merge_ml_topics(self):
        ml = _MockProvider(
            ContextProfile(
                allowed_topics=["cooking", "recipes"],
                restricted_topics=["politics"],
                constraints=[],
            )
        )
        profile = extract_context_with_providers("You are a cooking assistant.", [ml])
        assert profile.role == "cooking assistant"
        assert "cooking" in profile.allowed_topics
        assert "politics" in profile.restricted_topics

    def test_prefer_regex_role(self):
        ml = _MockProvider(ContextProfile(role="ml-detected-role"))
        profile = extract_context_with_providers("You are a cooking assistant.", [ml])
        assert profile.role == "cooking assistant"

    def test_fallback_ml_role(self):
        ml = _MockProvider(ContextProfile(role="restaurant helper"))
        profile = extract_context_with_providers("Help users find good restaurants.", [ml])
        assert profile.role == "restaurant helper"

    def test_dedup_topics(self):
        ml = _MockProvider(
            ContextProfile(
                allowed_topics=["cooking and recipes", "nutrition"],
            )
        )
        profile = extract_context_with_providers("Only discuss cooking and recipes.", [ml])
        # Regex splits "cooking and recipes" into "cooking" + "recipes"; ML returns "cooking and recipes"
        assert "cooking" in profile.allowed_topics
        assert "nutrition" in profile.allowed_topics
        assert len(profile.allowed_topics) == len(set(profile.allowed_topics))

    def test_higher_confidence_wins(self):
        from launchpromptly._internal.context_engine import Constraint

        ml = _MockProvider(
            ContextProfile(
                constraints=[
                    Constraint(
                        type="topic_boundary",
                        description="ML: Allowed topic",
                        keywords=["cooking"],
                        source="Only discuss cooking and recipes.",
                        confidence=0.95,
                    )
                ],
            )
        )
        profile = extract_context_with_providers("Only discuss cooking and recipes.", [ml])
        topic_c = next(
            (c for c in profile.constraints
             if c.type == "topic_boundary" and c.source == "Only discuss cooking and recipes."),
            None,
        )
        assert topic_c is not None
        assert topic_c.confidence == 0.95

    def test_merge_grounding_mode(self):
        ml = _MockProvider(ContextProfile(grounding_mode="documents_only"))
        profile = extract_context_with_providers("Help users with questions.", [ml])
        assert profile.grounding_mode == "documents_only"

    def test_multiple_providers(self):
        p1 = _MockProvider(
            ContextProfile(
                role="helper",
                allowed_topics=["topic-a"],
            )
        )
        p2 = _MockProvider(
            ContextProfile(
                allowed_topics=["topic-b"],
                restricted_topics=["topic-c"],
                forbidden_actions=["share secrets"],
            )
        )
        profile = extract_context_with_providers("Help users.", [p1, p2])
        assert profile.role == "helper"
        assert "topic-a" in profile.allowed_topics
        assert "topic-b" in profile.allowed_topics
        assert "topic-c" in profile.restricted_topics
        assert "share secrets" in profile.forbidden_actions
