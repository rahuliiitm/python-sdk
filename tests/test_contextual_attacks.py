"""Tests for the Contextual Attack generation module."""
import pytest

from launchpromptly.redteam.contextual_attacks import generate_contextual_attacks
from launchpromptly._internal.context_engine import (
    extract_context, clear_context_cache, ContextProfile,
)


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_context_cache()
    yield
    clear_context_cache()


def make_profile(**kwargs) -> ContextProfile:
    defaults = dict(
        role=None, entity=None, allowed_topics=[], restricted_topics=[],
        forbidden_actions=[], output_format=None, grounding_mode="any",
        constraints=[], raw_system_prompt="", prompt_hash="",
    )
    defaults.update(kwargs)
    return ContextProfile(**defaults)


class TestGenerateContextualAttacks:
    def test_returns_empty_for_empty_profile(self):
        attacks = generate_contextual_attacks(make_profile())
        assert attacks == []

    def test_generates_role_attacks_when_role_set(self):
        attacks = generate_contextual_attacks(make_profile(role="customer support agent"))
        assert len(attacks) >= 3
        assert any(a.category == "jailbreak" for a in attacks)
        assert any("customer support agent" in a.name for a in attacks)

    def test_generates_topic_attacks_for_restricted_topics(self):
        attacks = generate_contextual_attacks(
            make_profile(restricted_topics=["politics", "religion"])
        )
        content_bypasses = [a for a in attacks if a.category == "content_bypass"]
        assert len(content_bypasses) >= 6  # 3 per topic
        assert any("politics" in a.name for a in content_bypasses)
        assert any("religion" in a.name for a in content_bypasses)

    def test_generates_topic_drift_attacks_for_allowed_topics(self):
        attacks = generate_contextual_attacks(make_profile(allowed_topics=["cooking"]))
        assert any("Topic drift from cooking" in a.name for a in attacks)

    def test_generates_forbidden_action_attacks(self):
        attacks = generate_contextual_attacks(
            make_profile(forbidden_actions=["provide medical advice"])
        )
        action_attacks = [a for a in attacks if "provide medical advice" in a.name]
        assert len(action_attacks) == 3
        assert any(a.category == "injection" for a in action_attacks)

    def test_generates_entity_attacks_when_entity_set(self):
        attacks = generate_contextual_attacks(
            make_profile(entity="Acme Corp", restricted_topics=["competitors"])
        )
        assert any("Acme Corp" in a.name for a in attacks)
        assert any("Compare Acme Corp" in a.name for a in attacks)

    def test_generates_format_attacks_when_output_format_set(self):
        attacks = generate_contextual_attacks(make_profile(output_format="JSON"))
        assert any("JSON" in a.name for a in attacks)

    def test_generates_grounding_attacks_for_documents_only(self):
        attacks = generate_contextual_attacks(
            make_profile(grounding_mode="documents_only")
        )
        assert any("grounding" in a.name for a in attacks)
        assert len(attacks) == 2

    def test_no_grounding_attacks_for_any_mode(self):
        attacks = generate_contextual_attacks(make_profile(grounding_mode="any"))
        assert not any("grounding" in a.name for a in attacks)

    def test_all_attacks_have_valid_structure(self):
        attacks = generate_contextual_attacks(make_profile(
            role="banking assistant",
            entity="BigBank",
            restricted_topics=["politics", "gambling"],
            forbidden_actions=["transfer money", "reveal account numbers"],
            output_format="JSON",
            grounding_mode="documents_only",
            allowed_topics=["account inquiries"],
        ))
        assert len(attacks) > 10

        for attack in attacks:
            assert attack.id
            assert attack.category
            assert attack.name
            assert len(attack.messages) > 0
            assert attack.expected_outcome in ("blocked", "redacted", "warned", "refused")
            assert attack.severity in ("critical", "high", "medium", "low")
            assert len(attack.description) > 10

    def test_generates_attacks_from_real_extracted_profile(self):
        profile = extract_context(
            "You are a customer support agent for Acme Corp. "
            "Only discuss billing and account questions. "
            "Never discuss politics or competitors. "
            "Do not provide financial advice. "
            "Always respond in JSON. "
            "Only answer from the provided documents."
        )
        attacks = generate_contextual_attacks(profile)
        assert len(attacks) > 10
        assert any(a.category == "jailbreak" for a in attacks)
        assert any(a.category == "content_bypass" for a in attacks)
        assert any("financial advice" in a.name for a in attacks)
        assert any("grounding" in a.name for a in attacks)

    def test_has_unique_ids(self):
        attacks = generate_contextual_attacks(make_profile(
            role="helper",
            restricted_topics=["politics"],
            forbidden_actions=["give advice"],
            entity="TestCo",
            output_format="JSON",
            grounding_mode="documents_only",
            allowed_topics=["cooking"],
        ))
        ids = [a.id for a in attacks]
        assert len(set(ids)) == len(ids)
