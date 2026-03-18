"""Tests for the ML resolver module.

Tests resolve_guardrail_list and merge_ml_providers without requiring
any ML dependencies (no transformers, torch, etc.).
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from unittest.mock import MagicMock

import pytest

from launchpromptly._internal.ml_resolver import (
    _ALL_ML_GUARDRAILS,
    merge_ml_providers,
    resolve_guardrail_list,
)
from launchpromptly.types import (
    HallucinationSecurityOptions,
    InjectionSecurityOptions,
    JailbreakSecurityOptions,
    PIISecurityOptions,
    SecurityOptions,
)


# ---------------------------------------------------------------------------
# resolve_guardrail_list
# ---------------------------------------------------------------------------


def test_true_returns_all_guardrails():
    result = resolve_guardrail_list(True)
    assert result == _ALL_ML_GUARDRAILS
    assert len(result) == 7


def test_false_returns_empty():
    result = resolve_guardrail_list(False)
    assert result == []


def test_empty_list_returns_empty():
    result = resolve_guardrail_list([])
    assert result == []


def test_valid_array_returns_canonical_names():
    result = resolve_guardrail_list(["injection", "pii"])
    assert result == ["injection", "pii"]


def test_alias_content_filter_maps_to_toxicity():
    result = resolve_guardrail_list(["content_filter"])
    assert result == ["toxicity"]


def test_all_aliases_resolve_correctly():
    result = resolve_guardrail_list(["injection", "jailbreak", "pii", "toxicity", "hallucination"])
    assert result == ["injection", "jailbreak", "pii", "toxicity", "hallucination"]


def test_invalid_name_raises_value_error():
    with pytest.raises(ValueError, match='Invalid use_ml guardrail: "bogus"'):
        resolve_guardrail_list(["bogus"])


def test_invalid_name_in_middle_raises_value_error():
    with pytest.raises(ValueError, match='Invalid use_ml guardrail: "nope"'):
        resolve_guardrail_list(["injection", "nope", "pii"])


def test_deduplication_preserves_order():
    result = resolve_guardrail_list(["injection", "pii", "injection"])
    assert result == ["injection", "pii"]


def test_alias_deduplication():
    """content_filter and toxicity both resolve to 'toxicity' -- should deduplicate."""
    result = resolve_guardrail_list(["toxicity", "content_filter"])
    assert result == ["toxicity"]


def test_single_guardrail():
    result = resolve_guardrail_list(["jailbreak"])
    assert result == ["jailbreak"]


# ---------------------------------------------------------------------------
# merge_ml_providers
# ---------------------------------------------------------------------------


def _make_security(**kwargs) -> SecurityOptions:
    """Create a SecurityOptions with optional field overrides."""
    return SecurityOptions(**kwargs)


def test_merge_empty_providers_returns_original():
    security = _make_security()
    result = merge_ml_providers(security, {})
    assert result is security


def test_merge_injection_provider_creates_options():
    security = _make_security()
    mock_detector = MagicMock()
    result = merge_ml_providers(security, {"injection": mock_detector})
    assert result.injection is not None
    assert result.injection.providers == [mock_detector]


def test_merge_injection_provider_appends_to_existing():
    existing_provider = MagicMock()
    security = _make_security(
        injection=InjectionSecurityOptions(providers=[existing_provider]),
    )
    new_provider = MagicMock()
    result = merge_ml_providers(security, {"injection": new_provider})
    assert result.injection.providers == [existing_provider, new_provider]


def test_merge_jailbreak_provider_creates_options():
    security = _make_security()
    mock_detector = MagicMock()
    result = merge_ml_providers(security, {"jailbreak": mock_detector})
    assert result.jailbreak is not None
    assert result.jailbreak.providers == [mock_detector]


def test_merge_jailbreak_provider_appends_to_existing():
    existing_provider = MagicMock()
    security = _make_security(
        jailbreak=JailbreakSecurityOptions(providers=[existing_provider]),
    )
    new_provider = MagicMock()
    result = merge_ml_providers(security, {"jailbreak": new_provider})
    assert result.jailbreak.providers == [existing_provider, new_provider]


def test_merge_pii_provider_creates_options():
    security = _make_security()
    mock_detector = MagicMock()
    result = merge_ml_providers(security, {"pii": mock_detector})
    assert result.pii is not None
    assert result.pii.providers == [mock_detector]


def test_merge_pii_provider_appends_to_existing():
    existing_provider = MagicMock()
    security = _make_security(
        pii=PIISecurityOptions(providers=[existing_provider]),
    )
    new_provider = MagicMock()
    result = merge_ml_providers(security, {"pii": new_provider})
    assert result.pii.providers == [existing_provider, new_provider]


def test_merge_toxicity_provider_sets_content_filter():
    """Toxicity ML provider maps to the content_filter field on SecurityOptions."""
    security = _make_security()
    mock_detector = MagicMock()
    result = merge_ml_providers(security, {"toxicity": mock_detector})
    assert result.content_filter is not None
    assert result.content_filter.providers == [mock_detector]


def test_merge_hallucination_provider_creates_options():
    security = _make_security()
    mock_detector = MagicMock()
    result = merge_ml_providers(security, {"hallucination": mock_detector})
    assert result.hallucination is not None
    assert result.hallucination.providers == [mock_detector]


def test_merge_hallucination_provider_appends_to_existing():
    existing_provider = MagicMock()
    security = _make_security(
        hallucination=HallucinationSecurityOptions(providers=[existing_provider]),
    )
    new_provider = MagicMock()
    result = merge_ml_providers(security, {"hallucination": new_provider})
    assert result.hallucination.providers == [existing_provider, new_provider]


def test_merge_multiple_providers_at_once():
    security = _make_security()
    inj = MagicMock()
    pii = MagicMock()
    result = merge_ml_providers(security, {"injection": inj, "pii": pii})
    assert result.injection.providers == [inj]
    assert result.pii.providers == [pii]


def test_merge_does_not_mutate_original():
    security = _make_security()
    mock_detector = MagicMock()
    result = merge_ml_providers(security, {"injection": mock_detector})
    assert result is not security
    assert security.injection is None


def test_merge_preserves_existing_fields():
    security = _make_security(
        injection=InjectionSecurityOptions(enabled=True, block_threshold=0.8),
    )
    mock_detector = MagicMock()
    result = merge_ml_providers(security, {"injection": mock_detector})
    assert result.injection.enabled is True
    assert result.injection.block_threshold == 0.8
    assert result.injection.providers == [mock_detector]


# ---------------------------------------------------------------------------
# create_ml_providers — invalid guardrail names only (no ML deps)
# ---------------------------------------------------------------------------


def test_create_ml_providers_invalid_guardrail_raises():
    """create_ml_providers calls resolve_guardrail_list first, so invalid names raise."""
    from launchpromptly._internal.ml_resolver import create_ml_providers

    with pytest.raises(ValueError, match="Invalid use_ml guardrail"):
        create_ml_providers(["nonexistent"])


def test_create_ml_providers_false_returns_empty():
    from launchpromptly._internal.ml_resolver import create_ml_providers

    result = create_ml_providers(False)
    assert result == {}
