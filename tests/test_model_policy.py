"""Tests for model policy enforcement."""

import pytest

from launchpromptly._internal.model_policy import (
    ModelPolicyOptions,
    ModelPolicyViolation,
    check_model_policy,
)


# ── Model whitelist ──────────────────────────────────────────────────────────


class TestAllowedModels:
    def test_allows_model_on_whitelist(self):
        opts = ModelPolicyOptions(allowed_models=["gpt-4o", "gpt-4o-mini", "claude-sonnet-4-20250514"])
        result = check_model_policy(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]},
            opts,
        )
        assert result is None

    def test_blocks_model_not_on_whitelist(self):
        opts = ModelPolicyOptions(allowed_models=["gpt-4o", "gpt-4o-mini"])
        result = check_model_policy(
            {"model": "gpt-3.5-turbo", "messages": [{"role": "user", "content": "hi"}]},
            opts,
        )
        assert result is not None
        assert result.rule == "model_not_allowed"
        assert result.actual == "gpt-3.5-turbo"
        assert result.limit == ["gpt-4o", "gpt-4o-mini"]

    def test_passes_when_allowed_models_empty(self):
        opts = ModelPolicyOptions(allowed_models=[])
        result = check_model_policy({"model": "any-model", "messages": []}, opts)
        assert result is None

    def test_passes_when_allowed_models_not_set(self):
        opts = ModelPolicyOptions()
        result = check_model_policy({"model": "any-model", "messages": []}, opts)
        assert result is None


# ── Max tokens ───────────────────────────────────────────────────────────────


class TestMaxTokens:
    def test_allows_within_limit(self):
        opts = ModelPolicyOptions(max_tokens=4096)
        result = check_model_policy(
            {"model": "gpt-4o", "max_tokens": 2048, "messages": []}, opts
        )
        assert result is None

    def test_allows_exactly_at_limit(self):
        opts = ModelPolicyOptions(max_tokens=4096)
        result = check_model_policy(
            {"model": "gpt-4o", "max_tokens": 4096, "messages": []}, opts
        )
        assert result is None

    def test_blocks_exceeding_limit(self):
        opts = ModelPolicyOptions(max_tokens=4096)
        result = check_model_policy(
            {"model": "gpt-4o", "max_tokens": 8192, "messages": []}, opts
        )
        assert result is not None
        assert result.rule == "max_tokens_exceeded"
        assert result.actual == 8192
        assert result.limit == 4096

    def test_passes_when_max_tokens_not_in_params(self):
        opts = ModelPolicyOptions(max_tokens=4096)
        result = check_model_policy({"model": "gpt-4o", "messages": []}, opts)
        assert result is None


# ── Temperature ──────────────────────────────────────────────────────────────


class TestMaxTemperature:
    def test_allows_within_limit(self):
        opts = ModelPolicyOptions(max_temperature=1.0)
        result = check_model_policy(
            {"model": "gpt-4o", "temperature": 0.7, "messages": []}, opts
        )
        assert result is None

    def test_allows_exactly_at_limit(self):
        opts = ModelPolicyOptions(max_temperature=1.0)
        result = check_model_policy(
            {"model": "gpt-4o", "temperature": 1.0, "messages": []}, opts
        )
        assert result is None

    def test_blocks_exceeding_limit(self):
        opts = ModelPolicyOptions(max_temperature=1.0)
        result = check_model_policy(
            {"model": "gpt-4o", "temperature": 1.5, "messages": []}, opts
        )
        assert result is not None
        assert result.rule == "temperature_exceeded"
        assert result.actual == 1.5
        assert result.limit == 1.0

    def test_allows_zero_temperature(self):
        opts = ModelPolicyOptions(max_temperature=1.0)
        result = check_model_policy(
            {"model": "gpt-4o", "temperature": 0, "messages": []}, opts
        )
        assert result is None

    def test_passes_when_temperature_not_in_params(self):
        opts = ModelPolicyOptions(max_temperature=1.0)
        result = check_model_policy({"model": "gpt-4o", "messages": []}, opts)
        assert result is None


# ── System prompt blocking ───────────────────────────────────────────────────


class TestBlockSystemPrompt:
    def test_blocks_system_role_in_messages(self):
        opts = ModelPolicyOptions(block_system_prompt_override=True)
        result = check_model_policy(
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "hi"},
                ],
            },
            opts,
        )
        assert result is not None
        assert result.rule == "system_prompt_blocked"

    def test_blocks_system_field_anthropic_style(self):
        opts = ModelPolicyOptions(block_system_prompt_override=True)
        result = check_model_policy(
            {
                "model": "claude-sonnet-4-20250514",
                "messages": [{"role": "user", "content": "hi"}],
                "system": "You are a helpful assistant",
            },
            opts,
        )
        assert result is not None
        assert result.rule == "system_prompt_blocked"

    def test_allows_when_no_system_prompt(self):
        opts = ModelPolicyOptions(block_system_prompt_override=True)
        result = check_model_policy(
            {
                "model": "gpt-4o",
                "messages": [{"role": "user", "content": "hi"}],
            },
            opts,
        )
        assert result is None

    def test_passes_when_blocking_disabled(self):
        opts = ModelPolicyOptions(block_system_prompt_override=False)
        result = check_model_policy(
            {
                "model": "gpt-4o",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": "hi"},
                ],
            },
            opts,
        )
        assert result is None


# ── Combined policies ────────────────────────────────────────────────────────


class TestCombinedPolicies:
    def test_returns_first_violation_model_check_first(self):
        opts = ModelPolicyOptions(
            allowed_models=["gpt-4o"],
            max_tokens=4096,
            max_temperature=1.0,
            block_system_prompt_override=True,
        )
        result = check_model_policy(
            {
                "model": "gpt-3.5-turbo",
                "max_tokens": 10000,
                "temperature": 2.0,
                "messages": [{"role": "system", "content": "yo"}],
            },
            opts,
        )
        assert result is not None
        assert result.rule == "model_not_allowed"

    def test_passes_when_all_satisfied(self):
        opts = ModelPolicyOptions(
            allowed_models=["gpt-4o"],
            max_tokens=4096,
            max_temperature=1.0,
            block_system_prompt_override=True,
        )
        result = check_model_policy(
            {
                "model": "gpt-4o",
                "max_tokens": 1024,
                "temperature": 0.5,
                "messages": [{"role": "user", "content": "hi"}],
            },
            opts,
        )
        assert result is None

    def test_empty_config_passes_everything(self):
        opts = ModelPolicyOptions()
        result = check_model_policy(
            {
                "model": "anything",
                "max_tokens": 999999,
                "temperature": 99,
                "messages": [{"role": "system", "content": "x"}],
            },
            opts,
        )
        assert result is None
