"""Auto-resolve ML providers based on use_ml configuration.

Handles lazy creation of ML detectors and merging them into SecurityOptions.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import Any, Dict, List, Optional, Union

_GUARDRAIL_ALIASES: dict[str, str] = {
    "injection": "injection",
    "jailbreak": "jailbreak",
    "pii": "pii",
    "toxicity": "toxicity",
    "content_filter": "toxicity",  # alias
    "hallucination": "hallucination",
}

_ALL_ML_GUARDRAILS = ["injection", "jailbreak", "pii", "toxicity", "hallucination"]


def resolve_guardrail_list(use_ml: Union[bool, List[str]]) -> list[str]:
    """Normalize use_ml value into a deduplicated list of canonical guardrail types."""
    if use_ml is True:
        return list(_ALL_ML_GUARDRAILS)
    if use_ml is False or not use_ml:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for name in use_ml:
        normalized = _GUARDRAIL_ALIASES.get(name)
        if not normalized:
            raise ValueError(
                f'Invalid use_ml guardrail: "{name}". '
                f"Valid values: {', '.join(_GUARDRAIL_ALIASES.keys())}"
            )
        if normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def create_ml_providers(use_ml: Union[bool, List[str]]) -> dict[str, Any]:
    """Create ML providers for the requested guardrails.

    Dynamically imports from the ml/ module to avoid loading models when not needed.

    Raises ImportError if ML dependencies are not installed.
    """
    guardrails = set(resolve_guardrail_list(use_ml))
    if not guardrails:
        return {}

    result: dict[str, Any] = {}

    if "injection" in guardrails:
        from ..ml import MLInjectionDetector

        result["injection"] = MLInjectionDetector()

    if "jailbreak" in guardrails:
        from ..ml import MLJailbreakDetector

        result["jailbreak"] = MLJailbreakDetector()

    if "pii" in guardrails:
        from ..ml import PresidioPIIDetector

        result["pii"] = PresidioPIIDetector()

    if "toxicity" in guardrails:
        from ..ml import MLToxicityDetector

        result["toxicity"] = MLToxicityDetector()

    if "hallucination" in guardrails:
        from ..ml import MLHallucinationDetector

        result["hallucination"] = MLHallucinationDetector()

    return result


def merge_ml_providers(security: Any, ml_providers: dict[str, Any]) -> Any:
    """Merge resolved ML providers into SecurityOptions.

    ML providers are appended to any existing providers (not replaced).
    Returns a new SecurityOptions instance (does not mutate the original).
    """
    if not ml_providers:
        return security

    from ..types import (
        HallucinationSecurityOptions,
        InjectionSecurityOptions,
        JailbreakSecurityOptions,
        PIISecurityOptions,
    )
    from .content_filter import ContentFilterOptions

    changes: dict[str, Any] = {}

    if "injection" in ml_providers:
        existing = security.injection or InjectionSecurityOptions()
        providers = list(existing.providers or []) + [ml_providers["injection"]]
        changes["injection"] = dc_replace(existing, providers=providers)

    if "jailbreak" in ml_providers:
        existing = security.jailbreak or JailbreakSecurityOptions()
        providers = list(existing.providers or []) + [ml_providers["jailbreak"]]
        changes["jailbreak"] = dc_replace(existing, providers=providers)

    if "pii" in ml_providers:
        existing = security.pii or PIISecurityOptions()
        providers = list(existing.providers or []) + [ml_providers["pii"]]
        changes["pii"] = dc_replace(existing, providers=providers)

    if "toxicity" in ml_providers:
        existing = security.content_filter or ContentFilterOptions()
        providers = list(existing.providers or []) + [ml_providers["toxicity"]]
        changes["content_filter"] = dc_replace(existing, providers=providers)

    if "hallucination" in ml_providers:
        existing = security.hallucination or HallucinationSecurityOptions()
        providers = list(existing.providers or []) + [ml_providers["hallucination"]]
        changes["hallucination"] = dc_replace(existing, providers=providers)

    if changes:
        return dc_replace(security, **changes)
    return security
