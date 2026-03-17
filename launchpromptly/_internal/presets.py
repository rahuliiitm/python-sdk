"""
Sensitivity presets for quick security configuration.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from ..types import SecurityOptions

SensitivityPreset = Literal["strict", "balanced", "permissive"]

# Preset definitions — dicts matching SecurityOptions field names
_PRESETS: dict[str, dict[str, dict]] = {
    "strict": {
        "injection": {"block_threshold": 0.5, "block_on_high_risk": True},
        "jailbreak": {"block_threshold": 0.5, "warn_threshold": 0.2, "block_on_detection": True},
        "content_filter": {"block_on_violation": True},
        "pii": {"redaction": "placeholder", "scan_response": True},
        "secret_detection": {"action": "block", "scan_response": True},
        "unicode_sanitizer": {"action": "block", "detect_homoglyphs": True},
        "tool_guard": {"enabled": True, "dangerous_arg_detection": True, "action": "block"},
        "chain_of_thought": {"enabled": True, "injection_detection": True, "action": "block"},
    },
    "balanced": {
        "injection": {"block_threshold": 0.7, "block_on_high_risk": True},
        "jailbreak": {"block_threshold": 0.7, "warn_threshold": 0.3, "block_on_detection": True},
        "content_filter": {"block_on_violation": True},
        "pii": {"redaction": "placeholder", "scan_response": True},
        "secret_detection": {"action": "redact", "scan_response": True},
        "unicode_sanitizer": {"action": "strip", "detect_homoglyphs": True},
        "tool_guard": {"enabled": True, "dangerous_arg_detection": True, "action": "warn"},
        "chain_of_thought": {"enabled": True, "injection_detection": True, "action": "warn"},
    },
    "permissive": {
        "injection": {"block_threshold": 0.85, "block_on_high_risk": False},
        "jailbreak": {"block_threshold": 0.85, "warn_threshold": 0.5, "block_on_detection": False},
        "content_filter": {"block_on_violation": False},
        "pii": {"redaction": "placeholder", "scan_response": False},
        "secret_detection": {"action": "warn"},
        "unicode_sanitizer": {"action": "warn"},
        "tool_guard": {"enabled": True, "dangerous_arg_detection": True, "action": "flag"},
    },
}


def _get_type_for_key(key: str):
    """Get the dataclass type for a given security sub-option key."""
    from ..types import (
        InjectionSecurityOptions,
        JailbreakSecurityOptions,
        PIISecurityOptions,
        SecretDetectionSecurityOptions,
        UnicodeSanitizerSecurityOptions,
    )
    from .content_filter import ContentFilterOptions
    from .tool_guard import ToolGuardOptions
    from .cot_guard import ChainOfThoughtGuardOptions

    _map = {
        "injection": InjectionSecurityOptions,
        "jailbreak": JailbreakSecurityOptions,
        "pii": PIISecurityOptions,
        "secret_detection": SecretDetectionSecurityOptions,
        "unicode_sanitizer": UnicodeSanitizerSecurityOptions,
        "content_filter": ContentFilterOptions,
        "tool_guard": ToolGuardOptions,
        "chain_of_thought": ChainOfThoughtGuardOptions,
    }
    return _map.get(key)


def resolve_security_options(options: SecurityOptions) -> SecurityOptions:
    """Resolve security options by applying preset defaults, then user overrides.

    User values always win. Only fills in gaps.
    If no preset is specified, returns options unchanged.
    """
    if not options.preset:
        return options

    preset_defaults = _PRESETS.get(options.preset)
    if not preset_defaults:
        return options

    changes: dict = {}
    for key, preset_val in preset_defaults.items():
        user_val = getattr(options, key, None)
        if user_val is None:
            # User didn't set this module — construct from preset dict
            cls = _get_type_for_key(key)
            if cls:
                changes[key] = cls(**preset_val)
        elif hasattr(user_val, '__dataclass_fields__'):
            # Both are dataclasses — merge: preset fills gaps in user values
            merged = {}
            for pkey, pval in preset_val.items():
                current = getattr(user_val, pkey, None)
                if current is None:
                    merged[pkey] = pval
            if merged:
                changes[key] = dc_replace(user_val, **merged)

    if changes:
        return dc_replace(options, **changes)
    return options
