from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Optional

_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}(T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?)?")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_LONG_NUMBER_RE = re.compile(r"\b\d{4,}\b")
_URL_RE = re.compile(r"https?://[^\s]+")
_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_prompt(text: str) -> str:
    text = _UUID_RE.sub("<UUID>", text)
    text = _ISO_DATE_RE.sub("<DATE>", text)
    text = _EMAIL_RE.sub("<EMAIL>", text)
    text = _URL_RE.sub("<URL>", text)
    text = _LONG_NUMBER_RE.sub("<NUM>", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


def _hash_prompt(normalized_text: str) -> str:
    return hashlib.sha256(normalized_text.encode()).hexdigest()


@dataclass
class PromptFingerprint:
    system_hash: Optional[str]
    full_hash: str
    normalized_system: Optional[str]
    prompt_preview: str


def fingerprint_messages(
    messages: list[dict[str, str]],
    system_prompt: Optional[str] = None,
) -> PromptFingerprint:
    """Fingerprint a set of chat messages for deduplication and tracking."""
    full_text = "\n".join(f"{m['role']}:{m['content']}" for m in messages)
    normalized_full = _normalize_prompt(full_text)
    full_hash = _hash_prompt(normalized_full)

    system_hash: Optional[str] = None
    normalized_system: Optional[str] = None

    if system_prompt:
        normalized_system = _normalize_prompt(system_prompt)
        system_hash = _hash_prompt(normalized_system)

    prompt_preview = full_text[:200]

    return PromptFingerprint(
        system_hash=system_hash,
        full_hash=full_hash,
        normalized_system=normalized_system,
        prompt_preview=prompt_preview,
    )
