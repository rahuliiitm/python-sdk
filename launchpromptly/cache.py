from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional

_DEFAULT_MAX_SIZE = 1000


@dataclass
class CacheEntry:
    content: str
    managed_prompt_id: str
    prompt_version_id: str
    version: int
    expires_at: float  # time.monotonic() value


class PromptCache:
    """In-memory LRU cache for resolved prompts."""

    def __init__(self, max_size: int = _DEFAULT_MAX_SIZE) -> None:
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size

    def get(self, slug: str) -> Optional[CacheEntry]:
        entry = self._cache.get(slug)
        if entry is None:
            return None
        if time.monotonic() >= entry.expires_at:
            return None
        # LRU: move to end (most recently used)
        self._cache.move_to_end(slug)
        return entry

    def get_stale(self, slug: str) -> Optional[CacheEntry]:
        """Return entry even if expired (for stale-while-error fallback)."""
        return self._cache.get(slug)

    def set(
        self,
        slug: str,
        content: str,
        managed_prompt_id: str,
        prompt_version_id: str,
        version: int,
        ttl: float,
    ) -> None:
        # If key exists, remove first so re-insert goes to end
        if slug in self._cache:
            del self._cache[slug]

        # Evict oldest if at capacity
        if len(self._cache) >= self._max_size:
            self._cache.popitem(last=False)

        self._cache[slug] = CacheEntry(
            content=content,
            managed_prompt_id=managed_prompt_id,
            prompt_version_id=prompt_version_id,
            version=version,
            expires_at=time.monotonic() + ttl,
        )

    def invalidate(self, slug: str) -> None:
        self._cache.pop(slug, None)

    def invalidate_all(self) -> None:
        self._cache.clear()

    @property
    def size(self) -> int:
        return len(self._cache)
