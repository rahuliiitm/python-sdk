import time

from launchpromptly.cache import PromptCache


def _sample_data():
    return {
        "content": "You are helpful",
        "managed_prompt_id": "mp1",
        "prompt_version_id": "pv1",
        "version": 1,
    }


class TestPromptCache:
    def test_set_and_get(self):
        cache = PromptCache()
        cache.set("slug", **_sample_data(), ttl=60.0)
        entry = cache.get("slug")
        assert entry is not None
        assert entry.content == "You are helpful"
        assert entry.managed_prompt_id == "mp1"

    def test_get_returns_none_for_missing(self):
        cache = PromptCache()
        assert cache.get("nonexistent") is None

    def test_get_returns_none_after_expiry(self):
        cache = PromptCache()
        cache.set("slug", **_sample_data(), ttl=0.0)
        assert cache.get("slug") is None

    def test_overwrite(self):
        cache = PromptCache()
        cache.set("slug", **_sample_data(), ttl=60.0)
        cache.set("slug", content="Updated", managed_prompt_id="mp1",
                  prompt_version_id="pv1", version=1, ttl=60.0)
        assert cache.get("slug").content == "Updated"

    def test_invalidate(self):
        cache = PromptCache()
        cache.set("a", **_sample_data(), ttl=60.0)
        cache.set("b", **_sample_data(), ttl=60.0)
        cache.invalidate("a")
        assert cache.get("a") is None
        assert cache.get("b") is not None

    def test_invalidate_all(self):
        cache = PromptCache()
        cache.set("a", **_sample_data(), ttl=60.0)
        cache.set("b", **_sample_data(), ttl=60.0)
        cache.invalidate_all()
        assert cache.get("a") is None
        assert cache.get("b") is None

    def test_get_stale(self):
        cache = PromptCache()
        cache.set("slug", **_sample_data(), ttl=0.0)
        assert cache.get("slug") is None
        stale = cache.get_stale("slug")
        assert stale is not None
        assert stale.content == "You are helpful"

    def test_size(self):
        cache = PromptCache()
        assert cache.size == 0
        cache.set("a", **_sample_data(), ttl=60.0)
        assert cache.size == 1
        cache.set("b", **_sample_data(), ttl=60.0)
        assert cache.size == 2
        cache.invalidate("a")
        assert cache.size == 1


class TestLRUEviction:
    def test_evicts_oldest(self):
        cache = PromptCache(max_size=3)
        for key in ("a", "b", "c"):
            cache.set(key, content=key.upper(), managed_prompt_id="mp",
                      prompt_version_id="pv", version=1, ttl=60.0)
        cache.set("d", content="D", managed_prompt_id="mp",
                  prompt_version_id="pv", version=1, ttl=60.0)
        assert cache.get("a") is None  # evicted
        assert cache.get("b").content == "B"
        assert cache.get("d").content == "D"
        assert cache.size == 3

    def test_get_refreshes_lru(self):
        cache = PromptCache(max_size=3)
        for key in ("a", "b", "c"):
            cache.set(key, content=key.upper(), managed_prompt_id="mp",
                      prompt_version_id="pv", version=1, ttl=60.0)
        cache.get("a")  # refresh a
        cache.set("d", content="D", managed_prompt_id="mp",
                  prompt_version_id="pv", version=1, ttl=60.0)
        assert cache.get("a").content == "A"  # still alive
        assert cache.get("b") is None  # evicted

    def test_overwrite_no_size_increase(self):
        cache = PromptCache(max_size=3)
        cache.set("a", **_sample_data(), ttl=60.0)
        cache.set("b", **_sample_data(), ttl=60.0)
        cache.set("a", content="Updated", managed_prompt_id="mp",
                  prompt_version_id="pv", version=1, ttl=60.0)
        assert cache.size == 2
        assert cache.get("a").content == "Updated"
