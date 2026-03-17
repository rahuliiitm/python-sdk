"""
Attack pattern embedding index -- cosine similarity search against
a curated corpus of ~180 known attack patterns.

Patterns are loaded from ``data/attack-patterns.json`` and embedded
once on first use via the shared MLEmbeddingProvider.

Usage::

    index = load_attack_index(embedding_provider)
    matches = match_against_index(input_embedding, index, threshold=0.75)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np

AttackCategory = Literal[
    "injection",
    "jailbreak",
    "data_extraction",
    "manipulation",
    "role_escape",
    "social_engineering",
]


@dataclass
class AttackMatch:
    category: str
    pattern: str
    similarity: float


@dataclass
class AttackCategoryData:
    patterns: List[str]
    embeddings: List[Any]  # list of np.ndarray


@dataclass
class AttackEmbeddingIndex:
    version: str
    model: str
    dimension: int
    categories: Dict[str, AttackCategoryData] = field(default_factory=dict)


# ── Singleton cache ────────────────────────────────────────────────────────

_cached_index: Optional[AttackEmbeddingIndex] = None
_DATA_PATH = Path(__file__).parent / "data" / "attack-patterns.json"


# ── Public API ─────────────────────────────────────────────────────────────


def load_attack_index(provider: Any) -> AttackEmbeddingIndex:
    """Load (or return cached) the attack pattern embedding index.

    Embeds all patterns on first call using the provided embedding provider.
    """
    global _cached_index
    if _cached_index is not None:
        return _cached_index

    with open(_DATA_PATH, "r") as f:
        raw = json.load(f)

    categories: Dict[str, AttackCategoryData] = {}
    dimension = 0

    for category, patterns in raw["categories"].items():
        embeddings = provider.embed_batch(patterns)
        if embeddings:
            dimension = len(embeddings[0])
        categories[category] = AttackCategoryData(
            patterns=patterns,
            embeddings=embeddings,
        )

    _cached_index = AttackEmbeddingIndex(
        version=raw["version"],
        model=raw["model"],
        dimension=dimension,
        categories=categories,
    )
    return _cached_index


def match_against_index(
    embedding: Any,
    index: AttackEmbeddingIndex,
    threshold: float = 0.75,
) -> List[AttackMatch]:
    """Find attack patterns most similar to the given embedding.

    Returns matches above the threshold, sorted by similarity (descending).
    """
    matches: List[AttackMatch] = []

    for category, data in index.categories.items():
        for i, pattern_emb in enumerate(data.embeddings):
            sim = _cosine(embedding, pattern_emb)
            if sim >= threshold:
                matches.append(AttackMatch(
                    category=category,
                    pattern=data.patterns[i],
                    similarity=sim,
                ))

    matches.sort(key=lambda m: m.similarity, reverse=True)
    return matches


def has_attack_match(
    embedding: Any,
    index: AttackEmbeddingIndex,
    threshold: float = 0.75,
) -> bool:
    """Quick check: does the input match any attack pattern above threshold?"""
    for data in index.categories.values():
        for pattern_emb in data.embeddings:
            if _cosine(embedding, pattern_emb) >= threshold:
                return True
    return False


def reset_cache() -> None:
    """Reset cached index (for testing)."""
    global _cached_index
    _cached_index = None


# ── Helpers ────────────────────────────────────────────────────────────────


def _cosine(a: Any, b: Any) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    dot = float(np.dot(a, b))
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    denom = norm_a * norm_b
    return dot / denom if denom > 0 else 0.0
