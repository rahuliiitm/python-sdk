"""
Hallucination detection module -- detects unfaithful LLM responses.

Uses n-gram overlap heuristics as a rule-based baseline.
ML providers (e.g., vectara/HHEM cross-encoder) can be plugged in via the providers pattern.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Literal, Optional, Protocol


@dataclass
class HallucinationResult:
    """Result of hallucination detection."""

    hallucinated: bool
    """Whether hallucination was detected (faithfulness below threshold)."""
    faithfulness_score: float
    """Faithfulness score 0-1 from detection."""
    severity: Literal["low", "medium", "high"]
    """Severity classification based on score."""


class HallucinationDetectorProvider(Protocol):
    """Provider interface for pluggable hallucination detectors."""

    def detect(self, generated: str, source: str) -> HallucinationResult: ...

    @property
    def name(self) -> str: ...


def _normalize_tokens(text: str) -> List[str]:
    return re.sub(r"[^\w\s]", "", text.lower()).split()


def _ngram_overlap(generated: str, source: str, n: int) -> float:
    """Compute n-gram overlap between generated text and source text.

    Returns a score between 0 and 1 (1 = fully faithful).
    """
    gen_tokens = _normalize_tokens(generated)
    src_tokens = _normalize_tokens(source)

    if len(gen_tokens) < n or len(src_tokens) < n:
        return 1.0 if len(gen_tokens) == 0 else 0.5

    src_ngrams: set[str] = set()
    for i in range(len(src_tokens) - n + 1):
        src_ngrams.add(" ".join(src_tokens[i : i + n]))

    matches = 0
    gen_ngram_count = len(gen_tokens) - n + 1
    for i in range(gen_ngram_count):
        ngram = " ".join(gen_tokens[i : i + n])
        if ngram in src_ngrams:
            matches += 1

    return matches / gen_ngram_count if gen_ngram_count > 0 else 0.0


def detect_hallucination(
    generated: str,
    source: str,
    threshold: float = 0.5,
) -> HallucinationResult:
    """Rule-based hallucination detection using multi-level n-gram overlap.

    Computes a blended faithfulness score from unigram, bigram, and trigram overlap.
    This is a rough heuristic -- ML providers (e.g., HHEM) are far more accurate.
    """
    if not generated or not source:
        return HallucinationResult(
            hallucinated=False, faithfulness_score=1.0, severity="low"
        )

    # Blended n-gram overlap: unigram (40%), bigram (35%), trigram (25%)
    uni = _ngram_overlap(generated, source, 1)
    bi = _ngram_overlap(generated, source, 2)
    tri = _ngram_overlap(generated, source, 3)
    score = round(uni * 0.4 + bi * 0.35 + tri * 0.25, 2)

    hallucinated = score < threshold

    if score >= 0.7:
        severity: Literal["low", "medium", "high"] = "low"
    elif score >= 0.4:
        severity = "medium"
    else:
        severity = "high"

    return HallucinationResult(
        hallucinated=hallucinated, faithfulness_score=score, severity=severity
    )


def merge_hallucination_results(
    results: List[HallucinationResult],
) -> HallucinationResult:
    """Merge multiple hallucination results using minimum faithfulness score."""
    if not results:
        return HallucinationResult(
            hallucinated=False, faithfulness_score=1.0, severity="low"
        )

    min_score = min(r.faithfulness_score for r in results)
    min_score = round(min_score, 2)

    if min_score >= 0.7:
        severity: Literal["low", "medium", "high"] = "low"
    elif min_score >= 0.4:
        severity = "medium"
    else:
        severity = "high"

    return HallucinationResult(
        hallucinated=any(r.hallucinated for r in results),
        faithfulness_score=min_score,
        severity=severity,
    )
