"""
System prompt leakage detection module.
Output-side scanning to detect when an LLM response contains
fragments of the system prompt.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional

MAX_SCAN_LENGTH = 1024 * 1024  # 1MB


@dataclass
class PromptLeakageOptions:
    """Options for prompt leakage detection."""

    system_prompt: str = ""
    threshold: Optional[float] = None  # default: 0.4
    block_on_leak: Optional[bool] = None  # default: False


@dataclass
class PromptLeakageResult:
    """Result of prompt leakage detection."""

    leaked: bool
    similarity: float
    matched_fragments: List[str]
    meta_response_detected: bool


# -- N-gram helpers ------------------------------------------------------------

_NGRAM_SIZE = 4
_PUNCTUATION_RE = re.compile(r"[^\w\s]")
_WHITESPACE_RE = re.compile(r"\s+")


def _build_ngrams(text: str, n: int) -> List[str]:
    """Split text into sliding-window n-grams of `n` words.
    Returns lowercase, trimmed n-gram strings.
    """
    lower = text.lower()
    cleaned = _PUNCTUATION_RE.sub(" ", lower)
    words = [w for w in _WHITESPACE_RE.split(cleaned) if w]

    if len(words) < n:
        return []

    ngrams: List[str] = []
    for i in range(len(words) - n + 1):
        ngrams.append(" ".join(words[i : i + n]))
    return ngrams


# -- Meta-response patterns ----------------------------------------------------

_META_RESPONSE_PATTERNS: List[re.Pattern[str]] = [
    re.compile(r"my\s+instructions\s+are", re.IGNORECASE),
    re.compile(r"I\s+was\s+told\s+to", re.IGNORECASE),
    re.compile(r"my\s+system\s+prompt\s+is", re.IGNORECASE),
    re.compile(r"I(?:'m|\u2019m| am)\s+programmed\s+to", re.IGNORECASE),
    re.compile(r"according\s+to\s+my\s+instructions", re.IGNORECASE),
    re.compile(r"my\s+initial\s+instructions\s+were", re.IGNORECASE),
    re.compile(r"I\s+cannot\s+reveal\s+my\s+instructions", re.IGNORECASE),
    re.compile(r"my\s+rules\s+are", re.IGNORECASE),
    re.compile(r"my\s+guidelines\s+state", re.IGNORECASE),
]


# -- Verbatim substring detection ---------------------------------------------

_MIN_VERBATIM_LENGTH = 30


def _longest_common_substring_length(a: str, b: str) -> int:
    """Find the length of the longest common substring between two strings.
    Uses a sliding window approach bounded by MIN_VERBATIM_LENGTH to
    keep runtime reasonable on large inputs.
    """
    if len(a) < _MIN_VERBATIM_LENGTH or len(b) < _MIN_VERBATIM_LENGTH:
        return 0

    max_len = 0

    # Check substrings of the shorter string in the longer string
    shorter = a if len(a) <= len(b) else b
    longer = a if len(a) > len(b) else b

    for i in range(len(shorter) - _MIN_VERBATIM_LENGTH + 1):
        # Start with minimum length and extend while matching
        end = i + _MIN_VERBATIM_LENGTH
        candidate = shorter[i:end]
        if candidate in longer:
            # Extend the match as far as possible
            while end < len(shorter) and shorter[i : end + 1] in longer:
                end += 1
            match_len = end - i
            if match_len > max_len:
                max_len = match_len

    return max_len


# -- Detection ----------------------------------------------------------------


def detect_prompt_leakage(
    output_text: str,
    options: PromptLeakageOptions,
) -> PromptLeakageResult:
    """Detect whether an LLM output contains fragments of the system prompt.

    Uses three complementary methods:
    1. N-gram overlap -- fraction of system prompt 4-word n-grams found in output
    2. Meta-response patterns -- regex matches for self-referential phrases
    3. Verbatim substring -- longest common substring > 30 characters
    """
    clean_result = PromptLeakageResult(
        leaked=False,
        similarity=0.0,
        matched_fragments=[],
        meta_response_detected=False,
    )

    if not output_text or not options.system_prompt:
        return clean_result

    threshold = options.threshold if options.threshold is not None else 0.4

    # Cap input lengths to prevent DoS
    scan_output = output_text[:MAX_SCAN_LENGTH] if len(output_text) > MAX_SCAN_LENGTH else output_text
    scan_prompt = options.system_prompt[:MAX_SCAN_LENGTH] if len(options.system_prompt) > MAX_SCAN_LENGTH else options.system_prompt

    # -- 1. N-gram overlap -----------------------------------------------------
    prompt_ngrams = _build_ngrams(scan_prompt, _NGRAM_SIZE)
    matched_fragments: List[str] = []
    similarity = 0.0

    if prompt_ngrams:
        output_lower = scan_output.lower()
        for ngram in prompt_ngrams:
            if ngram in output_lower:
                matched_fragments.append(ngram)
        similarity = len(matched_fragments) / len(prompt_ngrams)
        # Round to 4 decimal places for clean output
        similarity = round(similarity * 10000) / 10000

    # -- 2. Meta-response patterns ---------------------------------------------
    meta_response_detected = False
    for pattern in _META_RESPONSE_PATTERNS:
        if pattern.search(scan_output):
            meta_response_detected = True
            break

    # -- 3. Verbatim substring -------------------------------------------------
    verbatim_len = _longest_common_substring_length(
        scan_prompt.lower(),
        scan_output.lower(),
    )
    has_verbatim_match = verbatim_len >= _MIN_VERBATIM_LENGTH

    # -- Final verdict ---------------------------------------------------------
    leaked = similarity >= threshold or has_verbatim_match or meta_response_detected

    return PromptLeakageResult(
        leaked=leaked,
        similarity=similarity,
        matched_fragments=matched_fragments,
        meta_response_detected=meta_response_detected,
    )
