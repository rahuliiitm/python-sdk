"""
Topic/scope restriction module -- keyword/phrase-based topic matching.
Zero dependencies. Blocks off-topic or restricted content based on
configurable allowed/blocked topic definitions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Literal, Optional

TopicViolationType = Literal["off_topic", "blocked_topic"]


@dataclass
class TopicDefinition:
    name: str
    keywords: List[str]
    threshold: Optional[float] = None  # default: 0.05


@dataclass
class TopicGuardOptions:
    allowed_topics: Optional[List[TopicDefinition]] = None
    blocked_topics: Optional[List[TopicDefinition]] = None


@dataclass
class TopicViolation:
    type: TopicViolationType
    topic: Optional[str] = None  # name of matched blocked topic, or None for off_topic
    matched_keywords: List[str] = field(default_factory=list)
    score: float = 0.0


# -- Constants -----------------------------------------------------------------

_DEFAULT_THRESHOLD = 0.05
_TOKEN_SPLIT_RE = re.compile(r"[\s,.!?;:()\[\]{}\"\'\`]+")


# -- Internal helpers ----------------------------------------------------------

@dataclass
class _TopicScore:
    name: str
    matched_keywords: List[str]
    score: float


def _score_topic(
    tokens: List[str],
    lower_text: str,
    topic: TopicDefinition,
) -> _TopicScore:
    """Score a single topic definition against the input text.
    Single-word keywords are matched as exact tokens (counting each occurrence).
    Multi-word phrases are matched as substring inclusions on the full lowered text.
    """
    matched_keywords: List[str] = []
    matched_count = 0

    for keyword in topic.keywords:
        lower_keyword = keyword.lower()

        if " " in lower_keyword:
            # Multi-word phrase -- substring match against full text
            if lower_keyword in lower_text:
                matched_keywords.append(keyword)
                # Count the number of tokens in the keyword as matched
                keyword_tokens = [t for t in _TOKEN_SPLIT_RE.split(lower_keyword) if t]
                matched_count += len(keyword_tokens)
        else:
            # Single-word keyword -- exact token match (count each occurrence)
            for token in tokens:
                if token == lower_keyword:
                    matched_count += 1
                    if keyword not in matched_keywords:
                        matched_keywords.append(keyword)

    score = matched_count / len(tokens) if tokens else 0.0

    return _TopicScore(name=topic.name, matched_keywords=matched_keywords, score=score)


# -- Public API ----------------------------------------------------------------

def check_topic_guard(
    text: str,
    options: TopicGuardOptions,
) -> Optional[TopicViolation]:
    """Check text against topic guard rules.

    Returns None if the text passes (no violation).
    Returns a TopicViolation if the text is off-topic or matches a blocked topic.

    Evaluation order:
    1. allowedTopics: if configured and NO topic scores above threshold -> off_topic
    2. blockedTopics: if ANY topic scores above threshold -> blocked_topic
    3. If an allowed topic matches, blocked topics are skipped entirely.
    """
    # Empty text -> no violation
    if not text:
        return None

    has_allowed = bool(options.allowed_topics and len(options.allowed_topics) > 0)
    has_blocked = bool(options.blocked_topics and len(options.blocked_topics) > 0)

    # No topics configured -> no violation
    if not has_allowed and not has_blocked:
        return None

    lower_text = text.lower()
    tokens = [t for t in _TOKEN_SPLIT_RE.split(lower_text) if t]

    # No tokens -> no violation
    if not tokens:
        return None

    # -- Allowed topics check --------------------------------------------------

    if has_allowed:
        any_allowed_match = False

        for topic in options.allowed_topics:  # type: ignore[union-attr]
            threshold = topic.threshold if topic.threshold is not None else _DEFAULT_THRESHOLD
            result = _score_topic(tokens, lower_text, topic)

            if result.score >= threshold:
                any_allowed_match = True
                break

        if not any_allowed_match:
            return TopicViolation(
                type="off_topic",
                topic=None,
                matched_keywords=[],
                score=0.0,
            )

        # Allowed topic matched -- skip blocked topics check
        return None

    # -- Blocked topics check --------------------------------------------------

    if has_blocked:
        for topic in options.blocked_topics:  # type: ignore[union-attr]
            threshold = topic.threshold if topic.threshold is not None else _DEFAULT_THRESHOLD
            result = _score_topic(tokens, lower_text, topic)

            if result.score >= threshold:
                return TopicViolation(
                    type="blocked_topic",
                    topic=result.name,
                    matched_keywords=result.matched_keywords,
                    score=result.score,
                )

    return None
