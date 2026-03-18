"""
Context Engine — System prompt analysis and constraint extraction.
Parses a system prompt once, extracts structured constraints (topics, role,
forbidden actions, output format, knowledge boundaries), and caches the result.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Protocol, Tuple

# ── Types ────────────────────────────────────────────────────────────────────

ConstraintType = Literal[
    "topic_boundary",
    "role_constraint",
    "action_restriction",
    "output_format",
    "knowledge_boundary",
    "persona_rule",
    "negative_instruction",
]

GroundingMode = Literal["any", "documents_only", "system_only"]


@dataclass
class Constraint:
    type: ConstraintType
    description: str
    keywords: List[str]
    source: str
    confidence: float


@dataclass
class ContextProfile:
    role: Optional[str] = None
    entity: Optional[str] = None
    allowed_topics: List[str] = field(default_factory=list)
    restricted_topics: List[str] = field(default_factory=list)
    forbidden_actions: List[str] = field(default_factory=list)
    output_format: Optional[str] = None
    grounding_mode: GroundingMode = "any"
    constraints: List[Constraint] = field(default_factory=list)
    raw_system_prompt: str = ""
    prompt_hash: str = ""


@dataclass
class ContextEngineOptions:
    """Options for extractContext()."""

    cache: bool = True


class ContextExtractorProvider(Protocol):
    """Provider interface for pluggable context extractors (e.g., ML-based)."""

    def extract(self, system_prompt: str) -> ContextProfile: ...

    @property
    def name(self) -> str: ...


@dataclass
class ContextEngineSecurityOptions:
    """User-facing security options for the context engine."""

    enabled: Optional[bool] = None
    system_prompt: Optional[str] = None
    """Override system prompt — if omitted, auto-extracted from messages."""
    providers: Optional[List[ContextExtractorProvider]] = None
    """Pluggable ML extraction providers."""
    cache_profiles: Optional[bool] = None
    """Cache extracted profiles by prompt hash. Default: True."""
    enhance_injection: Optional[bool] = None
    """Feed extracted context into injection detection for enhanced accuracy. Default: True."""


# ── Cache ────────────────────────────────────────────────────────────────────

_profile_cache: Dict[str, ContextProfile] = {}


def clear_context_cache() -> None:
    """Clear the internal profile cache. Useful for testing."""
    _profile_cache.clear()


# ── Sentence Splitting ──────────────────────────────────────────────────────

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in parts if s.strip()]


# ── Keyword Extraction ──────────────────────────────────────────────────────

_STOP_WORDS = frozenset(
    "a an the is are was were be been being "
    "have has had do does did will would could "
    "should may might must shall can need dare "
    "to of in for on with at by from as "
    "into through during before after above below "
    "between out off over under again further then "
    "and but or nor not so yet both either "
    "neither each every all any few more most "
    "other some such no only own same than too "
    "very just because if when while where how "
    "what which who whom this that these those "
    "i me my myself we our ours ourselves "
    "you your yours yourself yourselves "
    "he him his she her hers it its "
    "they them their theirs themselves "
    "about up also well back even still "
    "never always ever already now".split()
)

_TOKEN_SPLIT_RE = re.compile(r'[\s,.!?;:()\[\]{}"\']+')


def _extract_keywords(text: str) -> List[str]:
    return [
        t
        for t in _TOKEN_SPLIT_RE.split(text.lower())
        if len(t) > 2 and t not in _STOP_WORDS
    ]


# ── Extraction Patterns ─────────────────────────────────────────────────────

_ROLE_PATTERNS = [
    re.compile(r"you\s+are\s+(?:a|an)\s+([^.,:;!?\n]{3,60})", re.I),
    re.compile(r"act\s+as\s+(?:a|an)?\s*([^.,:;!?\n]{3,60})", re.I),
    re.compile(r"your\s+role\s+is\s+(?:to\s+)?([^.,:;!?\n]{3,60})", re.I),
    re.compile(r"behave\s+as\s+(?:a|an)?\s*([^.,:;!?\n]{3,60})", re.I),
    re.compile(r"you\s+(?:will\s+)?serve\s+as\s+(?:a|an)?\s*([^.,:;!?\n]{3,60})", re.I),
]

_ENTITY_PATTERNS = [
    re.compile(r"you\s+are\s+([A-Z][A-Za-z0-9\s&'\-]{1,40}?)(?:'s|'s)\s+", re.I),
    re.compile(r"you\s+work\s+for\s+([A-Z][A-Za-z0-9\s&'\-]{1,40})", re.I),
    re.compile(r"you\s+represent\s+([A-Z][A-Za-z0-9\s&'\-]{1,40})", re.I),
    re.compile(r"(?:built|created|developed|made)\s+by\s+([A-Z][A-Za-z0-9\s&'\-]{1,40})", re.I),
]

_ALLOWED_TOPIC_PATTERNS = [
    re.compile(r"only\s+(?:discuss|talk\s+about|answer\s+(?:questions?\s+)?(?:about|regarding|related\s+to))\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"(?:limit|restrict|confine)\s+(?:yourself|your\s+(?:responses?|answers?|discussions?))\s+to\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"you\s+(?:should\s+)?only\s+(?:respond|help)\s+(?:about|with|regarding)\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"your\s+(?:scope|domain|area)\s+is\s+(?:limited\s+to\s+)?([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"focus\s+(?:exclusively\s+)?on\s+([^.,:;!?\n]{3,80})", re.I),
]

_RESTRICTED_TOPIC_PATTERNS = [
    re.compile(r"(?:never|don'?t|do\s+not|avoid)\s+(?:discuss|talk\s+about|mention|bring\s+up|address)\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"(?:do\s+not|don'?t|never)\s+provide\s+(?:(?:\w+\s+)?(?:advice|guidance|recommendations?|information))\s+(?:on|about|regarding|related\s+to)\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"(?:never|don'?t|do\s+not)\s+provide\s+([^.,:;!?\n]*?(?:advice|guidance|recommendations?|information)[^.,:;!?\n]{0,40})", re.I),
    re.compile(r"(?:stay\s+away|steer\s+clear)\s+(?:from|of)\s+(?:topics?\s+(?:like|such\s+as|including)\s+)?([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"(?:off[- ]limits?|forbidden|prohibited)\s*(?:topics?)?\s*(?:include|are)?\s*:?\s*([^.!?\n]{3,80})", re.I),
]

_FORBIDDEN_ACTION_PATTERNS = [
    re.compile(r"(?:never|don'?t|do\s+not)\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"under\s+no\s+circumstances?\s+(?:should\s+you\s+)?([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"you\s+(?:must|should)\s+(?:not|never)\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"(?:refrain|abstain)\s+from\s+([^.,:;!?\n]{3,80})", re.I),
    re.compile(r"it\s+is\s+(?:strictly\s+)?(?:forbidden|prohibited)\s+(?:to|for\s+you\s+to)\s+([^.,:;!?\n]{3,80})", re.I),
]

_OUTPUT_FORMAT_PATTERNS = [
    re.compile(r"(?:always\s+)?respond\s+(?:in|using|with)\s+(JSON|json|XML|xml|markdown|Markdown|YAML|yaml|HTML|html|plain\s+text|csv|CSV)", re.I),
    re.compile(r"(?:format|structure)\s+(?:your\s+)?(?:responses?|answers?|output)\s+(?:as|in|using)\s+(JSON|json|XML|xml|markdown|Markdown|YAML|yaml|HTML|html|plain\s+text|csv|CSV)", re.I),
    re.compile(r"(?:output|return|give)\s+(?:only\s+)?(?:valid\s+)?(JSON|json|XML|xml|YAML|yaml)", re.I),
    re.compile(r"responses?\s+(?:should|must)\s+be\s+(?:in\s+)?(JSON|json|XML|xml|markdown|Markdown|YAML|yaml|HTML|html|plain\s+text|csv|CSV)", re.I),
]

_KNOWLEDGE_BOUNDARY_PATTERNS = [
    re.compile(r"only\s+(?:answer|respond|use\s+information)\s+(?:from|based\s+on)\s+(?:the\s+)?(?:provided|given|supplied|attached)\s+(?:documents?|context|data|sources?|information)", re.I),
    re.compile(r"(?:do\s+not|don'?t|never)\s+(?:use|rely\s+on|include)\s+(?:external|outside|your\s+own)\s+(?:knowledge|information|data|sources?)", re.I),
    re.compile(r"(?:ground|base)\s+(?:your\s+)?(?:responses?|answers?)\s+(?:only\s+)?(?:on|in)\s+(?:the\s+)?(?:provided|given|supplied)\s+(?:context|documents?|data|information)", re.I),
    re.compile(r"(?:stick|adhere)\s+(?:strictly\s+)?to\s+(?:the\s+)?(?:provided|given|supplied)\s+(?:context|documents?|data|information)", re.I),
    re.compile(r"(?:ground|base)\s+(?:your\s+)?(?:responses?|answers?)\s+(?:only\s+)?(?:on|in)\s+(?:the\s+)?(?:system\s+)?(?:context|information)\s+provided", re.I),
]

_PERSONA_PATTERNS = [
    re.compile(r"(?:be|stay|remain)\s+(professional|friendly|polite|formal|casual|concise|brief|helpful|empathetic|neutral|objective|respectful|warm|enthusiastic)", re.I),
    re.compile(r"maintain\s+(?:a\s+)?(professional|friendly|polite|formal|casual|concise|brief|helpful|empathetic|neutral|objective|respectful|warm|enthusiastic)\s+(?:tone|demeanor|manner|attitude|style)", re.I),
    re.compile(r"(?:use|adopt|keep)\s+(?:a\s+)?(professional|friendly|polite|formal|casual|concise|brief|helpful|empathetic|neutral|objective|respectful|warm|enthusiastic)\s+(?:tone|voice|style)", re.I),
    re.compile(r"your\s+tone\s+(?:should|must)\s+be\s+(professional|friendly|polite|formal|casual|concise|brief|helpful|empathetic|neutral|objective|respectful|warm|enthusiastic)", re.I),
]

# ── Compound Constraint Splitting ──────────────────────────────────────────

_COMPOUND_DELIMITER_RE = re.compile(r",\s+(?:and|or)\s+|;\s+(?:and|or)\s+|;\s+|,\s+|\s+(?:and|or)\s+", re.I)


def _split_compound_items(text: str) -> List[str]:
    """Split compound constraints like 'politics, religion, or adult content'
    into individual items: ['politics', 'religion', 'adult content'].
    """
    items = _COMPOUND_DELIMITER_RE.split(text)
    return [s.strip() for s in items if s.strip()]


def _is_compound_text(text: str) -> bool:
    """Check if text contains list delimiters indicating multiple items."""
    return bool(re.search(r",\s+|\s+(?:and|or)\s+|;\s+", text, re.I))


# ── Extraction Functions ────────────────────────────────────────────────────


def _extract_role(sentences: List[str]) -> Optional[str]:
    for sentence in sentences:
        for pattern in _ROLE_PATTERNS:
            m = pattern.search(sentence)
            if m and m.group(1):
                return m.group(1).strip().lower()
    return None


def _extract_entity(text: str) -> Optional[str]:
    for pattern in _ENTITY_PATTERNS:
        m = pattern.search(text)
        if m and m.group(1):
            return m.group(1).strip()
    return None


def _extract_allowed_topics(
    sentences: List[str],
) -> Tuple[List[str], List[Constraint]]:
    topics: List[str] = []
    constraints: List[Constraint] = []
    for sentence in sentences:
        for pattern in _ALLOWED_TOPIC_PATTERNS:
            m = pattern.search(sentence)
            if m and m.group(1):
                raw_text = m.group(1).strip()
                items = _split_compound_items(raw_text) if _is_compound_text(raw_text) else [raw_text.lower()]

                for item in items:
                    item = item.lower()
                    if item not in topics:
                        topics.append(item)
                    constraints.append(
                        Constraint(
                            type="topic_boundary",
                            description=f"Allowed topic: {item}",
                            keywords=_extract_keywords(item),
                            source=sentence,
                            confidence=0.8,
                        )
                    )
    return topics, constraints


def _extract_restricted_topics(
    sentences: List[str],
) -> Tuple[List[str], List[Constraint]]:
    topics: List[str] = []
    constraints: List[Constraint] = []
    for sentence in sentences:
        for pattern in _RESTRICTED_TOPIC_PATTERNS:
            m = pattern.search(sentence)
            if m and m.group(1):
                raw_text = m.group(1).strip()
                items = _split_compound_items(raw_text) if _is_compound_text(raw_text) else [raw_text.lower()]

                for item in items:
                    item = item.lower()
                    if item not in topics:
                        topics.append(item)
                    constraints.append(
                        Constraint(
                            type="topic_boundary",
                            description=f"Restricted topic: {item}",
                            keywords=_extract_keywords(item),
                            source=sentence,
                            confidence=0.8,
                        )
                    )
    return topics, constraints


def _extract_forbidden_actions(
    sentences: List[str],
) -> Tuple[List[str], List[Constraint]]:
    actions: List[str] = []
    constraints: List[Constraint] = []
    negative_re = re.compile(
        r"(?:never|don'?t|do\s+not|must\s+not|should\s+not|under\s+no\s+circumstances?|refrain|abstain|forbidden|prohibited)",
        re.I,
    )
    for sentence in sentences:
        if not negative_re.search(sentence):
            continue
        for pattern in _FORBIDDEN_ACTION_PATTERNS:
            m = pattern.search(sentence)
            if m and m.group(1):
                raw_text = m.group(1).strip()
                items = _split_compound_items(raw_text) if _is_compound_text(raw_text) else [raw_text.lower()]

                for item in items:
                    item = item.lower()
                    if len(item) < 5:
                        continue
                    if item not in actions:
                        actions.append(item)
                    constraints.append(
                        Constraint(
                            type="action_restriction",
                            description=f"Forbidden: {item}",
                            keywords=_extract_keywords(item),
                            source=sentence,
                            confidence=0.75,
                        )
                    )
    return actions, constraints


def _extract_output_format(
    sentences: List[str],
) -> Tuple[Optional[str], List[Constraint]]:
    constraints: List[Constraint] = []
    for sentence in sentences:
        for pattern in _OUTPUT_FORMAT_PATTERNS:
            m = pattern.search(sentence)
            if m and m.group(1):
                fmt = m.group(1).strip().upper()
                constraints.append(
                    Constraint(
                        type="output_format",
                        description=f"Output format: {fmt}",
                        keywords=[fmt.lower()],
                        source=sentence,
                        confidence=0.9,
                    )
                )
                return fmt, constraints
    return None, constraints


def _extract_grounding_mode(
    sentences: List[str],
) -> Tuple[GroundingMode, List[Constraint]]:
    constraints: List[Constraint] = []
    for sentence in sentences:
        for pattern in _KNOWLEDGE_BOUNDARY_PATTERNS:
            if pattern.search(sentence):
                is_doc_only = bool(
                    re.search(
                        r"(?:provided|given|supplied|attached)\s+(?:documents?|context|sources?)",
                        sentence,
                        re.I,
                    )
                )
                mode: GroundingMode = "documents_only" if is_doc_only else "system_only"
                constraints.append(
                    Constraint(
                        type="knowledge_boundary",
                        description=f"Knowledge restricted to {'provided documents' if mode == 'documents_only' else 'system context'}",
                        keywords=_extract_keywords(sentence),
                        source=sentence,
                        confidence=0.85,
                    )
                )
                return mode, constraints
    return "any", constraints


def _extract_persona_rules(sentences: List[str]) -> List[Constraint]:
    constraints: List[Constraint] = []
    for sentence in sentences:
        for pattern in _PERSONA_PATTERNS:
            m = pattern.search(sentence)
            if m and m.group(1):
                trait = m.group(1).strip().lower()
                constraints.append(
                    Constraint(
                        type="persona_rule",
                        description=f"Persona: {trait}",
                        keywords=[trait],
                        source=sentence,
                        confidence=0.7,
                    )
                )
    return constraints


# ── Conflict Detection ────────────────────────────────────────────────────────


@dataclass
class ConstraintConflict:
    constraint_a: Constraint
    constraint_b: Constraint
    conflict_type: Literal["contradiction", "ambiguity", "redundancy"]
    description: str


_CONTRADICTORY_PERSONA_PAIRS: List[Tuple[str, str]] = [
    ("formal", "casual"),
    ("professional", "casual"),
    ("concise", "detailed"),
    ("brief", "detailed"),
    ("neutral", "enthusiastic"),
    ("objective", "empathetic"),
]

_EXPANSIVE_ROLE_INDICATORS = ["general", "any", "all", "everything", "anything", "universal"]


def detect_conflicts(profile: ContextProfile) -> List[ConstraintConflict]:
    """Detect logical conflicts between extracted constraints.

    Checks:
    - Allowed topic keywords that overlap with restricted topic keywords (contradiction)
    - Expansive role definitions alongside restrictive constraints (ambiguity)
    - Contradictory persona traits (contradiction)
    """
    conflicts: List[ConstraintConflict] = []

    allowed_constraints = [
        c for c in profile.constraints
        if c.type == "topic_boundary" and c.description.startswith("Allowed")
    ]
    restricted_constraints = [
        c for c in profile.constraints
        if c.type == "topic_boundary" and c.description.startswith("Restricted")
    ]
    persona_constraints = [c for c in profile.constraints if c.type == "persona_rule"]
    role_constraint = next((c for c in profile.constraints if c.type == "role_constraint"), None)

    # 1. Allowed vs restricted topic keyword overlap
    for allowed in allowed_constraints:
        for restricted in restricted_constraints:
            overlap = [k for k in allowed.keywords if k in restricted.keywords]
            if overlap:
                conflicts.append(ConstraintConflict(
                    constraint_a=allowed,
                    constraint_b=restricted,
                    conflict_type="contradiction",
                    description=(
                        f'Allowed topic "{allowed.description}" overlaps with restricted topic '
                        f'"{restricted.description}" on keywords: {", ".join(overlap)}'
                    ),
                ))

    # 2. Expansive role vs restrictive constraints
    if role_constraint and (restricted_constraints or profile.forbidden_actions):
        role_text = role_constraint.description.lower()
        is_expansive = any(ind in role_text for ind in _EXPANSIVE_ROLE_INDICATORS)
        if is_expansive:
            restrictive = restricted_constraints[0] if restricted_constraints else next(
                (c for c in profile.constraints if c.type == "action_restriction"), None
            )
            if restrictive:
                conflicts.append(ConstraintConflict(
                    constraint_a=role_constraint,
                    constraint_b=restrictive,
                    conflict_type="ambiguity",
                    description=(
                        f'Expansive role "{role_constraint.description}" may conflict with '
                        f'restrictive constraint "{restrictive.description}"'
                    ),
                ))

    # 3. Contradictory persona traits
    for i in range(len(persona_constraints)):
        for j in range(i + 1, len(persona_constraints)):
            trait_a = persona_constraints[i].keywords[0].lower() if persona_constraints[i].keywords else ""
            trait_b = persona_constraints[j].keywords[0].lower() if persona_constraints[j].keywords else ""
            if not trait_a or not trait_b:
                continue

            is_contradictory = any(
                (trait_a == a and trait_b == b) or (trait_a == b and trait_b == a)
                for a, b in _CONTRADICTORY_PERSONA_PAIRS
            )
            if is_contradictory:
                conflicts.append(ConstraintConflict(
                    constraint_a=persona_constraints[i],
                    constraint_b=persona_constraints[j],
                    conflict_type="contradiction",
                    description=f'Persona trait "{trait_a}" contradicts "{trait_b}"',
                ))

    return conflicts


# ── Public API ───────────────────────────────────────────────────────────────


def extract_context(
    system_prompt: str,
    options: Optional[ContextEngineOptions] = None,
) -> ContextProfile:
    """
    Extract a structured context profile from a system prompt.

    Parses the prompt into sentences and applies regex-based extraction for:
    - Role identification (e.g., "You are a customer support agent")
    - Entity/company detection (e.g., "You work for Acme Corp")
    - Allowed/restricted topics
    - Forbidden actions
    - Output format requirements
    - Knowledge boundaries (grounding mode)
    - Persona rules

    Results are cached by prompt hash — extraction runs once per unique prompt.
    """
    if not system_prompt or not system_prompt.strip():
        return ContextProfile(raw_system_prompt=system_prompt or "", prompt_hash="")

    prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()
    use_cache = options is None or options.cache

    if use_cache:
        cached = _profile_cache.get(prompt_hash)
        if cached is not None:
            return cached

    sentences = _split_sentences(system_prompt)
    all_constraints: List[Constraint] = []

    # Extract role
    role = _extract_role(sentences)
    if role:
        role_source = ""
        for s in sentences:
            if any(p.search(s) for p in _ROLE_PATTERNS):
                role_source = s
                break
        all_constraints.append(
            Constraint(
                type="role_constraint",
                description=f"Role: {role}",
                keywords=_extract_keywords(role),
                source=role_source,
                confidence=0.85,
            )
        )

    # Extract entity
    entity = _extract_entity(system_prompt)

    # Extract allowed topics
    allowed_topics, allowed_constraints = _extract_allowed_topics(sentences)
    all_constraints.extend(allowed_constraints)

    # Extract restricted topics
    restricted_topics, restricted_constraints = _extract_restricted_topics(sentences)
    all_constraints.extend(restricted_constraints)

    # Extract forbidden actions
    forbidden_actions, forbidden_constraints = _extract_forbidden_actions(sentences)
    all_constraints.extend(forbidden_constraints)

    # Extract output format
    output_format, format_constraints = _extract_output_format(sentences)
    all_constraints.extend(format_constraints)

    # Extract grounding mode
    grounding_mode, grounding_constraints = _extract_grounding_mode(sentences)
    all_constraints.extend(grounding_constraints)

    # Extract persona rules
    persona_constraints = _extract_persona_rules(sentences)
    all_constraints.extend(persona_constraints)

    profile = ContextProfile(
        role=role,
        entity=entity,
        allowed_topics=allowed_topics,
        restricted_topics=restricted_topics,
        forbidden_actions=forbidden_actions,
        output_format=output_format,
        grounding_mode=grounding_mode,
        constraints=all_constraints,
        raw_system_prompt=system_prompt,
        prompt_hash=prompt_hash,
    )

    if use_cache:
        _profile_cache[prompt_hash] = profile

    return profile


def extract_context_with_providers(
    system_prompt: str,
    providers: List["ContextExtractorProvider"],
    options: Optional[ContextEngineOptions] = None,
) -> ContextProfile:
    """Extract context using regex baseline + optional ML providers.

    Runs ``extract_context()`` for the regex baseline, then calls each
    provider's ``extract()`` synchronously. Results are merged: regex findings
    are preferred for explicitly matched patterns, ML findings fill gaps.
    """
    regex_profile = extract_context(system_prompt, options)
    if not providers:
        return regex_profile

    ml_profiles = [p.extract(system_prompt) for p in providers]
    return _merge_profiles(regex_profile, ml_profiles)


def _merge_profiles(
    regex_profile: ContextProfile,
    ml_profiles: List[ContextProfile],
) -> ContextProfile:
    """Merge regex profile with ML provider profiles.

    Strategy:
    - Role/entity/output_format/grounding_mode: prefer regex, ML fallback
    - Topics/actions: union (deduplicated)
    - Constraints: merge all, dedup by source+type, keep higher confidence
    """
    # Start from copies of regex lists
    role = regex_profile.role
    entity = regex_profile.entity
    output_format = regex_profile.output_format
    grounding_mode = regex_profile.grounding_mode
    allowed_topics = list(regex_profile.allowed_topics)
    restricted_topics = list(regex_profile.restricted_topics)
    forbidden_actions = list(regex_profile.forbidden_actions)
    constraints = list(regex_profile.constraints)

    for ml in ml_profiles:
        # Scalars: regex wins if present, otherwise take ML
        if not role and ml.role:
            role = ml.role
        if not entity and ml.entity:
            entity = ml.entity
        if not output_format and ml.output_format:
            output_format = ml.output_format
        if grounding_mode == "any" and ml.grounding_mode != "any":
            grounding_mode = ml.grounding_mode

        # Topics: union with deduplication
        for topic in ml.allowed_topics:
            if topic not in allowed_topics:
                allowed_topics.append(topic)
        for topic in ml.restricted_topics:
            if topic not in restricted_topics:
                restricted_topics.append(topic)

        # Forbidden actions: union
        for action in ml.forbidden_actions:
            if action not in forbidden_actions:
                forbidden_actions.append(action)

        # Constraints: merge by source+type, keep higher confidence
        for ml_c in ml.constraints:
            existing = next(
                (c for c in constraints if c.type == ml_c.type and c.source == ml_c.source),
                None,
            )
            if existing:
                if ml_c.confidence > existing.confidence:
                    existing.confidence = ml_c.confidence
                    existing.description = ml_c.description
                    existing.keywords = ml_c.keywords
            else:
                constraints.append(ml_c)

    return ContextProfile(
        role=role,
        entity=entity,
        allowed_topics=allowed_topics,
        restricted_topics=restricted_topics,
        forbidden_actions=forbidden_actions,
        output_format=output_format,
        grounding_mode=grounding_mode,
        constraints=constraints,
        raw_system_prompt=regex_profile.raw_system_prompt,
        prompt_hash=regex_profile.prompt_hash,
    )
