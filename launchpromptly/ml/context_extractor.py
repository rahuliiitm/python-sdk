"""
ML-based context extractor using sentence embeddings.

Classifies system prompt sentences into constraint types via cosine
similarity against pre-embedded templates. Catches implicit constraints
that regex patterns miss (e.g., "Keep conversations family-friendly" ->
persona_rule + implicit restricted topics).

Uses the shared MLEmbeddingProvider (all-MiniLM-L6-v2, 384-dim).
Designed to run alongside regex extraction -- results are merged in
extract_context_with_providers().

Requires::

    pip install onnxruntime tokenizers

Or simply::

    pip install launchpromptly[ml-onnx]
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .embedding_provider import MLEmbeddingProvider
from .._internal.context_engine import (
    ContextProfile,
    Constraint,
    ConstraintType,
    GroundingMode,
)

# -- Configuration ----------------------------------------------------------------

CLASSIFICATION_THRESHOLD = 0.45
ML_CONFIDENCE = 0.65

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?\n])\s+")

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

_TOKEN_SPLIT_RE = re.compile(r"[\s,.!?;:()\[\]{}'\"]+")


# -- Constraint type templates ---------------------------------------------------

@dataclass
class _ConstraintTemplateSpec:
    text: str
    type: ConstraintType
    subtype: Optional[str] = None  # 'allowed' | 'restricted'


_CONSTRAINT_TEMPLATE_SPECS: List[_ConstraintTemplateSpec] = [
    _ConstraintTemplateSpec(
        text="The assistant must play a specific role, character, or professional identity",
        type="role_constraint",
    ),
    _ConstraintTemplateSpec(
        text="The assistant is restricted to discussing only certain permitted topics",
        type="topic_boundary",
        subtype="allowed",
    ),
    _ConstraintTemplateSpec(
        text="The assistant is prohibited from discussing certain topics",
        type="topic_boundary",
        subtype="restricted",
    ),
    _ConstraintTemplateSpec(
        text="The assistant must never perform certain forbidden actions or behaviors",
        type="action_restriction",
    ),
    _ConstraintTemplateSpec(
        text="The response must always be in a specific output format like JSON or markdown",
        type="output_format",
    ),
    _ConstraintTemplateSpec(
        text="The assistant must only use information from provided documents and not external knowledge",
        type="knowledge_boundary",
    ),
    _ConstraintTemplateSpec(
        text="The assistant must maintain a specific personality trait, tone, or communication style",
        type="persona_rule",
    ),
]

_NEGATIVE_INDICATORS = re.compile(
    r"\b(?:never|don'?t|do\s+not|must\s+not|should\s+not|cannot|can'?t|"
    r"forbidden|prohibited|not\s+allowed|refrain|avoid|stay\s+away)\b",
    re.IGNORECASE,
)

_POSITIVE_SCOPE_INDICATORS = re.compile(
    r"\b(?:only|limited\s+to|restricted\s+to|exclusively|solely|focus\s+on|confine|stick\s+to)\b",
    re.IGNORECASE,
)


# -- Helpers ---------------------------------------------------------------------

def _split_sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def _extract_keywords(text: str) -> List[str]:
    return [
        t
        for t in _TOKEN_SPLIT_RE.split(text.lower())
        if len(t) > 2 and t not in _STOP_WORDS
    ]


def _empty_profile(system_prompt: str) -> ContextProfile:
    return ContextProfile(raw_system_prompt=system_prompt or "")


@dataclass
class _ConstraintTemplate:
    text: str
    type: ConstraintType
    subtype: Optional[str]
    embedding: np.ndarray


# -- MLContextExtractor ----------------------------------------------------------


class MLContextExtractor:
    """ML-based context extractor using sentence embeddings for zero-shot
    constraint classification.

    Implements the ``ContextExtractorProvider`` protocol -- drop-in alongside
    regex extraction. Results are merged in ``extract_context_with_providers()``.

    Example::

        from launchpromptly.ml import MLContextExtractor
        extractor = MLContextExtractor.create()
        # Register via security options:
        security={"context_engine": {"providers": [extractor]}}
    """

    name = "ml-context-extractor"

    def __init__(
        self,
        embedding: MLEmbeddingProvider,
        templates: List[_ConstraintTemplate],
        threshold: float = CLASSIFICATION_THRESHOLD,
    ) -> None:
        self._embedding = embedding
        self._templates = templates
        self._threshold = threshold

    @classmethod
    def create(
        cls,
        *,
        cache_dir=None,
        embedding_provider: Optional[MLEmbeddingProvider] = None,
        threshold: float = CLASSIFICATION_THRESHOLD,
    ) -> "MLContextExtractor":
        """Create an MLContextExtractor by loading the embedding model and
        pre-embedding all constraint type templates."""
        embedding = embedding_provider or MLEmbeddingProvider.create(cache_dir=cache_dir)

        templates: List[_ConstraintTemplate] = []
        for spec in _CONSTRAINT_TEMPLATE_SPECS:
            emb = embedding.embed(spec.text)
            templates.append(
                _ConstraintTemplate(
                    text=spec.text,
                    type=spec.type,
                    subtype=spec.subtype,
                    embedding=emb,
                )
            )

        return cls(embedding, templates, threshold)

    @classmethod
    def _create_for_test(
        cls,
        embedding: MLEmbeddingProvider,
        threshold: float = CLASSIFICATION_THRESHOLD,
    ) -> "MLContextExtractor":
        """For testing: create with pre-built embedding provider."""
        templates: List[_ConstraintTemplate] = []
        for spec in _CONSTRAINT_TEMPLATE_SPECS:
            emb = embedding.embed(spec.text)
            templates.append(
                _ConstraintTemplate(
                    text=spec.text,
                    type=spec.type,
                    subtype=spec.subtype,
                    embedding=emb,
                )
            )
        return cls(embedding, templates, threshold)

    @property
    def embedding_provider(self) -> MLEmbeddingProvider:
        """Get the shared embedding provider (for reuse by other guards)."""
        return self._embedding

    def extract(self, system_prompt: str) -> ContextProfile:
        """Extract a context profile from a system prompt using ML.

        Classifies each sentence against constraint type templates via
        embedding cosine similarity. Returns an independent profile that
        gets merged with regex results in extract_context_with_providers().
        """
        if not system_prompt or not system_prompt.strip():
            return _empty_profile(system_prompt)

        sentences = _split_sentences(system_prompt)
        if not sentences:
            return _empty_profile(system_prompt)

        # Embed all sentences in one batch
        sentence_embeddings = self._embedding.embed_batch(sentences)

        constraints: List[Constraint] = []
        allowed_topics: List[str] = []
        restricted_topics: List[str] = []
        forbidden_actions: List[str] = []
        role: Optional[str] = None
        output_format: Optional[str] = None
        grounding_mode: GroundingMode = "any"

        for i, sentence in enumerate(sentences):
            sentence_emb = sentence_embeddings[i]
            match = self._classify_sentence(sentence_emb)
            if match is None:
                continue

            match_type, subtype, score = match
            keywords = _extract_keywords(sentence)
            confidence = min(score * (ML_CONFIDENCE / CLASSIFICATION_THRESHOLD), 0.9)

            if match_type == "role_constraint":
                if role is None:
                    role = self._extract_role_text(sentence)
                constraints.append(
                    Constraint(
                        type="role_constraint",
                        description=f"Role: {role or sentence}",
                        keywords=keywords,
                        source=sentence,
                        confidence=confidence,
                    )
                )

            elif match_type == "topic_boundary":
                topic_text = self._extract_topic_text(sentence)
                is_negative = bool(_NEGATIVE_INDICATORS.search(sentence))
                is_positive_scope = bool(_POSITIVE_SCOPE_INDICATORS.search(sentence))

                if subtype == "restricted" or (is_negative and subtype != "allowed"):
                    if topic_text not in restricted_topics:
                        restricted_topics.append(topic_text)
                    constraints.append(
                        Constraint(
                            type="topic_boundary",
                            description=f"Restricted topic: {topic_text}",
                            keywords=keywords,
                            source=sentence,
                            confidence=confidence,
                        )
                    )
                elif subtype == "allowed" or is_positive_scope:
                    if topic_text not in allowed_topics:
                        allowed_topics.append(topic_text)
                    constraints.append(
                        Constraint(
                            type="topic_boundary",
                            description=f"Allowed topic: {topic_text}",
                            keywords=keywords,
                            source=sentence,
                            confidence=confidence,
                        )
                    )

            elif match_type == "action_restriction":
                action_text = self._extract_action_text(sentence)
                if len(action_text) >= 5 and action_text not in forbidden_actions:
                    forbidden_actions.append(action_text)
                constraints.append(
                    Constraint(
                        type="action_restriction",
                        description=f"Forbidden: {action_text}",
                        keywords=keywords,
                        source=sentence,
                        confidence=confidence,
                    )
                )

            elif match_type == "output_format":
                if output_format is None:
                    output_format = self._extract_format_text(sentence)
                constraints.append(
                    Constraint(
                        type="output_format",
                        description=f"Output format: {output_format}",
                        keywords=[output_format.lower()],
                        source=sentence,
                        confidence=confidence,
                    )
                )

            elif match_type == "knowledge_boundary":
                if re.search(
                    r"(?:provided|given|supplied|attached)\s+(?:documents?|context|sources?)",
                    sentence,
                    re.IGNORECASE,
                ):
                    grounding_mode = "documents_only"
                else:
                    grounding_mode = "system_only"
                constraints.append(
                    Constraint(
                        type="knowledge_boundary",
                        description=f"Knowledge restricted to {'provided documents' if grounding_mode == 'documents_only' else 'system context'}",
                        keywords=keywords,
                        source=sentence,
                        confidence=confidence,
                    )
                )

            elif match_type == "persona_rule":
                trait = self._extract_persona_trait(sentence)
                constraints.append(
                    Constraint(
                        type="persona_rule",
                        description=f"Persona: {trait}",
                        keywords=[trait],
                        source=sentence,
                        confidence=confidence,
                    )
                )

        return ContextProfile(
            role=role,
            entity=None,
            allowed_topics=allowed_topics,
            restricted_topics=restricted_topics,
            forbidden_actions=forbidden_actions,
            output_format=output_format,
            grounding_mode=grounding_mode,
            constraints=constraints,
            raw_system_prompt=system_prompt,
            prompt_hash="",
        )

    # -- Private helpers ----------------------------------------------------------

    def _classify_sentence(
        self, sentence_emb: np.ndarray
    ) -> Optional[Tuple[ConstraintType, Optional[str], float]]:
        """Classify a sentence embedding against all templates.
        Returns (type, subtype, score) or None if below threshold."""
        best_template: Optional[_ConstraintTemplate] = None
        best_score = 0.0

        for template in self._templates:
            sim = self._embedding.cosine(sentence_emb, template.embedding)
            if sim > best_score:
                best_score = sim
                best_template = template

        if best_template is None or best_score < self._threshold:
            return None

        return best_template.type, best_template.subtype, best_score

    @staticmethod
    def _extract_role_text(sentence: str) -> str:
        m = re.search(
            r"(?:you\s+are\s+(?:a|an)\s+|act\s+as\s+(?:a|an)?\s*|your\s+role\s+is\s+(?:to\s+)?"
            r"|behave\s+as\s+(?:a|an)?\s*|you\s+(?:will\s+)?serve\s+as\s+(?:a|an)?\s*)(.+)",
            sentence,
            re.IGNORECASE,
        )
        return (m.group(1) if m else sentence).strip().lower().rstrip(".!?")

    @staticmethod
    def _extract_topic_text(sentence: str) -> str:
        m = re.search(
            r"(?:about|regarding|related\s+to|on\s+the\s+(?:topic|subject)\s+of"
            r"|discuss(?:ing)?|conversations?\s+about)\s+(.+)",
            sentence,
            re.IGNORECASE,
        )
        if m:
            return m.group(1).strip().lower().rstrip(".!?")
        keywords = _extract_keywords(sentence)
        return " ".join(keywords[:4]) or sentence.lower().rstrip(".!?")

    @staticmethod
    def _extract_action_text(sentence: str) -> str:
        m = re.search(
            r"(?:never|don'?t|do\s+not|must\s+not|should\s+not|cannot)\s+(.+)",
            sentence,
            re.IGNORECASE,
        )
        return (m.group(1) if m else sentence).strip().lower().rstrip(".!?")

    @staticmethod
    def _extract_format_text(sentence: str) -> str:
        m = re.search(
            r"\b(JSON|XML|YAML|HTML|markdown|plain\s+text|CSV|bullet\s+points?"
            r"|numbered\s+list|paragraphs?|tables?)\b",
            sentence,
            re.IGNORECASE,
        )
        return (m.group(1).upper() if m else "STRUCTURED")

    @staticmethod
    def _extract_persona_trait(sentence: str) -> str:
        m = re.search(
            r"\b(professional|friendly|polite|formal|casual|concise|brief|helpful"
            r"|empathetic|neutral|objective|respectful|warm|enthusiastic|serious"
            r"|humorous|patient|assertive|family[- ]friendly|appropriate)\b",
            sentence,
            re.IGNORECASE,
        )
        return m.group(1).lower() if m else " ".join(_extract_keywords(sentence)[:2])
