"""
ML-based toxicity / content-safety detector.

Uses a small toxicity classifier for nuanced content detection.
Catches content that keyword rules miss.

Prefers onnxruntime + tokenizers for native inference (8-20ms).
Falls back to transformers + torch if onnxruntime is not installed.

Requires::

    pip install onnxruntime tokenizers   # recommended
    pip install launchpromptly[ml-onnx]  # same as above
"""
from __future__ import annotations

from typing import Any, List, Optional

from launchpromptly._internal.content_filter import (
    ContentLocation,
    ContentSeverity,
    ContentViolation,
)

# Mapping from toxic-bert labels to our ContentCategory strings.
_TOXICITY_LABEL_MAP: dict[str, str] = {
    "toxic": "hate_speech",
    "severe_toxic": "hate_speech",
    "obscene": "sexual",
    "threat": "violence",
    "insult": "hate_speech",
    "identity_hate": "hate_speech",
}

# Default confidence threshold below which a label is not reported.
_DEFAULT_THRESHOLD = 0.5

# Labels whose score at or above this level are considered "block" severity.
_BLOCK_THRESHOLD = 0.8


class MLToxicityDetector:
    """ML-based toxicity / content-safety detector.

    Uses the ``unitary/toxic-bert`` classifier by default for nuanced content
    detection that keyword rules miss.

    Tries ONNX Runtime first (8-20ms), falls back to transformers (500ms-2s).

    Example::

        from launchpromptly.ml import MLToxicityDetector
        detector = MLToxicityDetector()
        violations = detector.detect("You are a terrible person", "input")
    """

    def __init__(
        self,
        model_name: str = "unitary/toxic-bert",
        threshold: float = _DEFAULT_THRESHOLD,
        device: Optional[int] = None,
    ) -> None:
        self._model_name = model_name
        self._threshold = threshold

        # Try ONNX Runtime first (25-100x faster)
        use_onnx = False
        try:
            import onnxruntime  # noqa: F401
            from tokenizers import Tokenizer  # noqa: F401

            use_onnx = True
        except ImportError:
            pass

        if use_onnx:
            from .onnx_runtime import OnnxSession

            session = OnnxSession.create(model_name, max_length=512)
            self._classifier = lambda text: session.classify(text, top_k=None)
            return

        # Fallback: transformers pipeline
        try:
            from transformers import pipeline  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "MLToxicityDetector requires onnxruntime+tokenizers (recommended) "
                "or transformers. "
                "Install with: pip install launchpromptly[ml-onnx]"
            )

        kwargs: dict = {
            "task": "text-classification",
            "model": model_name,
            "truncation": True,
            "max_length": 512,
            "top_k": None,
        }
        if device is not None:
            kwargs["device"] = device

        self._classifier = pipeline(**kwargs)

    # -- Provider interface -------------------------------------------------------

    @property
    def name(self) -> str:
        return "ml-toxicity"

    def detect(
        self,
        text: str,
        location: ContentLocation = "input",
    ) -> List[ContentViolation]:
        """Classify *text* for toxicity and return content violations.

        Parameters
        ----------
        text:
            The input text to scan.
        location:
            Whether the text is user ``"input"`` or model ``"output"``.
        """
        if not text:
            return []

        # Run the classifier.  With ``top_k=None`` we get all labels, e.g.:
        # [[{"label": "toxic", "score": 0.91}, {"label": "obscene", "score": 0.42}, ...]]
        raw_results = self._classifier(text)

        # Normalise: the pipeline wraps results in an outer list for a single input.
        if raw_results and isinstance(raw_results[0], list):
            label_scores = raw_results[0]
        else:
            label_scores = raw_results if raw_results else []

        violations: List[ContentViolation] = []
        # Keep track of which ContentCategory values we've already reported so
        # we don't return duplicate categories (e.g. both "toxic" and
        # "severe_toxic" map to "hate_speech").
        seen_categories: set[str] = set()

        for item in label_scores:
            label = item.get("label", "").lower()
            score = float(item.get("score", 0.0))

            if score < self._threshold:
                continue

            category = _TOXICITY_LABEL_MAP.get(label)
            if category is None:
                # Unknown label -- skip.
                continue

            if category in seen_categories:
                continue
            seen_categories.add(category)

            severity: ContentSeverity = "block" if score >= _BLOCK_THRESHOLD else "warn"

            violations.append(
                ContentViolation(
                    category=category,
                    matched=text,
                    severity=severity,
                    location=location,
                )
            )

        return violations
