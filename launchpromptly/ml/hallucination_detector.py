"""
ML-based hallucination detector using a cross-encoder model.

Compares generated text against source text to score faithfulness.
Uses the vectara/hallucination_evaluation_model (HHEM) by default.

Requires:
    pip install transformers torch  (or transformers onnxruntime)

Or simply:
    pip install launchpromptly[ml]
"""
from __future__ import annotations

from typing import Optional

from launchpromptly._internal.hallucination import HallucinationResult


class MLHallucinationDetector:
    """ML-based hallucination detector using a cross-encoder model.

    Uses the ``vectara/hallucination_evaluation_model`` by default --
    a 137M parameter cross-encoder trained to score text faithfulness.

    Example::

        from launchpromptly.ml import MLHallucinationDetector
        detector = MLHallucinationDetector()
        result = detector.detect("Paris is in Germany", "Paris is the capital of France")
    """

    def __init__(
        self,
        model_name: str = "vectara/hallucination_evaluation_model",
        threshold: float = 0.5,
        device: Optional[int] = None,
        quantized: bool = True,
    ) -> None:
        try:
            from transformers import pipeline  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "MLHallucinationDetector requires transformers. "
                "Install with: pip install launchpromptly[ml]"
            )

        kwargs: dict = {
            "task": "text-classification",
            "model": model_name,
            "truncation": True,
            "max_length": 512,
        }
        if device is not None:
            kwargs["device"] = device

        # Use ONNX runtime if available and quantized mode requested
        if quantized:
            try:
                import onnxruntime  # noqa: F401

                kwargs["model_kwargs"] = {"from_tf": False}
            except ImportError:
                pass  # Fall back to PyTorch

        self._classifier = pipeline(**kwargs)
        self._model_name = model_name
        self._threshold = threshold

    # -- Provider interface -------------------------------------------------------

    @property
    def name(self) -> str:
        return "ml-hallucination"

    def detect(
        self,
        generated: str,
        source: str,
    ) -> HallucinationResult:
        """Score the faithfulness of *generated* text against *source*.

        Parameters
        ----------
        generated:
            The LLM-generated response to check.
        source:
            The reference text (e.g., retrieved documents or system prompt).
        """
        if not generated or not source:
            return HallucinationResult(
                hallucinated=False, faithfulness_score=1.0, severity="low"
            )

        # Cross-encoder takes (premise, hypothesis) pair
        result = self._classifier(
            {"text": source, "text_pair": generated},
        )

        if isinstance(result, list) and len(result) > 0:
            prediction = result[0] if isinstance(result[0], dict) else result[0]
        else:
            return HallucinationResult(
                hallucinated=False, faithfulness_score=1.0, severity="low"
            )

        label = prediction.get("label", "").upper()
        score = float(prediction.get("score", 0.0))

        # HHEM outputs labels like 'CONSISTENT' / 'HALLUCINATED'
        consistent_labels = {"CONSISTENT", "ENTAILMENT", "LABEL_1"}
        hallucinated_labels = {"HALLUCINATED", "CONTRADICTION", "LABEL_0"}

        if label in consistent_labels:
            faithfulness_score = score
        elif label in hallucinated_labels:
            faithfulness_score = 1.0 - score
        else:
            faithfulness_score = score

        faithfulness_score = round(faithfulness_score * 100) / 100

        hallucinated = faithfulness_score < self._threshold

        if faithfulness_score >= 0.7:
            severity = "low"
        elif faithfulness_score >= 0.4:
            severity = "medium"
        else:
            severity = "high"

        return HallucinationResult(
            hallucinated=hallucinated,
            faithfulness_score=faithfulness_score,
            severity=severity,
        )
