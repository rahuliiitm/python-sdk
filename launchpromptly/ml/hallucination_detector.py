"""
ML-based hallucination detector using a cross-encoder model.

Compares generated text against source text to score faithfulness.
Requires a cross-encoder model (e.g. vectara/HHEM) with ONNX weights.

Prefers onnxruntime + tokenizers for native inference (8-20ms).
Falls back to transformers + torch if onnxruntime is not installed.

Requires::

    pip install onnxruntime tokenizers   # recommended
    pip install launchpromptly[ml-onnx]  # same as above
"""
from __future__ import annotations

from typing import Any, Optional

from launchpromptly._internal.hallucination import HallucinationResult


class MLHallucinationDetector:
    """ML-based hallucination detector using a cross-encoder model.

    Requires a cross-encoder model with ONNX weights.
    Defaults to ``vectara/hallucination_evaluation_model`` (137M params).

    Tries ONNX Runtime first (8-20ms), falls back to transformers (500ms-2s).

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

            session = OnnxSession.create(
                model_name, max_length=512, quantized=quantized
            )
            self._classifier = lambda inputs: session.classify_pair(
                inputs["text"], inputs["text_pair"]
            )
            return

        # Fallback: transformers pipeline
        try:
            from transformers import pipeline  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "MLHallucinationDetector requires onnxruntime+tokenizers (recommended) "
                "or transformers. "
                "Install with: pip install launchpromptly[ml-onnx]"
            )

        kwargs: dict = {
            "task": "text-classification",
            "model": model_name,
            "truncation": True,
            "max_length": 512,
        }
        if device is not None:
            kwargs["device"] = device

        if quantized:
            try:
                import onnxruntime  # noqa: F401
                kwargs["model_kwargs"] = {"from_tf": False}
            except ImportError:
                pass

        self._classifier = pipeline(**kwargs)

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
