"""
Multi-class attack classifier using fine-tuned MiniLM.

7-class output: safe, injection, jailbreak, data_extraction,
manipulation, role_escape, social_engineering.

Complements existing binary detectors with multi-class categorization
and faster inference (<5ms, 22M params, INT8 quantized).

Requires::

    pip install onnxruntime tokenizers   # recommended
    pip install launchpromptly[ml-onnx]  # same as above
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, List, Literal, Optional

from launchpromptly._internal.injection import (
    InjectionAction,
    InjectionAnalysis,
    InjectionOptions,
)

AttackLabel = Literal[
    "safe",
    "injection",
    "jailbreak",
    "data_extraction",
    "manipulation",
    "role_escape",
    "social_engineering",
]

ATTACK_LABELS = frozenset({
    "injection",
    "jailbreak",
    "data_extraction",
    "manipulation",
    "role_escape",
    "social_engineering",
})

_DEFAULT_WARN_THRESHOLD = 0.3
_DEFAULT_BLOCK_THRESHOLD = 0.7
_CATEGORY_TRIGGER_THRESHOLD = 0.15


@dataclass
class AttackClassification:
    """Full multi-class classification result."""

    label: str
    score: float
    all_scores: List[dict]
    is_attack: bool


class MLAttackClassifier:
    """Multi-class attack classifier using a fine-tuned all-MiniLM-L6-v2.

    Implements the injection detector provider interface for drop-in use
    in the security pipeline. Also exposes ``classify()`` for full 7-class
    output with per-category scores.

    Example::

        from launchpromptly.ml import MLAttackClassifier
        classifier = MLAttackClassifier()
        result = classifier.classify("Ignore all previous instructions")
        # result.label == "injection", result.is_attack == True

        # Or use as injection provider in the pipeline:
        lp = LaunchPromptly(api_key="...")
        wrapped = lp.wrap(openai, security={
            "injection": {"providers": [classifier]},
        })
    """

    def __init__(
        self,
        model_name: str = "launchpromptly/attack-classifier-v1",
        quantized: bool = True,
        cache_dir: Optional[str] = None,
    ) -> None:
        self._model_name = model_name

        use_onnx = False
        try:
            import onnxruntime  # noqa: F401
            from tokenizers import Tokenizer  # noqa: F401

            use_onnx = True
        except ImportError:
            pass

        _cache_path = Path(cache_dir) if cache_dir else None

        if use_onnx:
            from .onnx_runtime import OnnxSession

            session = OnnxSession.create(
                model_name, max_length=256, quantized=quantized, cache_dir=_cache_path
            )
            self._classifier = lambda text, **kw: session.classify(text, **kw)
            return

        # Fallback: transformers pipeline
        try:
            from transformers import pipeline  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "MLAttackClassifier requires onnxruntime+tokenizers (recommended) "
                "or transformers. "
                "Install with: pip install launchpromptly[ml-onnx]"
            )

        self._classifier = pipeline(
            task="text-classification",
            model=model_name,
            truncation=True,
            max_length=256,
            top_k=None,
        )

    @property
    def name(self) -> str:
        return "ml-attack-classifier"

    def classify(self, text: str) -> AttackClassification:
        """Full multi-class classification.

        Returns the top label, all scores, and whether it's an attack.
        """
        if not text:
            return AttackClassification(
                label="safe",
                score=1.0,
                all_scores=[{"label": "safe", "score": 1.0}],
                is_attack=False,
            )

        results = self._classifier(text, top_k=None)

        # Normalize output format
        if isinstance(results, list) and results and isinstance(results[0], list):
            results = results[0]

        all_scores = [
            {"label": r.get("label", ""), "score": float(r.get("score", 0))}
            for r in results
        ]
        all_scores.sort(key=lambda s: s["score"], reverse=True)

        top = all_scores[0] if all_scores else {"label": "safe", "score": 1.0}
        is_attack = top["label"] in ATTACK_LABELS

        return AttackClassification(
            label=top["label"],
            score=top["score"],
            all_scores=all_scores,
            is_attack=is_attack,
        )

    def detect(
        self,
        text: str,
        options: Optional[InjectionOptions] = None,
    ) -> InjectionAnalysis:
        """InjectionDetectorProvider interface.

        Maps multi-class output to binary InjectionAnalysis.
        Risk score = 1 - P(safe).
        """
        if not text:
            return InjectionAnalysis(risk_score=0.0, triggered=[], action="allow")

        warn_threshold = (
            options.warn_threshold
            if options and options.warn_threshold is not None
            else _DEFAULT_WARN_THRESHOLD
        )
        block_threshold = (
            options.block_threshold
            if options and options.block_threshold is not None
            else _DEFAULT_BLOCK_THRESHOLD
        )

        classification = self.classify(text)

        # Risk score = 1 - P(safe)
        safe_score = 0.0
        for s in classification.all_scores:
            if s["label"] == "safe":
                safe_score = s["score"]
                break
        risk_score = round((1 - safe_score) * 100) / 100

        # Triggered categories
        triggered: list[str] = []
        for s in classification.all_scores:
            if s["label"] in ATTACK_LABELS and s["score"] >= _CATEGORY_TRIGGER_THRESHOLD:
                triggered.append(s["label"])

        action: InjectionAction
        if risk_score >= block_threshold:
            action = "block"
        elif risk_score >= warn_threshold:
            action = "warn"
        else:
            action = "allow"

        return InjectionAnalysis(
            risk_score=risk_score,
            triggered=triggered,
            action=action,
        )
