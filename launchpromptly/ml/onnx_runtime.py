"""
ONNX Runtime integration for ML detectors.

Replaces the ``transformers`` pipeline inference (500ms-2s) with native
``onnxruntime`` inference (8-20ms) -- a 25-100x speedup.

Tokenization uses the lightweight ``tokenizers`` library (HuggingFace Rust
bindings, ~5MB) instead of the heavy ``transformers`` + ``torch`` combo (~2.5GB).

Requires::

    pip install onnxruntime tokenizers

Or simply::

    pip install launchpromptly[ml-onnx]
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax: converts logits to probabilities."""
    shifted = logits - logits.max()
    exps = np.exp(shifted)
    return exps / exps.sum()


# Session cache -- avoids reloading the same model.
_session_cache: dict[str, "OnnxSession"] = {}


class OnnxSession:
    """Wraps an ONNX Runtime inference session with tokenization.

    Provides three inference modes:

    * :meth:`classify` -- text classification (injection, toxicity)
    * :meth:`classify_pair` -- cross-encoder classification (hallucination)
    * :meth:`token_classify` -- token classification / NER (PII)
    """

    def __init__(
        self,
        session: Any,
        tokenizer: Any,
        config: dict,
        max_length: int,
        model_id: str,
    ) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self._config = config
        self._max_length = max_length
        self._model_id = model_id

    @classmethod
    def create(
        cls,
        model_id: str,
        *,
        max_length: int = 512,
        quantized: bool = True,
    ) -> "OnnxSession":
        """Create (or retrieve cached) ONNX session for a model.

        Downloads model files on first use.
        """
        cache_key = f"{model_id}:{quantized}"
        cached = _session_cache.get(cache_key)
        if cached is not None:
            return cached

        # Download model files
        from .model_cache import ensure_model

        model_dir = ensure_model(model_id, quantized=quantized)

        # Load ONNX Runtime
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "OnnxSession requires onnxruntime. "
                "Install with: pip install onnxruntime"
            )

        session = ort.InferenceSession(
            str(model_dir / "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

        # Load tokenizer via the lightweight `tokenizers` library
        try:
            from tokenizers import Tokenizer
        except ImportError:
            raise ImportError(
                "OnnxSession requires tokenizers. "
                "Install with: pip install tokenizers"
            )

        tokenizer_path = model_dir / "tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = Tokenizer.from_file(str(tokenizer_path))
        else:
            # Fallback: download from HuggingFace Hub
            tokenizer = Tokenizer.from_pretrained(model_id)

        tokenizer.enable_truncation(max_length=max_length)

        # Load config for label mapping
        config_path = model_dir / "config.json"
        config: dict = {}
        if config_path.exists():
            config = json.loads(config_path.read_text())

        instance = cls(session, tokenizer, config, max_length, model_id)
        _session_cache[cache_key] = instance
        return instance

    @property
    def id2label(self) -> dict[int, str]:
        """id -> label mapping from config.json."""
        raw = self._config.get("id2label", {})
        return {int(k): v for k, v in raw.items()}

    def classify(
        self,
        text: str,
        *,
        top_k: Optional[int] = 1,
    ) -> list[dict[str, Any]]:
        """Text classification -- returns labels sorted by score (descending).

        For injection detection, toxicity, etc.
        """
        feeds = self._tokenize(text)
        logits = self._run_inference(feeds)
        return self._logits_to_labels(logits, top_k=top_k)

    def classify_pair(
        self,
        text: str,
        text_pair: str,
        *,
        top_k: Optional[int] = 1,
    ) -> list[dict[str, Any]]:
        """Cross-encoder classification -- takes a text pair.

        For hallucination detection (source vs generated text).
        """
        feeds = self._tokenize(text, text_pair)
        logits = self._run_inference(feeds)
        return self._logits_to_labels(logits, top_k=top_k)

    def token_classify(
        self,
        text: str,
    ) -> list[dict[str, Any]]:
        """Token classification (NER) -- returns entity spans with scores.

        Implements simple BIO tag aggregation.
        """
        encoded = self._tokenizer.encode(text)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Add token_type_ids if model expects it
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            token_type_ids = np.array(
                [encoded.type_ids] if encoded.type_ids else [
                    [0] * len(encoded.ids)
                ],
                dtype=np.int64,
            )
            feeds["token_type_ids"] = token_type_ids

        outputs = self._session.run(None, feeds)
        logits = outputs[0][0]  # shape: [seq_len, num_labels]

        id2label = self.id2label
        offsets = encoded.offsets  # list of (start, end) tuples
        seq_len = len(encoded.ids)

        entities: list[dict[str, Any]] = []
        current: Optional[dict] = None

        def flush_current():
            nonlocal current
            if current is None:
                return
            entities.append(
                {
                    "entity_group": current["label"],
                    "score": round(current["score_sum"] / current["count"], 2),
                    "word": text[current["start"] : current["end"]],
                    "start": current["start"],
                    "end": current["end"],
                }
            )
            current = None

        for i in range(seq_len):
            offset = offsets[i] if i < len(offsets) else (0, 0)

            # Skip special tokens
            if offset[0] == 0 and offset[1] == 0 and i > 0:
                flush_current()
                continue

            token_logits = logits[i]
            probs = _softmax(token_logits)
            best_idx = int(np.argmax(probs))
            label = id2label.get(best_idx, f"LABEL_{best_idx}")
            score = float(probs[best_idx])

            # Outside entity
            if label == "O" or label == "LABEL_0":
                flush_current()
                continue

            # Strip B-/I- prefix
            base_label = label
            is_beginning = label.startswith("B-")
            if label.startswith("B-") or label.startswith("I-"):
                base_label = label[2:]

            if (
                current is not None
                and current["label"] == base_label
                and not is_beginning
            ):
                current["score_sum"] += score
                current["count"] += 1
                current["end"] = offset[1]
            else:
                flush_current()
                current = {
                    "label": base_label,
                    "score_sum": score,
                    "count": 1,
                    "start": offset[0],
                    "end": offset[1],
                }

        flush_current()
        return entities

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all cached sessions."""
        _session_cache.clear()

    # -- Private helpers -------------------------------------------------------

    def _tokenize(
        self, text: str, text_pair: Optional[str] = None
    ) -> dict[str, np.ndarray]:
        """Tokenize text into numpy arrays for ONNX inference."""
        if text_pair:
            encoded = self._tokenizer.encode(text, text_pair)
        else:
            encoded = self._tokenizer.encode(text)

        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        feeds: dict[str, np.ndarray] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            token_type_ids = np.array(
                [encoded.type_ids] if encoded.type_ids else [
                    [0] * len(encoded.ids)
                ],
                dtype=np.int64,
            )
            feeds["token_type_ids"] = token_type_ids

        return feeds

    def _run_inference(self, feeds: dict[str, np.ndarray]) -> np.ndarray:
        """Run ONNX inference and return raw logits."""
        outputs = self._session.run(None, feeds)
        return outputs[0][0]  # shape: [num_classes]

    def _logits_to_labels(
        self,
        logits: np.ndarray,
        *,
        top_k: Optional[int] = 1,
    ) -> list[dict[str, Any]]:
        """Convert logits to sorted label-score pairs."""
        probs = _softmax(logits)
        id2label = self.id2label

        results = [
            {"label": id2label.get(i, f"LABEL_{i}"), "score": float(p)}
            for i, p in enumerate(probs)
        ]
        results.sort(key=lambda x: x["score"], reverse=True)

        if top_k is None:
            return results
        return results[:top_k]
