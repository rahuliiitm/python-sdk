"""
Shared sentence embedding provider using all-MiniLM-L6-v2.

Loads the model once (singleton via OnnxSession cache) and provides
embeddings to all L3-L4 guards:
- Response Judge: semantic topic matching
- Conversation Guard: embedding-based drift detection
- CoT Guard: goal drift via embeddings

Requires::

    pip install onnxruntime tokenizers

Or simply::

    pip install launchpromptly[ml-onnx]
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .onnx_runtime import OnnxSession

_DEFAULT_MODEL = "Xenova/all-MiniLM-L6-v2"


class MLEmbeddingProvider:
    """Shared embedding provider -- load once, use across all guards.

    Example::

        from launchpromptly.ml import MLEmbeddingProvider
        emb = MLEmbeddingProvider.create()
        vec = emb.embed("Hello world")
    """

    name = "ml-embedding"

    def __init__(self, session: OnnxSession, model_name: str) -> None:
        self._session = session
        self._model_name = model_name

    @classmethod
    def create(
        cls,
        *,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ) -> "MLEmbeddingProvider":
        """Create an MLEmbeddingProvider by loading the sentence-transformer model."""
        name = model_name or _DEFAULT_MODEL
        session = OnnxSession.create(
            name,
            max_length=256,  # Sentence embeddings rarely need > 256 tokens
            cache_dir=cache_dir,
        )
        return cls(session, name)

    @classmethod
    def _create_for_test(cls, session: OnnxSession) -> "MLEmbeddingProvider":
        """For testing: create with a pre-built OnnxSession."""
        return cls(session, "test-model")

    def embed(self, text: str) -> np.ndarray:
        """Embed a single text. Returns a numpy array of dimension 384."""
        return self._session.embed(text)

    def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Embed multiple texts."""
        return [self._session.embed(t) for t in texts]

    def cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return OnnxSession.cosine(a, b)

    @property
    def model_name(self) -> str:
        """The underlying model name."""
        return self._model_name
