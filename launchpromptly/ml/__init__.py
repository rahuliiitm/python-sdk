"""
LaunchPromptly ML plugin -- optional ML-based security detectors.

Install via::

    pip install launchpromptly[ml-onnx]   # recommended: ONNX Runtime (fast, ~20MB)
    pip install launchpromptly[ml]         # heavy: transformers + torch (~2.5GB)

Uses ONNX Runtime for native inference (8-20ms) when onnxruntime + tokenizers
are installed. Falls back to transformers (500ms-2s) otherwise.

Detectors:

* :class:`MLInjectionDetector` -- Prompt injection detection
* :class:`MLJailbreakDetector` -- Jailbreak detection
* :class:`MLToxicityDetector` -- Toxicity / content-safety detection
* :class:`MLHallucinationDetector` -- Hallucination detection (cross-encoder)
* :class:`PresidioPIIDetector` -- NER-based PII detection (Microsoft Presidio)

Each provider satisfies the corresponding Protocol interface defined in the
core SDK so it can be registered as a drop-in replacement.
"""
from __future__ import annotations

_import_errors: dict[str, str] = {}

try:
    from .presidio_detector import PresidioPIIDetector
except ImportError as exc:
    _import_errors["PresidioPIIDetector"] = str(exc)

    class PresidioPIIDetector:  # type: ignore[no-redef]
        """Placeholder that raises when Presidio deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "PresidioPIIDetector requires presidio-analyzer and spacy. "
                f"Install with: pip install launchpromptly[ml]\n"
                f"Original error: {_import_errors['PresidioPIIDetector']}"
            )


try:
    from .injection_detector import MLInjectionDetector
except ImportError as exc:
    _import_errors["MLInjectionDetector"] = str(exc)

    class MLInjectionDetector:  # type: ignore[no-redef]
        """Placeholder that raises when transformers deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLInjectionDetector requires transformers. "
                f"Install with: pip install launchpromptly[ml]\n"
                f"Original error: {_import_errors['MLInjectionDetector']}"
            )


try:
    from .jailbreak_detector import MLJailbreakDetector
except ImportError as exc:
    _import_errors["MLJailbreakDetector"] = str(exc)

    class MLJailbreakDetector:  # type: ignore[no-redef]
        """Placeholder that raises when transformers deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLJailbreakDetector requires transformers. "
                f"Install with: pip install launchpromptly[ml]\n"
                f"Original error: {_import_errors['MLJailbreakDetector']}"
            )


try:
    from .toxicity_detector import MLToxicityDetector
except ImportError as exc:
    _import_errors["MLToxicityDetector"] = str(exc)

    class MLToxicityDetector:  # type: ignore[no-redef]
        """Placeholder that raises when transformers deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLToxicityDetector requires transformers. "
                f"Install with: pip install launchpromptly[ml]\n"
                f"Original error: {_import_errors['MLToxicityDetector']}"
            )


try:
    from .hallucination_detector import MLHallucinationDetector
except ImportError as exc:
    _import_errors["MLHallucinationDetector"] = str(exc)

    class MLHallucinationDetector:  # type: ignore[no-redef]
        """Placeholder that raises when transformers deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLHallucinationDetector requires transformers. "
                f"Install with: pip install launchpromptly[ml]\n"
                f"Original error: {_import_errors['MLHallucinationDetector']}"
            )


try:
    from .embedding_provider import MLEmbeddingProvider
except ImportError as exc:
    _import_errors["MLEmbeddingProvider"] = str(exc)

    class MLEmbeddingProvider:  # type: ignore[no-redef]
        """Placeholder that raises when ONNX deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLEmbeddingProvider requires onnxruntime and tokenizers. "
                f"Install with: pip install launchpromptly[ml-onnx]\n"
                f"Original error: {_import_errors['MLEmbeddingProvider']}"
            )


try:
    from .response_judge import MLResponseJudge
except ImportError as exc:
    _import_errors["MLResponseJudge"] = str(exc)

    class MLResponseJudge:  # type: ignore[no-redef]
        """Placeholder that raises when ONNX deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLResponseJudge requires onnxruntime and tokenizers. "
                f"Install with: pip install launchpromptly[ml-onnx]\n"
                f"Original error: {_import_errors['MLResponseJudge']}"
            )


try:
    from .context_extractor import MLContextExtractor
except ImportError as exc:
    _import_errors["MLContextExtractor"] = str(exc)

    class MLContextExtractor:  # type: ignore[no-redef]
        """Placeholder that raises when ONNX deps are missing."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "MLContextExtractor requires onnxruntime and tokenizers. "
                f"Install with: pip install launchpromptly[ml-onnx]\n"
                f"Original error: {_import_errors['MLContextExtractor']}"
            )


try:
    from .attack_embeddings import load_attack_index, match_against_index, has_attack_match
except ImportError:
    pass

try:
    from .onnx_runtime import OnnxSession
except ImportError:
    pass

try:
    from .model_cache import ensure_model, get_cache_dir, remove_model, list_cached_models, get_registered_models, MODEL_NAME_MAP
except Exception:
    pass

__all__ = [
    "PresidioPIIDetector",
    "MLInjectionDetector",
    "MLJailbreakDetector",
    "MLToxicityDetector",
    "MLHallucinationDetector",
    "MLEmbeddingProvider",
    "MLResponseJudge",
    "MLContextExtractor",
    "load_attack_index",
    "match_against_index",
    "has_attack_match",
    "OnnxSession",
    "ensure_model",
    "get_cache_dir",
    "remove_model",
    "list_cached_models",
    "get_registered_models",
    "MODEL_NAME_MAP",
]
