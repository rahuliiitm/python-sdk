"""
LaunchPromptly ML plugin -- optional ML-based security detectors.

Install via::

    pip install launchpromptly[ml]

This package provides three ML-powered providers that can be used alongside
(or in place of) the built-in regex / rule-based detectors:

* :class:`PresidioPIIDetector` -- NER-based PII detection (Microsoft Presidio)
* :class:`MLInjectionDetector` -- Transformer-based prompt injection detection
* :class:`MLJailbreakDetector` -- Transformer-based jailbreak detection
* :class:`MLToxicityDetector` -- Transformer-based toxicity / content-safety detection

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


__all__ = [
    "PresidioPIIDetector",
    "MLInjectionDetector",
    "MLJailbreakDetector",
    "MLToxicityDetector",
    "MLHallucinationDetector",
]
