"""Tests for the ML-based toxicity / content-safety detector.

All tests mock the ``transformers`` dependency so they run without
installing any heavy ML libraries.
"""
from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import pytest

from launchpromptly._internal.content_filter import ContentViolation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_pipeline(label_scores: List[dict]) -> MagicMock:
    """Return a callable mock that simulates ``transformers.pipeline(...)``
    configured with ``top_k=None``.

    *label_scores* is a list of ``{"label": ..., "score": ...}`` dicts.
    The pipeline wraps these in an outer list for a single text input.
    """
    classifier = MagicMock()
    classifier.return_value = [label_scores]  # outer list for batch dim
    return classifier


def _build_detector(
    label_scores: List[dict] | None = None,
    threshold: float = 0.5,
):
    """Construct an ``MLToxicityDetector`` with a mocked transformers pipeline."""
    if label_scores is None:
        label_scores = [{"label": "toxic", "score": 0.01}]

    classifier = _make_mock_pipeline(label_scores)

    fake_transformers = MagicMock()
    fake_transformers.pipeline.return_value = classifier

    with patch.dict("sys.modules", {"transformers": fake_transformers}):
        import importlib
        import launchpromptly.ml.toxicity_detector as mod

        importlib.reload(mod)
        detector = mod.MLToxicityDetector(threshold=threshold)
        detector._classifier = classifier
        return detector


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestMLToxicityDetectorInit:

    def test_raises_import_error_when_transformers_missing(self):
        with patch.dict("sys.modules", {"transformers": None}):
            import importlib
            import launchpromptly.ml.toxicity_detector as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="transformers"):
                mod.MLToxicityDetector()

    def test_successful_init_with_mocked_transformers(self):
        detector = _build_detector()
        assert detector is not None

    def test_stores_model_name(self):
        fake_transformers = MagicMock()
        fake_transformers.pipeline.return_value = MagicMock(
            return_value=[[{"label": "toxic", "score": 0.01}]]
        )

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            import importlib
            import launchpromptly.ml.toxicity_detector as mod

            importlib.reload(mod)
            detector = mod.MLToxicityDetector(model_name="custom/toxic-model")
            assert detector._model_name == "custom/toxic-model"

    def test_custom_threshold(self):
        detector = _build_detector(threshold=0.7)
        assert detector._threshold == 0.7


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestMLToxicityDetectorProperties:

    def test_name_is_ml_toxicity(self):
        detector = _build_detector()
        assert detector.name == "ml-toxicity"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class TestMLToxicityDetectorDetect:

    # -- toxic content --

    def test_detects_toxic_content_above_threshold(self):
        detector = _build_detector([
            {"label": "toxic", "score": 0.91},
            {"label": "obscene", "score": 0.12},
            {"label": "threat", "score": 0.05},
            {"label": "insult", "score": 0.08},
            {"label": "identity_hate", "score": 0.03},
            {"label": "severe_toxic", "score": 0.02},
        ])
        violations = detector.detect("You are a terrible person", "input")
        assert len(violations) == 1
        assert violations[0].category == "hate_speech"

    def test_detects_multiple_toxicity_categories(self):
        detector = _build_detector([
            {"label": "toxic", "score": 0.85},
            {"label": "threat", "score": 0.72},
            {"label": "obscene", "score": 0.68},
            {"label": "insult", "score": 0.20},
            {"label": "identity_hate", "score": 0.10},
            {"label": "severe_toxic", "score": 0.05},
        ])
        violations = detector.detect("Very bad text", "input")
        categories = {v.category for v in violations}
        assert "hate_speech" in categories  # from toxic
        assert "violence" in categories  # from threat
        assert "sexual" in categories  # from obscene

    # -- safe content --

    def test_safe_text_returns_no_violations(self):
        detector = _build_detector([
            {"label": "toxic", "score": 0.02},
            {"label": "obscene", "score": 0.01},
            {"label": "threat", "score": 0.005},
            {"label": "insult", "score": 0.01},
            {"label": "identity_hate", "score": 0.003},
            {"label": "severe_toxic", "score": 0.001},
        ])
        violations = detector.detect("The weather is nice today", "input")
        assert violations == []

    # -- empty text --

    def test_empty_text_returns_empty(self):
        detector = _build_detector()
        violations = detector.detect("", "input")
        assert violations == []

    # -- category mapping --

    def test_toxic_maps_to_hate_speech(self):
        detector = _build_detector([{"label": "toxic", "score": 0.9}])
        violations = detector.detect("text", "input")
        assert violations[0].category == "hate_speech"

    def test_severe_toxic_maps_to_hate_speech(self):
        detector = _build_detector([{"label": "severe_toxic", "score": 0.9}])
        violations = detector.detect("text", "input")
        assert violations[0].category == "hate_speech"

    def test_obscene_maps_to_sexual(self):
        detector = _build_detector([{"label": "obscene", "score": 0.9}])
        violations = detector.detect("text", "input")
        assert violations[0].category == "sexual"

    def test_threat_maps_to_violence(self):
        detector = _build_detector([{"label": "threat", "score": 0.9}])
        violations = detector.detect("text", "input")
        assert violations[0].category == "violence"

    def test_insult_maps_to_hate_speech(self):
        detector = _build_detector([{"label": "insult", "score": 0.9}])
        violations = detector.detect("text", "input")
        assert violations[0].category == "hate_speech"

    def test_identity_hate_maps_to_hate_speech(self):
        detector = _build_detector([{"label": "identity_hate", "score": 0.9}])
        violations = detector.detect("text", "input")
        assert violations[0].category == "hate_speech"

    # -- severity thresholds --

    def test_severity_warn_below_block_threshold(self):
        """Score >= 0.5 but < 0.8 should produce 'warn' severity."""
        detector = _build_detector([{"label": "toxic", "score": 0.65}])
        violations = detector.detect("text", "input")
        assert violations[0].severity == "warn"

    def test_severity_block_above_block_threshold(self):
        """Score >= 0.8 should produce 'block' severity."""
        detector = _build_detector([{"label": "toxic", "score": 0.90}])
        violations = detector.detect("text", "input")
        assert violations[0].severity == "block"

    # -- location passthrough --

    def test_location_input_passthrough(self):
        detector = _build_detector([{"label": "toxic", "score": 0.9}])
        violations = detector.detect("text", "input")
        assert violations[0].location == "input"

    def test_location_output_passthrough(self):
        detector = _build_detector([{"label": "toxic", "score": 0.9}])
        violations = detector.detect("text", "output")
        assert violations[0].location == "output"

    # -- deduplication of same category --

    def test_deduplicates_same_category(self):
        """If both 'toxic' and 'insult' fire, they both map to
        'hate_speech' but only one violation should be returned."""
        detector = _build_detector([
            {"label": "toxic", "score": 0.85},
            {"label": "insult", "score": 0.75},
        ])
        violations = detector.detect("text", "input")
        hate_violations = [v for v in violations if v.category == "hate_speech"]
        assert len(hate_violations) == 1

    # -- custom threshold --

    def test_custom_threshold_filters_correctly(self):
        """With threshold=0.8, a score of 0.65 should not be reported."""
        detector = _build_detector(
            [{"label": "toxic", "score": 0.65}],
            threshold=0.8,
        )
        violations = detector.detect("text", "input")
        assert violations == []

    # -- unknown labels --

    def test_unknown_labels_are_skipped(self):
        detector = _build_detector([
            {"label": "unknown_category", "score": 0.99},
        ])
        violations = detector.detect("text", "input")
        assert violations == []

    # -- matched field --

    def test_matched_field_contains_input_text(self):
        detector = _build_detector([{"label": "toxic", "score": 0.9}])
        violations = detector.detect("offensive text here", "input")
        assert violations[0].matched == "offensive text here"
