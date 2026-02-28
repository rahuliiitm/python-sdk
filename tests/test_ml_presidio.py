"""Tests for the Presidio-based ML PII detector.

All tests mock the ``presidio_analyzer`` dependency so they run without
installing any heavy ML libraries.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from launchpromptly._internal.pii import PIIDetection, PIIDetectOptions


# ---------------------------------------------------------------------------
# Helpers -- lightweight stand-ins for presidio_analyzer objects
# ---------------------------------------------------------------------------

@dataclass
class _FakePresidioResult:
    """Mimics ``presidio_analyzer.RecognizerResult``."""
    entity_type: str
    start: int
    end: int
    score: float


def _make_fake_analyzer(results: List[_FakePresidioResult]) -> MagicMock:
    """Return a mock ``AnalyzerEngine`` whose ``analyze`` returns *results*."""
    engine = MagicMock()
    engine.analyze.return_value = results
    return engine


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestPresidioPIIDetectorInit:
    """Tests around construction and lazy import behaviour."""

    def test_raises_import_error_when_presidio_missing(self):
        """If ``presidio_analyzer`` is not installed the constructor must
        raise ``ImportError`` with a helpful message."""
        with patch.dict("sys.modules", {"presidio_analyzer": None}):
            # Force re-import so the try/except inside __init__ fires.
            import importlib
            import launchpromptly.ml.presidio_detector as mod

            importlib.reload(mod)
            with pytest.raises(ImportError, match="presidio-analyzer"):
                mod.PresidioPIIDetector()

    def test_successful_init_with_mocked_presidio(self):
        """When presidio_analyzer is importable, construction succeeds."""
        fake_module = MagicMock()
        fake_module.AnalyzerEngine.return_value = MagicMock()

        with patch.dict("sys.modules", {"presidio_analyzer": fake_module}):
            import importlib
            import launchpromptly.ml.presidio_detector as mod

            importlib.reload(mod)
            detector = mod.PresidioPIIDetector()
            assert detector is not None

    def test_custom_model_name_stored(self):
        fake_module = MagicMock()
        fake_module.AnalyzerEngine.return_value = MagicMock()

        with patch.dict("sys.modules", {"presidio_analyzer": fake_module}):
            import importlib
            import launchpromptly.ml.presidio_detector as mod

            importlib.reload(mod)
            detector = mod.PresidioPIIDetector(model_name="en_core_web_lg")
            assert detector._model_name == "en_core_web_lg"


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------

class TestPresidioPIIDetectorProperties:

    @pytest.fixture()
    def detector(self):
        fake_module = MagicMock()
        fake_module.AnalyzerEngine.return_value = MagicMock()

        with patch.dict("sys.modules", {"presidio_analyzer": fake_module}):
            import importlib
            import launchpromptly.ml.presidio_detector as mod

            importlib.reload(mod)
            return mod.PresidioPIIDetector()

    def test_name_is_presidio(self, detector):
        assert detector.name == "presidio"

    def test_supported_types_contains_core_types(self, detector):
        types = detector.supported_types
        for t in ["email", "phone", "ssn", "credit_card", "ip_address", "us_address"]:
            assert t in types

    def test_supported_types_does_not_contain_extended(self, detector):
        """Extended types (person_name, org_name) should NOT be in
        ``supported_types`` -- they are detected but aren't part of the
        core PIIType literal."""
        types = detector.supported_types
        assert "person_name" not in types
        assert "org_name" not in types


# ---------------------------------------------------------------------------
# Detection -- entity mapping
# ---------------------------------------------------------------------------

class TestPresidioPIIDetectorDetect:

    def _build_detector(self, fake_results):
        """Build a detector with a mocked analyzer that returns *fake_results*."""
        fake_module = MagicMock()
        engine = _make_fake_analyzer(fake_results)
        fake_module.AnalyzerEngine.return_value = engine

        with patch.dict("sys.modules", {"presidio_analyzer": fake_module}):
            import importlib
            import launchpromptly.ml.presidio_detector as mod

            importlib.reload(mod)
            detector = mod.PresidioPIIDetector()
            # Overwrite the engine that __init__ created with our mock.
            detector._analyzer = engine
            return detector

    # -- basic mapping --

    def test_maps_email_address(self):
        text = "Contact john@acme.com"
        detector = self._build_detector([
            _FakePresidioResult("EMAIL_ADDRESS", 8, 21, 0.95),
        ])
        dets = detector.detect(text)
        assert len(dets) == 1
        assert dets[0].type == "email"
        assert dets[0].value == "john@acme.com"
        assert dets[0].confidence == 0.95

    def test_maps_phone_number(self):
        text = "Call 555-123-4567"
        detector = self._build_detector([
            _FakePresidioResult("PHONE_NUMBER", 5, 17, 0.88),
        ])
        dets = detector.detect(text)
        assert len(dets) == 1
        assert dets[0].type == "phone"

    def test_maps_us_ssn(self):
        text = "SSN: 123-45-6789"
        detector = self._build_detector([
            _FakePresidioResult("US_SSN", 5, 16, 0.99),
        ])
        dets = detector.detect(text)
        assert dets[0].type == "ssn"

    def test_maps_credit_card(self):
        text = "Card: 4111111111111111"
        detector = self._build_detector([
            _FakePresidioResult("CREDIT_CARD", 6, 22, 0.92),
        ])
        dets = detector.detect(text)
        assert dets[0].type == "credit_card"

    def test_maps_ip_address(self):
        text = "IP: 10.20.30.40"
        detector = self._build_detector([
            _FakePresidioResult("IP_ADDRESS", 4, 15, 0.80),
        ])
        dets = detector.detect(text)
        assert dets[0].type == "ip_address"

    def test_maps_location_to_us_address(self):
        text = "Lives in New York"
        detector = self._build_detector([
            _FakePresidioResult("LOCATION", 9, 17, 0.85),
        ])
        dets = detector.detect(text)
        assert dets[0].type == "us_address"

    def test_maps_person_to_person_name(self):
        text = "Name: John Smith"
        detector = self._build_detector([
            _FakePresidioResult("PERSON", 6, 16, 0.90),
        ])
        dets = detector.detect(text)
        assert dets[0].type == "person_name"
        assert dets[0].value == "John Smith"

    def test_maps_organization_to_org_name(self):
        text = "Works at Acme Corp"
        detector = self._build_detector([
            _FakePresidioResult("ORGANIZATION", 9, 18, 0.87),
        ])
        dets = detector.detect(text)
        assert dets[0].type == "org_name"

    def test_maps_date_time_to_date_of_birth(self):
        text = "Born on January 1, 1990"
        detector = self._build_detector([
            _FakePresidioResult("DATE_TIME", 8, 23, 0.75),
        ])
        dets = detector.detect(text)
        assert dets[0].type == "date_of_birth"

    # -- filtering --

    def test_respects_type_filtering(self):
        text = "john@acme.com and 123-45-6789"
        detector = self._build_detector([
            _FakePresidioResult("EMAIL_ADDRESS", 0, 13, 0.95),
            _FakePresidioResult("US_SSN", 18, 29, 0.99),
        ])
        dets = detector.detect(text, PIIDetectOptions(types=["email"]))
        assert len(dets) == 1
        assert dets[0].type == "email"

    def test_type_filter_returns_empty_when_no_match(self):
        text = "john@acme.com"
        detector = self._build_detector([
            _FakePresidioResult("EMAIL_ADDRESS", 0, 13, 0.95),
        ])
        dets = detector.detect(text, PIIDetectOptions(types=["ssn"]))
        assert len(dets) == 0

    # -- edge cases --

    def test_empty_text_returns_empty(self):
        detector = self._build_detector([])
        dets = detector.detect("")
        assert dets == []

    def test_skips_unknown_presidio_entities(self):
        text = "Some text here"
        detector = self._build_detector([
            _FakePresidioResult("UNKNOWN_ENTITY", 0, 14, 0.50),
        ])
        dets = detector.detect(text)
        assert len(dets) == 0

    def test_results_sorted_by_start_position(self):
        text = "SSN 123-45-6789 email john@acme.com"
        detector = self._build_detector([
            _FakePresidioResult("EMAIL_ADDRESS", 22, 35, 0.95),
            _FakePresidioResult("US_SSN", 4, 15, 0.99),
        ])
        dets = detector.detect(text)
        assert len(dets) == 2
        assert dets[0].start < dets[1].start

    def test_confidence_scores_passed_through(self):
        text = "john@acme.com"
        detector = self._build_detector([
            _FakePresidioResult("EMAIL_ADDRESS", 0, 13, 0.42),
        ])
        dets = detector.detect(text)
        assert dets[0].confidence == 0.42
