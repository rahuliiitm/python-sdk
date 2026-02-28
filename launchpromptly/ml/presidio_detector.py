"""
ML-based PII detector using Microsoft Presidio + spaCy NER.

Catches PII that regex patterns cannot: person names, organization names,
locations, and free-form addresses.

Requires:
    pip install presidio-analyzer spacy
    python -m spacy download en_core_web_sm

Or simply:
    pip install launchpromptly[ml]
"""
from __future__ import annotations

from typing import List, Optional

from launchpromptly._internal.pii import PIIDetection, PIIDetectOptions


# Mapping from Presidio entity types to our PIIType strings.
# Note: "person_name" and "org_name" are EXTENDED types that don't exist in the
# core regex detector.  The ML plugin adds capabilities beyond what regex can do.
_PRESIDIO_TO_PII_TYPE = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "US_SSN": "ssn",
    "CREDIT_CARD": "credit_card",
    "IP_ADDRESS": "ip_address",
    "LOCATION": "us_address",
    "PERSON": "person_name",
    "ORGANIZATION": "org_name",
    "DATE_TIME": "date_of_birth",
}

# The subset that matches the core PIIType literal exactly.
_CORE_SUPPORTED_TYPES = [
    "email",
    "phone",
    "ssn",
    "credit_card",
    "ip_address",
    "us_address",
]


class PresidioPIIDetector:
    """ML-based PII detector using Microsoft Presidio + spaCy NER.

    Catches PII that regex can't: person names, org names, locations.
    Requires: ``pip install presidio-analyzer spacy``
    Also: ``python -m spacy download en_core_web_sm``

    Example::

        from launchpromptly.ml import PresidioPIIDetector
        detector = PresidioPIIDetector()
        detections = detector.detect("John Smith lives at 123 Main St")
    """

    def __init__(self, model_name: str = "en_core_web_sm") -> None:
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                "PresidioPIIDetector requires presidio-analyzer. "
                "Install with: pip install launchpromptly[ml]"
            )

        self._model_name = model_name
        self._analyzer = AnalyzerEngine()
        # Store the entity types we ask Presidio to look for.
        self._presidio_entities = list(_PRESIDIO_TO_PII_TYPE.keys())

    # -- Provider interface -------------------------------------------------------

    @property
    def name(self) -> str:
        return "presidio"

    @property
    def supported_types(self) -> list:
        """Return the core PIIType values this detector supports.

        Extended types (person_name, org_name) are also detected but are not
        listed here because they fall outside the core PIIType literal.
        """
        return list(_CORE_SUPPORTED_TYPES)

    def detect(
        self,
        text: str,
        options: Optional[PIIDetectOptions] = None,
    ) -> List[PIIDetection]:
        """Run Presidio analysis and return mapped PIIDetection objects.

        Parameters
        ----------
        text:
            The input text to scan.
        options:
            Optional filtering.  When ``options.types`` is set only detections
            whose mapped type is in that list are returned.
        """
        if not text:
            return []

        # Determine which types to include.
        allowed_types = set(options.types) if options and options.types else None

        # Build the list of Presidio entities to request.
        if allowed_types is not None:
            # Reverse-map: find which Presidio entity names map to allowed types.
            entities_to_request = [
                presidio_entity
                for presidio_entity, pii_type in _PRESIDIO_TO_PII_TYPE.items()
                if pii_type in allowed_types
            ]
            if not entities_to_request:
                return []
        else:
            entities_to_request = self._presidio_entities

        # Run the Presidio analyzer.
        results = self._analyzer.analyze(
            text=text,
            entities=entities_to_request,
            language="en",
        )

        detections: List[PIIDetection] = []
        for result in results:
            mapped_type = _PRESIDIO_TO_PII_TYPE.get(result.entity_type)
            if mapped_type is None:
                # Unknown entity type from Presidio -- skip.
                continue

            if allowed_types is not None and mapped_type not in allowed_types:
                continue

            detections.append(
                PIIDetection(
                    type=mapped_type,  # type: ignore[arg-type]
                    value=text[result.start : result.end],
                    start=result.start,
                    end=result.end,
                    confidence=result.score,
                )
            )

        # Sort by start position, then by confidence descending (matches core behaviour).
        detections.sort(key=lambda d: (d.start, -d.confidence))
        return detections
