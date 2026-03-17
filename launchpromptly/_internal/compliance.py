"""
Pre-built compliance configurations for regulated industries.
Each template provides a structured config that can be used with LaunchPromptly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .topic_guard import TopicDefinition
from .topic_templates import MEDICAL_ADVICE, LEGAL_ADVICE, FINANCIAL_ADVICE


@dataclass
class _PIIConfig:
    enabled: bool = True
    block_on_detection: bool = True


@dataclass
class _ContentFilterConfig:
    enabled: bool = True
    block_on_violation: bool = True
    categories: List[str] = field(default_factory=list)


@dataclass
class _TopicGuardConfig:
    blocked_topics: List[TopicDefinition] = field(default_factory=list)


@dataclass
class _SecretConfig:
    enabled: bool = True


@dataclass
class _OutputSafetyConfig:
    enabled: bool = True


@dataclass
class ComplianceTemplate:
    name: str
    description: str
    pii: _PIIConfig = field(default_factory=_PIIConfig)
    content_filter: _ContentFilterConfig = field(default_factory=_ContentFilterConfig)
    topic_guard: _TopicGuardConfig = field(default_factory=_TopicGuardConfig)
    secret_detection: _SecretConfig = field(default_factory=_SecretConfig)
    output_safety: _OutputSafetyConfig = field(default_factory=_OutputSafetyConfig)


# -- Healthcare (HIPAA-aligned) ------------------------------------------------

HEALTHCARE_COMPLIANCE = ComplianceTemplate(
    name="healthcare",
    description="HIPAA-aligned guardrails: blocks PII, medical advice, and sensitive content in LLM interactions.",
    pii=_PIIConfig(enabled=True, block_on_detection=True),
    content_filter=_ContentFilterConfig(
        enabled=True,
        block_on_violation=True,
        categories=["hate_speech", "sexual", "violence", "self_harm", "bias"],
    ),
    topic_guard=_TopicGuardConfig(blocked_topics=[
        MEDICAL_ADVICE,
        TopicDefinition(
            name="phi_disclosure",
            keywords=[
                "patient name", "medical record", "health record",
                "diagnosis of", "treatment plan for", "prescription for",
                "insurance number", "social security", "date of birth",
                "medical history", "lab results for",
            ],
            threshold=0.03,
        ),
    ]),
    secret_detection=_SecretConfig(enabled=True),
    output_safety=_OutputSafetyConfig(enabled=True),
)

# -- Finance (SOX / financial regulation aligned) ------------------------------

FINANCE_COMPLIANCE = ComplianceTemplate(
    name="finance",
    description="Financial regulation guardrails: blocks PII, investment advice, and unauthorized financial guidance.",
    pii=_PIIConfig(enabled=True, block_on_detection=True),
    content_filter=_ContentFilterConfig(
        enabled=True,
        block_on_violation=True,
        categories=["hate_speech", "sexual", "violence", "self_harm", "bias", "illegal"],
    ),
    topic_guard=_TopicGuardConfig(blocked_topics=[
        FINANCIAL_ADVICE,
        LEGAL_ADVICE,
        TopicDefinition(
            name="insider_trading",
            keywords=[
                "insider information", "insider trading", "material non-public",
                "front-running", "pump and dump", "market manipulation",
                "wash trading", "spoofing",
            ],
            threshold=0.02,
        ),
    ]),
    secret_detection=_SecretConfig(enabled=True),
    output_safety=_OutputSafetyConfig(enabled=True),
)

# -- E-commerce (consumer protection aligned) ----------------------------------

ECOMMERCE_COMPLIANCE = ComplianceTemplate(
    name="ecommerce",
    description="Consumer protection guardrails: blocks PII, deceptive practices, and bias in product interactions.",
    pii=_PIIConfig(enabled=True, block_on_detection=True),
    content_filter=_ContentFilterConfig(
        enabled=True,
        block_on_violation=True,
        categories=["hate_speech", "sexual", "violence", "bias"],
    ),
    topic_guard=_TopicGuardConfig(blocked_topics=[
        TopicDefinition(
            name="deceptive_practices",
            keywords=[
                "fake review", "write a review", "generate reviews",
                "fake discount", "bait and switch", "hidden fees",
                "misleading price", "false advertising", "counterfeit",
            ],
            threshold=0.03,
        ),
        TopicDefinition(
            name="pricing_manipulation",
            keywords=[
                "charge more", "price discrimination", "dynamic pricing based on",
                "raise the price for", "different price for", "surge pricing",
            ],
            threshold=0.03,
        ),
    ]),
    secret_detection=_SecretConfig(enabled=True),
    output_safety=_OutputSafetyConfig(enabled=True),
)

# -- Insurance -----------------------------------------------------------------

INSURANCE_COMPLIANCE = ComplianceTemplate(
    name="insurance",
    description="Insurance regulation guardrails: blocks PII, unauthorized claim decisions, and discriminatory practices.",
    pii=_PIIConfig(enabled=True, block_on_detection=True),
    content_filter=_ContentFilterConfig(
        enabled=True,
        block_on_violation=True,
        categories=["hate_speech", "sexual", "violence", "bias"],
    ),
    topic_guard=_TopicGuardConfig(blocked_topics=[
        MEDICAL_ADVICE,
        LEGAL_ADVICE,
        TopicDefinition(
            name="claim_decision",
            keywords=[
                "your claim is denied", "claim approved", "claim rejected",
                "coverage denied", "not covered", "policy void",
                "cancel your policy", "terminate coverage",
                "pre-existing condition",
            ],
            threshold=0.03,
        ),
    ]),
    secret_detection=_SecretConfig(enabled=True),
    output_safety=_OutputSafetyConfig(enabled=True),
)
