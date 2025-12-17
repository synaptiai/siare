"""Hallucination detection and claim verification services."""

from siare.services.hallucination.citation_accuracy import (
    CitationChecker,
    citation_accuracy_metric,
)
from siare.services.hallucination.claims import extract_claims, verify_claim
from siare.services.hallucination.faithfulness import (
    FaithfulnessChecker,
    faithfulness_metric,
)
from siare.services.hallucination.groundedness import (
    GroundednessChecker,
    groundedness_metric,
)
from siare.services.hallucination.ragas_adapter import (
    RAGAS_AVAILABLE,
    RAGASAdapter,
    ragas_faithfulness_metric,
)
from siare.services.hallucination.types import (
    CitationResult,
    ClaimVerification,
    FaithfulnessResult,
    GroundednessResult,
)

__all__ = [
    "RAGAS_AVAILABLE",
    "CitationChecker",
    "CitationResult",
    "ClaimVerification",
    "FaithfulnessChecker",
    "FaithfulnessResult",
    "GroundednessChecker",
    "GroundednessResult",
    "RAGASAdapter",
    "citation_accuracy_metric",
    "extract_claims",
    "faithfulness_metric",
    "groundedness_metric",
    "ragas_faithfulness_metric",
    "verify_claim",
]
