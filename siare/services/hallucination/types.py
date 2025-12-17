"""Data types for hallucination detection."""

from dataclasses import dataclass, field
from typing import Any, Literal


def _empty_str_list() -> list[str]:
    return []


def _empty_claim_list() -> list["ClaimVerification"]:
    return []


def _empty_dict_list() -> list[dict[str, Any]]:
    return []


@dataclass
class ClaimVerification:
    """Result of verifying a single claim against context."""

    claim: str
    score: float  # 0.0 to 1.0
    verdict: Literal["supported", "contradicted", "unverifiable"]
    evidence: str | None = None
    reasoning: str | None = None


@dataclass
class FaithfulnessResult:
    """Aggregate result of faithfulness evaluation."""

    score: float
    claim_count: int
    supported_count: int
    hallucination_count: int
    unsupported_claims: list[str] = field(default_factory=_empty_str_list)
    claim_verifications: list[ClaimVerification] = field(default_factory=_empty_claim_list)


@dataclass
class GroundednessResult:
    """Result of groundedness evaluation."""

    score: float
    grounded_claims: int
    ungrounded_claims: int
    source_coverage: float  # Percentage of sources used
    missing_attributions: list[str] = field(default_factory=_empty_str_list)


@dataclass
class CitationResult:
    """Result of citation accuracy evaluation."""

    score: float
    total_citations: int
    accurate_citations: int
    inaccurate_citations: list[dict[str, Any]] = field(default_factory=_empty_dict_list)
