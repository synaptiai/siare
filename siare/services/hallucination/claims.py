"""Claim extraction and verification for hallucination detection."""

import json
import logging
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from siare.services.hallucination.types import ClaimVerification
    from siare.services.llm_provider import LLMProvider

# Default model for claim extraction and verification
DEFAULT_MODEL = "gpt-4o-mini"

logger = logging.getLogger(__name__)

CLAIM_EXTRACTION_PROMPT = """Extract atomic factual claims from the following text.
Only extract verifiable factual statements, not opinions or questions.
Ignore hedged language like "might", "could", "approximately".

Text: {text}

Return claims as a JSON array of strings: ["claim 1", "claim 2", ...]
If no factual claims, return: []"""

CLAIM_VERIFICATION_PROMPT = """Verify if the following claim is supported by the context.

Claim: {claim}

Context: {context}

Analyze carefully and respond with JSON:
{{
    "verdict": "supported" | "contradicted" | "unverifiable",
    "score": 0.0-1.0 (confidence in verdict),
    "evidence": "quote or explanation from context",
    "reasoning": "brief explanation of your analysis"
}}

Guidelines:
- "supported": The context directly supports the claim with evidence
- "contradicted": The context contains information that contradicts the claim
- "unverifiable": The context neither supports nor contradicts the claim"""

# Minimum expected parts when splitting by ```
_MIN_CODE_BLOCK_PARTS = 2


def extract_claims(
    text: str, llm_provider: Optional["LLMProvider"], model: str = DEFAULT_MODEL
) -> list[str]:
    """Extract atomic factual claims from text using LLM.

    Args:
        text: Text to extract claims from
        llm_provider: LLM provider for extraction
        model: LLM model to use for extraction

    Returns:
        List of extracted claim strings

    Raises:
        RuntimeError: If llm_provider is None
    """
    if llm_provider is None:
        raise RuntimeError("LLM provider required for claim extraction")

    from siare.services.llm_provider import LLMMessage

    prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)
    messages = [LLMMessage(role="user", content=prompt)]

    response = llm_provider.complete(messages, model=model, temperature=0.0)

    return _parse_claims_response(response.content)


def _parse_claims_response(content: str) -> list[str]:
    """Parse LLM response to extract claims list.

    Handles:
    - Plain JSON arrays
    - Markdown code blocks (```json ... ```)
    - Malformed responses (returns empty list)
    """
    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("```")
        if len(lines) >= _MIN_CODE_BLOCK_PARTS:
            content = lines[1]
            # Remove language identifier (e.g., "json")
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

    try:
        claims: Any = json.loads(content)
        if not isinstance(claims, list):
            return []
        # Cast each claim to string to handle mixed types
        return [str(c) for c in claims]  # type: ignore[misc]
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse claims response: {content[:100]}")
        return []


def verify_claim(
    claim: str,
    context: str,
    llm_provider: Optional["LLMProvider"],
    model: str = DEFAULT_MODEL,
) -> "ClaimVerification":
    """Verify if a claim is supported by context using LLM.

    Args:
        claim: The factual claim to verify
        context: Retrieved context to verify against
        llm_provider: LLM provider for verification
        model: LLM model to use for verification

    Returns:
        ClaimVerification with verdict, score, and evidence

    Raises:
        RuntimeError: If llm_provider is None
    """
    if llm_provider is None:
        raise RuntimeError("LLM provider required for claim verification")

    from siare.services.llm_provider import LLMMessage

    prompt = CLAIM_VERIFICATION_PROMPT.format(claim=claim, context=context)
    messages = [LLMMessage(role="user", content=prompt)]

    response = llm_provider.complete(messages, model=model, temperature=0.0)

    return _parse_verification_response(claim, response.content)


def _parse_verification_response(claim: str, content: str) -> "ClaimVerification":
    """Parse LLM response to ClaimVerification.

    Handles malformed responses by returning unverifiable verdict.
    """
    from siare.services.hallucination.types import ClaimVerification

    content = content.strip()

    # Handle markdown code blocks
    if content.startswith("```"):
        lines = content.split("```")
        if len(lines) >= _MIN_CODE_BLOCK_PARTS:
            content = lines[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

    try:
        data = json.loads(content)
        verdict = data.get("verdict", "unverifiable")
        # Validate verdict is one of allowed values
        if verdict not in ("supported", "contradicted", "unverifiable"):
            verdict = "unverifiable"

        return ClaimVerification(
            claim=claim,
            score=float(data.get("score", 0.5)),
            verdict=verdict,
            evidence=data.get("evidence"),
            reasoning=data.get("reasoning"),
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        logger.warning(f"Failed to parse verification response: {content[:100]}")
        return ClaimVerification(
            claim=claim,
            score=0.5,
            verdict="unverifiable",
            evidence=None,
            reasoning="Failed to parse LLM response",
        )
