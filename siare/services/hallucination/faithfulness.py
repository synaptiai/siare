"""Faithfulness evaluation for RAG systems."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from siare.services.hallucination.claims import extract_claims, verify_claim
from siare.services.hallucination.types import ClaimVerification, FaithfulnessResult


if TYPE_CHECKING:
    from siare.services.execution_engine import ExecutionTrace
    from siare.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


DEFAULT_MODEL = "gpt-4o-mini"


class FaithfulnessChecker:
    """Evaluates faithfulness of generated text against source context."""

    def __init__(
        self, llm_provider: Optional["LLMProvider"], model: str = DEFAULT_MODEL
    ):
        """Initialize faithfulness checker.

        Args:
            llm_provider: LLM provider for claim extraction and verification
            model: LLM model to use for evaluation

        Raises:
            RuntimeError: If llm_provider is None
        """
        if llm_provider is None:
            raise RuntimeError("LLM provider required for faithfulness checking")
        self.llm_provider = llm_provider
        self.model = model

    def evaluate(self, answer: str, context: str) -> FaithfulnessResult:
        """Evaluate faithfulness of an answer against context.

        Args:
            answer: Generated answer to evaluate
            context: Source context that should support the answer

        Returns:
            FaithfulnessResult with score and detailed claim analysis
        """
        # Extract claims from answer
        claims = extract_claims(answer, self.llm_provider, model=self.model)

        if not claims:
            # No claims means nothing to verify - perfect faithfulness
            return FaithfulnessResult(
                score=1.0,
                claim_count=0,
                supported_count=0,
                hallucination_count=0,
                unsupported_claims=[],
                claim_verifications=[],
            )

        # Verify each claim against context
        verifications: list[ClaimVerification] = []
        supported_count = 0
        hallucination_count = 0
        unsupported_claims: list[str] = []

        for claim in claims:
            verification = verify_claim(claim, context, self.llm_provider, model=self.model)
            verifications.append(verification)

            if verification.verdict == "supported":
                supported_count += 1
            elif verification.verdict == "contradicted":
                hallucination_count += 1
                unsupported_claims.append(claim)
            else:  # unverifiable
                unsupported_claims.append(claim)

        # Calculate faithfulness score
        # Score is the ratio of supported claims to total claims
        score = supported_count / len(claims) if claims else 1.0

        return FaithfulnessResult(
            score=score,
            claim_count=len(claims),
            supported_count=supported_count,
            hallucination_count=hallucination_count,
            unsupported_claims=unsupported_claims,
            claim_verifications=verifications,
        )


def faithfulness_metric(trace: "ExecutionTrace", task_data: dict[str, Any]) -> float:
    """Programmatic metric for EvaluationService registration.

    This function provides a standard interface for the EvaluationService
    to evaluate faithfulness of RAG pipeline outputs.

    Args:
        trace: Execution trace containing answer and context
        task_data: Must contain:
            - llm_provider: LLMProvider instance (required)
            - context_key: Key in node outputs containing context (default: "documents")
            - answer_key: Key in final_outputs containing answer (default: "answer")

    Returns:
        Faithfulness score between 0.0 and 1.0

    Raises:
        RuntimeError: If llm_provider not in task_data

    Example:
        >>> evaluation_service.register_metric_function(
        ...     "faithfulness", faithfulness_metric
        ... )
    """

    llm_provider = task_data.get("llm_provider")
    if llm_provider is None:
        raise RuntimeError("faithfulness metric requires 'llm_provider' in task_data")

    context_key = task_data.get("context_key", "documents")
    answer_key = task_data.get("answer_key", "answer")

    # Extract answer from trace
    answer = trace.final_outputs.get(answer_key, "")
    if not answer:
        return 1.0  # No answer means nothing to evaluate

    # Extract context from trace node executions
    context = _extract_context_from_trace(trace, context_key)
    if not context:
        return 0.0  # No context means we can't verify

    # Run faithfulness check
    checker = FaithfulnessChecker(llm_provider)
    result = checker.evaluate(answer, context)

    return result.score


def _extract_context_from_trace(trace: "ExecutionTrace", key: str) -> str:
    """Extract context from trace node executions.

    Searches through node_executions for the specified output key
    and concatenates any found context.

    Args:
        trace: Execution trace to search
        key: Output key to look for (e.g., "documents")

    Returns:
        Concatenated context string, or empty string if not found
    """

    context_parts: list[str] = []

    for node in trace.node_executions:
        outputs = node.get("outputs", {})
        if key in outputs:
            ctx = outputs[key]
            if isinstance(ctx, list):
                # Convert each item to string
                # Type ignore: ctx comes from dynamic execution trace structure
                context_parts.extend([str(item) for item in ctx])  # type: ignore[reportUnknownArgumentType,reportUnknownVariableType]
            else:
                context_parts.append(str(ctx))

    return "\n\n".join(context_parts)
