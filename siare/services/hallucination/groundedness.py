"""Groundedness Metric - Verify claims can be traced to sources."""

import logging
from typing import TYPE_CHECKING, Any, Optional

from siare.services.hallucination.claims import extract_claims, verify_claim
from siare.services.hallucination.types import GroundednessResult


if TYPE_CHECKING:
    from siare.services.execution_engine import ExecutionTrace
    from siare.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


class GroundednessChecker:
    """Checks if claims can be traced to specific sources."""

    def __init__(self, llm_provider: Optional["LLMProvider"]):
        if llm_provider is None:
            raise RuntimeError("LLM provider required for groundedness checking")
        self._llm_provider = llm_provider

    def evaluate(self, answer: str, sources: list[str]) -> GroundednessResult:
        """Evaluate groundedness of answer against sources."""
        claims = extract_claims(answer, self._llm_provider)

        if not claims:
            return GroundednessResult(
                score=1.0,
                grounded_claims=0,
                ungrounded_claims=0,
                source_coverage=1.0,
            )

        # Combine sources as context
        context = "\n\n".join(sources)
        grounded = 0
        missing_attributions: list[str] = []

        for claim in claims:
            verification = verify_claim(claim, context, self._llm_provider)
            if verification.verdict == "supported":
                grounded += 1
            else:
                missing_attributions.append(claim)

        score = grounded / len(claims) if claims else 1.0
        source_coverage = 1.0 if grounded == len(claims) else grounded / len(claims)

        return GroundednessResult(
            score=score,
            grounded_claims=grounded,
            ungrounded_claims=len(claims) - grounded,
            source_coverage=source_coverage,
            missing_attributions=missing_attributions,
        )


def groundedness_metric(trace: "ExecutionTrace", task_data: dict[str, Any]) -> float:
    """Programmatic metric for EvaluationService registration."""
    llm_provider = task_data.get("llm_provider")
    if llm_provider is None:
        raise RuntimeError("groundedness metric requires 'llm_provider' in task_data")

    answer_key = task_data.get("answer_key", "answer")
    context_key = task_data.get("context_key", "documents")

    answer = trace.final_outputs.get(answer_key, "")
    if not answer:
        return 1.0

    sources = _extract_sources_from_trace(trace, context_key)
    if not sources:
        return 0.0

    checker = GroundednessChecker(llm_provider)
    result = checker.evaluate(answer, sources)
    return result.score


def _extract_sources_from_trace(trace: "ExecutionTrace", key: str) -> list[str]:
    """Extract sources from trace node executions."""
    sources: list[str] = []
    for node in trace.node_executions:
        outputs = node.get("outputs", {})
        if key in outputs:
            ctx = outputs[key]
            if isinstance(ctx, list):
                # Type ignore: ctx comes from dynamic execution trace structure
                for c in ctx:  # type: ignore[reportUnknownVariableType]
                    sources.append(str(c))  # type: ignore[reportUnknownArgumentType]
            else:
                sources.append(str(ctx))
    return sources
