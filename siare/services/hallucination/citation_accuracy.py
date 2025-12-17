"""Citation Accuracy Metric - Verify citations match sources."""

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from siare.services.execution_engine import ExecutionTrace
    from siare.services.llm_provider import LLMProvider

from siare.services.hallucination.types import CitationResult


logger = logging.getLogger(__name__)

CITATION_VERIFICATION_PROMPT = """Verify if the citation accurately reflects the source content.

Citation text: {citation_text}
Source content: {source_content}

Respond with JSON:
{{
    "accurate": true | false,
    "reasoning": "explanation"
}}"""


class CitationChecker:
    """Checks if citations accurately reference their sources."""

    def __init__(self, llm_provider: Optional["LLMProvider"]):
        if llm_provider is None:
            raise RuntimeError("LLM provider required for citation checking")
        self._llm_provider = llm_provider

    def evaluate(self, answer: str, sources: dict[str, str]) -> CitationResult:
        """Evaluate citation accuracy in answer."""
        citations = self._extract_citations(answer)

        if not citations:
            return CitationResult(
                score=1.0,
                total_citations=0,
                accurate_citations=0,
            )

        accurate = 0
        inaccurate: list[dict[str, Any]] = []

        for citation_id, citation_text in citations.items():
            source = sources.get(citation_id, "")
            if not source:
                inaccurate.append({"citation": citation_id, "reason": "Source not found"})
                continue

            is_accurate = self._verify_citation(citation_text, source)
            if is_accurate:
                accurate += 1
            else:
                inaccurate.append({"citation": citation_id, "reason": "Does not match source"})

        score = accurate / len(citations) if citations else 1.0

        return CitationResult(
            score=score,
            total_citations=len(citations),
            accurate_citations=accurate,
            inaccurate_citations=inaccurate,
        )

    def _extract_citations(self, text: str) -> dict[str, str]:
        """Extract citations from text (e.g., [1], [2])."""
        pattern = r"\[(\d+)\]([^[]*)"
        matches = re.findall(pattern, text)
        return {m[0]: m[1].strip() for m in matches if m[1].strip()}

    def _verify_citation(self, citation_text: str, source: str) -> bool:
        """Verify if citation accurately reflects source."""
        from siare.services.llm_provider import LLMMessage

        prompt = CITATION_VERIFICATION_PROMPT.format(
            citation_text=citation_text,
            source_content=source,
        )
        messages = [LLMMessage(role="user", content=prompt)]
        response = self._llm_provider.complete(messages, temperature=0.0)

        try:
            data = json.loads(response.content)
            return data.get("accurate", False)
        except json.JSONDecodeError:
            return False


def _extract_sources_from_trace(
    trace: "ExecutionTrace",
    sources_key: str,
) -> dict[str, str]:
    """Extract sources from trace node executions.

    Maps source indices (1, 2, 3...) to source content for citation verification.

    Args:
        trace: Execution trace to search
        sources_key: Output key to look for (e.g., "documents", "sources")

    Returns:
        Dictionary mapping citation IDs ("1", "2"...) to source content
    """
    sources: dict[str, str] = {}
    source_idx = 1

    for node in trace.node_executions:
        outputs: dict[str, Any] = node.get("outputs", {})
        if sources_key in outputs:
            source_data: Any = outputs[sources_key]
            if isinstance(source_data, list):
                for item in source_data:  # type: ignore[reportUnknownVariableType]
                    sources[str(source_idx)] = str(item)  # type: ignore[reportUnknownArgumentType]
                    source_idx += 1
            elif isinstance(source_data, dict):
                # Handle dict format like {"1": "content", "2": "content"}
                for key, value in source_data.items():  # type: ignore[reportUnknownVariableType]
                    sources[str(key)] = str(value)  # type: ignore[reportUnknownArgumentType]
            else:
                sources[str(source_idx)] = str(source_data)
                source_idx += 1

    return sources


def citation_accuracy_metric(trace: "ExecutionTrace", task_data: dict[str, Any]) -> float:
    """Programmatic metric for EvaluationService registration.

    Evaluates whether citations in the answer accurately reference sources.

    Requires task_data to contain:
        - llm_provider: LLMProvider instance
        - answer_key: Key in final_outputs containing answer (default: "answer")
        - sources_key: Key in node outputs containing sources (default: "documents")

    Returns:
        Citation accuracy score between 0.0 and 1.0

    Raises:
        RuntimeError: If llm_provider not in task_data
    """
    llm_provider = task_data.get("llm_provider")
    if llm_provider is None:
        raise RuntimeError("citation_accuracy metric requires 'llm_provider' in task_data")

    answer_key = task_data.get("answer_key", "answer")
    sources_key = task_data.get("sources_key", "documents")

    answer = trace.final_outputs.get(answer_key, "")
    if not answer:
        return 1.0  # No answer means nothing to evaluate

    # Return 1.0 if no citation pattern detected (no citations to verify)
    if not re.search(r"\[\d+\]", answer):
        return 1.0

    # Extract sources from trace
    sources = _extract_sources_from_trace(trace, sources_key)
    if not sources:
        # No sources found - citations cannot be verified
        # Count citations and penalize proportionally
        citation_count = len(re.findall(r"\[\d+\]", answer))
        return 0.0 if citation_count > 0 else 1.0

    checker = CitationChecker(llm_provider)
    result = checker.evaluate(answer, sources)
    return result.score
