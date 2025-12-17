"""Optional RAGAS integration for hallucination metrics.

RAGAS (Retrieval Augmented Generation Assessment) is a framework
for evaluating RAG pipelines. This adapter provides optional
integration with SIARE's evaluation system.

To use RAGAS metrics, install: pip install ragas

Example:
    >>> from siare.services.hallucination.ragas_adapter import RAGASAdapter, RAGAS_AVAILABLE
    >>> if RAGAS_AVAILABLE:
    ...     adapter = RAGASAdapter()
    ...     score = adapter.evaluate_faithfulness("What is ML?", "ML is...", ["Doc 1"])
"""
import logging
from typing import Any


logger = logging.getLogger(__name__)


def _check_ragas_available() -> bool:
    """Check if RAGAS library is available."""
    try:
        import ragas  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        return False
    else:
        return True


RAGAS_AVAILABLE: bool = _check_ragas_available()

if not RAGAS_AVAILABLE:
    logger.info("RAGAS not installed. Install with: pip install ragas")


class RAGASAdapter:
    """Adapter to use RAGAS metrics with SIARE EvaluationService.

    RAGAS provides additional hallucination detection metrics that can
    complement SIARE's built-in metrics.

    Raises:
        ImportError: If RAGAS is not installed
    """

    def __init__(self) -> None:
        """Initialize the RAGAS adapter.

        Raises:
            ImportError: If RAGAS is not installed
        """
        if not RAGAS_AVAILABLE:
            raise ImportError(
                "RAGAS not available. Install with: pip install ragas"
            )

    def _evaluate_metric(
        self,
        metric_name: str,
        data: dict[str, Any],
    ) -> float:
        """Generic RAGAS metric evaluation.

        Args:
            metric_name: Name of the RAGAS metric (e.g., "faithfulness")
            data: Data dict for Dataset creation

        Returns:
            Score between 0.0 and 1.0

        Raises:
            RuntimeError: If RAGAS evaluation fails
        """
        try:
            import ragas.metrics as ragas_metrics  # type: ignore[import-not-found]
            from datasets import Dataset  # type: ignore[import-not-found]
            from ragas import evaluate  # type: ignore[import-not-found]

            metric = getattr(ragas_metrics, metric_name)
            dataset = Dataset.from_dict(data)  # type: ignore[reportUnknownMemberType]
            result = evaluate(dataset, metrics=[metric])  # type: ignore[reportUnknownVariableType]
            return float(result[metric_name])  # type: ignore[reportUnknownArgumentType]

        except (ImportError, KeyError, ValueError, TypeError, AttributeError) as e:
            logger.error(f"RAGAS {metric_name} evaluation failed: {e}")
            raise RuntimeError(f"RAGAS {metric_name} evaluation failed: {e}") from e

    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> float:
        """Evaluate faithfulness using RAGAS metric.

        Args:
            question: The question asked
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Faithfulness score between 0.0 and 1.0

        Raises:
            RuntimeError: If RAGAS evaluation fails
        """
        return self._evaluate_metric("faithfulness", {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        })

    def evaluate_context_relevancy(
        self,
        question: str,
        contexts: list[str],
    ) -> float:
        """Evaluate context relevancy using RAGAS metric.

        Args:
            question: The question asked
            contexts: List of retrieved context documents

        Returns:
            Context relevancy score between 0.0 and 1.0

        Raises:
            RuntimeError: If RAGAS evaluation fails
        """
        return self._evaluate_metric("context_relevancy", {
            "question": [question],
            "contexts": [contexts],
        })

    def evaluate_answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """Evaluate answer relevancy using RAGAS metric.

        Args:
            question: The question asked
            answer: The generated answer

        Returns:
            Answer relevancy score between 0.0 and 1.0

        Raises:
            RuntimeError: If RAGAS evaluation fails
        """
        return self._evaluate_metric("answer_relevancy", {
            "question": [question],
            "answer": [answer],
        })


def ragas_faithfulness_metric(trace: Any, task_data: dict[str, Any]) -> float:
    """RAGAS-based faithfulness metric for EvaluationService.

    Requires RAGAS to be installed. If not available, returns 0.0.

    Args:
        trace: ExecutionTrace with answer and context
        task_data: Task data (optional, not used by RAGAS)

    Returns:
        Faithfulness score from RAGAS, or 0.0 if RAGAS unavailable
    """
    if not RAGAS_AVAILABLE:
        logger.warning("RAGAS not available, returning 0.0")
        return 0.0

    answer_key = task_data.get("answer_key", "answer")
    context_key = task_data.get("context_key", "documents")
    query_key = task_data.get("query_key", "query")

    answer = trace.final_outputs.get(answer_key, "")
    query = task_data.get(query_key, "")

    # Extract contexts from trace
    contexts: list[str] = []
    for node in trace.node_executions:
        outputs: dict[str, Any] = node.get("outputs", {})
        if context_key in outputs:
            ctx: Any = outputs[context_key]
            if isinstance(ctx, list):
                contexts.extend(str(c) for c in ctx)  # type: ignore[reportUnknownVariableType]
            else:
                contexts.append(str(ctx))

    if not answer or not contexts:
        return 0.0

    adapter = RAGASAdapter()
    return adapter.evaluate_faithfulness(query, answer, contexts)
