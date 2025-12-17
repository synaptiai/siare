"""Retrieval metrics for evolution loop integration.

These metrics extract retrieval results from execution traces and
compute standard IR metrics (Recall, NDCG, MRR) for use in the
evolution fitness function.

The metrics can guide Director to optimize retrieval parameters
like top_k and similarity_threshold.
"""

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from siare.services.evaluation_service import EvaluationService
    from siare.services.execution_engine import ExecutionTrace

from siare.benchmarks.metrics.retrieval import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)

logger = logging.getLogger(__name__)


def _extract_retrieved_doc_ids(trace: "ExecutionTrace") -> list[str]:
    """Extract retrieved document IDs from execution trace.

    Args:
        trace: Execution trace from SOP run

    Returns:
        List of document IDs in retrieval order
    """
    doc_ids = []

    # Check tool_calls attribute
    if hasattr(trace, "tool_calls"):
        for tool_call in trace.tool_calls:
            if tool_call.get("tool") == "vector_search":
                results = tool_call.get("results", {}).get("results", [])
                for result in results:
                    if "id" in result:
                        doc_ids.append(result["id"])
                    elif "doc_id" in result:
                        doc_ids.append(result["doc_id"])

    # Fallback: check node_executions for tool results
    if not doc_ids and hasattr(trace, "node_executions"):
        for node_id, execution in trace.node_executions.items():
            if hasattr(execution, "tool_results"):
                for tool, result in execution.tool_results.items():
                    if tool == "vector_search":
                        for item in result.get("results", []):
                            if "id" in item:
                                doc_ids.append(item["id"])
                            elif "doc_id" in item:
                                doc_ids.append(item["doc_id"])

    return doc_ids


def _get_relevance_judgments(task_data: dict[str, Any]) -> dict[str, int]:
    """Extract relevance judgments from task data.

    Supports multiple formats:
    - relevance_scores: {"doc_id": score} (graded)
    - relevant_doc_ids: ["doc_id", ...] (binary)

    Args:
        task_data: Task data dictionary

    Returns:
        Dictionary mapping doc_id to relevance score
    """
    metadata = task_data.get("metadata", {})

    # Graded relevance
    if "relevance_scores" in metadata:
        return metadata["relevance_scores"]

    # Binary relevance from doc ID list
    if "relevant_doc_ids" in metadata:
        return dict.fromkeys(metadata["relevant_doc_ids"], 1)

    return {}


def retrieval_recall_at_k(
    trace: "ExecutionTrace",
    task_data: dict[str, Any],
    k: int = 10,
) -> float:
    """Compute Recall@K from execution trace.

    Args:
        trace: Execution trace from SOP run
        task_data: Task data with ground truth
        k: Cutoff position

    Returns:
        Recall@K score between 0 and 1
    """
    retrieved_docs = _extract_retrieved_doc_ids(trace)
    qrels = _get_relevance_judgments(task_data)

    if not qrels:
        logger.debug("No relevance judgments in task_data, returning 0")
        return 0.0

    return recall_at_k(retrieved_docs, qrels, k)


def retrieval_ndcg_at_k(
    trace: "ExecutionTrace",
    task_data: dict[str, Any],
    k: int = 10,
) -> float:
    """Compute NDCG@K from execution trace.

    Args:
        trace: Execution trace from SOP run
        task_data: Task data with ground truth
        k: Cutoff position

    Returns:
        NDCG@K score between 0 and 1
    """
    retrieved_docs = _extract_retrieved_doc_ids(trace)
    qrels = _get_relevance_judgments(task_data)

    if not qrels:
        logger.debug("No relevance judgments in task_data, returning 0")
        return 0.0

    return ndcg_at_k(retrieved_docs, qrels, k)


def retrieval_mrr(
    trace: "ExecutionTrace",
    task_data: dict[str, Any],
) -> float:
    """Compute Mean Reciprocal Rank from execution trace.

    Args:
        trace: Execution trace from SOP run
        task_data: Task data with ground truth

    Returns:
        MRR score between 0 and 1
    """
    retrieved_docs = _extract_retrieved_doc_ids(trace)
    qrels = _get_relevance_judgments(task_data)

    if not qrels:
        logger.debug("No relevance judgments in task_data, returning 0")
        return 0.0

    return mean_reciprocal_rank(retrieved_docs, qrels)


def register_retrieval_evolution_metrics(
    evaluation_service: "EvaluationService",
) -> None:
    """Register retrieval metrics with EvaluationService for evolution.

    Args:
        evaluation_service: EvaluationService to register metrics with
    """
    # Create metric functions with bound k values
    def recall_at_5(trace: "ExecutionTrace", task_data: dict) -> float:
        return retrieval_recall_at_k(trace, task_data, k=5)

    def recall_at_10(trace: "ExecutionTrace", task_data: dict) -> float:
        return retrieval_recall_at_k(trace, task_data, k=10)

    def ndcg_at_5(trace: "ExecutionTrace", task_data: dict) -> float:
        return retrieval_ndcg_at_k(trace, task_data, k=5)

    def ndcg_at_10(trace: "ExecutionTrace", task_data: dict) -> float:
        return retrieval_ndcg_at_k(trace, task_data, k=10)

    def mrr(trace: "ExecutionTrace", task_data: dict) -> float:
        return retrieval_mrr(trace, task_data)

    # Register with evaluation service
    evaluation_service.register_metric_function("retrieval_recall_at_5", recall_at_5)
    evaluation_service.register_metric_function("retrieval_recall_at_10", recall_at_10)
    evaluation_service.register_metric_function("retrieval_ndcg_at_5", ndcg_at_5)
    evaluation_service.register_metric_function("retrieval_ndcg_at_10", ndcg_at_10)
    evaluation_service.register_metric_function("retrieval_mrr", mrr)

    logger.info("Registered retrieval evolution metrics")


# Metric definitions for evolution job configuration
RETRIEVAL_METRIC_CONFIGS = {
    "retrieval_recall_at_5": {
        "id": "retrieval_recall_at_5",
        "type": "programmatic",
        "weight": 0.15,
        "description": "Fraction of relevant docs found in top 5",
    },
    "retrieval_recall_at_10": {
        "id": "retrieval_recall_at_10",
        "type": "programmatic",
        "weight": 0.20,
        "description": "Fraction of relevant docs found in top 10",
    },
    "retrieval_ndcg_at_10": {
        "id": "retrieval_ndcg_at_10",
        "type": "programmatic",
        "weight": 0.25,
        "description": "Normalized DCG at rank 10",
    },
    "retrieval_mrr": {
        "id": "retrieval_mrr",
        "type": "programmatic",
        "weight": 0.15,
        "description": "Mean Reciprocal Rank",
    },
}
