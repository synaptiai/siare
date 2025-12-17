"""Integration layer for retrieval metrics with benchmark runner."""

import logging
from typing import Any, Optional

from siare.benchmarks.metrics.retrieval import (
    RetrievalMetrics,
    evaluate_retrieval,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)


logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """Evaluates retrieval quality across a benchmark dataset.

    Computes aggregated metrics (NDCG, Recall, MRR) across all queries.

    Example:
        >>> evaluator = RetrievalEvaluator()
        >>> metrics = evaluator.evaluate(retrieval_results, qrels)
        >>> print(f"NDCG@10: {metrics['ndcg@10']:.3f}")
    """

    def __init__(self, k_values: Optional[list[int]] = None) -> None:
        """Initialize evaluator.

        Args:
            k_values: K values for metrics (default: [1, 3, 5, 10])
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def evaluate(
        self,
        retrieval_results: dict[str, list[str]],
        qrels: dict[str, dict[str, int]],
        k_values: Optional[list[int]] = None,
    ) -> dict[str, float]:
        """Compute retrieval metrics across all queries.

        Args:
            retrieval_results: {query_id: [doc_id, ...]} ranked lists
            qrels: {query_id: {doc_id: relevance_score}} ground truth
            k_values: K values to evaluate at (overrides default)

        Returns:
            Dict of metric_name -> aggregated score
        """
        k_values = k_values or self.k_values

        # Collect per-query metrics
        all_ndcg: dict[int, list[float]] = {k: [] for k in k_values}
        all_recall: dict[int, list[float]] = {k: [] for k in k_values}
        all_mrr: list[float] = []

        for query_id, retrieved_docs in retrieval_results.items():
            query_qrels = qrels.get(query_id, {})

            # Skip if no relevance judgments for this query
            if not query_qrels:
                logger.debug(f"No qrels for query {query_id}, using zero scores")
                for k in k_values:
                    all_ndcg[k].append(0.0)
                    all_recall[k].append(0.0)
                all_mrr.append(0.0)
                continue

            # Compute metrics
            for k in k_values:
                all_ndcg[k].append(ndcg_at_k(retrieved_docs, query_qrels, k))
                all_recall[k].append(recall_at_k(retrieved_docs, query_qrels, k))

            all_mrr.append(mean_reciprocal_rank(retrieved_docs, query_qrels))

        # Aggregate (mean across queries)
        metrics: dict[str, float] = {}

        for k in k_values:
            if all_ndcg[k]:
                metrics[f"ndcg@{k}"] = sum(all_ndcg[k]) / len(all_ndcg[k])
            else:
                metrics[f"ndcg@{k}"] = 0.0

            if all_recall[k]:
                metrics[f"recall@{k}"] = sum(all_recall[k]) / len(all_recall[k])
            else:
                metrics[f"recall@{k}"] = 0.0

        if all_mrr:
            metrics["mrr"] = sum(all_mrr) / len(all_mrr)
        else:
            metrics["mrr"] = 0.0

        return metrics

    def evaluate_single_query(
        self,
        retrieved_docs: list[str],
        qrels: dict[str, int],
        k_values: Optional[list[int]] = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval for a single query.

        Args:
            retrieved_docs: Ranked list of document IDs
            qrels: {doc_id: relevance_score} for this query
            k_values: K values to evaluate at

        Returns:
            RetrievalMetrics with all scores
        """
        return evaluate_retrieval(
            retrieved_docs,
            qrels,
            k_values=k_values or self.k_values,
        )


def extract_retrieved_docs_from_trace(
    trace: Any,
    tool_name: str = "vector_search",
) -> list[str]:
    """Extract retrieved document IDs from an execution trace.

    Args:
        trace: ExecutionTrace from benchmark run
        tool_name: Name of retrieval tool to look for

    Returns:
        List of retrieved document IDs
    """
    doc_ids = []

    # Check if trace has tool_calls attribute
    if hasattr(trace, "tool_calls"):
        for tool_call in trace.tool_calls:
            if tool_call.get("tool") == tool_name:
                results = tool_call.get("results", {}).get("results", [])
                for result in results:
                    if "id" in result:
                        doc_ids.append(result["id"])

    # Fallback: check node executions for tool results
    # node_executions is a list[dict], not a dict
    if not doc_ids and hasattr(trace, "node_executions"):
        for execution in trace.node_executions:
            if isinstance(execution, dict):
                tool_results = execution.get("tool_results", {})
                if isinstance(tool_results, dict):
                    for tool, result in tool_results.items():
                        if tool == tool_name and isinstance(result, dict):
                            for item in result.get("results", []):
                                if isinstance(item, dict) and "id" in item:
                                    doc_ids.append(item["id"])

    return doc_ids
