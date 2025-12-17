"""
Standard retrieval metrics for BEIR-compatible evaluation.

Implements: NDCG@K, MAP@K, Recall@K, Precision@K, MRR
Standard k-values: [1, 3, 5, 10, 100, 1000]
"""

import math
from dataclasses import dataclass


STANDARD_K_VALUES = [1, 3, 5, 10, 100, 1000]


@dataclass
class RetrievalMetrics:
    """Results at all standard k-values."""

    ndcg: dict[int, float]
    map: dict[int, float]
    recall: dict[int, float]
    precision: dict[int, float]
    mrr: float


def dcg_at_k(relevances: list[int], k: int) -> float:
    """Compute Discounted Cumulative Gain at K.

    Args:
        relevances: List of relevance scores in retrieved order
        k: Cutoff position

    Returns:
        DCG score
    """
    relevances = relevances[:k]
    return sum(
        rel / math.log2(i + 2) for i, rel in enumerate(relevances) if rel > 0
    )


def ndcg_at_k(
    retrieved_docs: list[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    """Compute Normalized DCG at K.

    Args:
        retrieved_docs: Ranked list of document IDs
        qrels: Query-document relevance judgments {doc_id: relevance_score}
        k: Cutoff position

    Returns:
        NDCG@K score between 0.0 and 1.0
    """
    relevances = [qrels.get(doc_id, 0) for doc_id in retrieved_docs[:k]]
    dcg = dcg_at_k(relevances, k)

    # Ideal DCG (sorted relevances)
    ideal_relevances = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_relevances, k)

    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(
    retrieved_docs: list[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    """Compute Precision at K.

    Args:
        retrieved_docs: Ranked list of document IDs
        qrels: Query-document relevance judgments
        k: Cutoff position

    Returns:
        Precision@K score between 0.0 and 1.0
    """
    retrieved = retrieved_docs[:k]
    relevant = sum(1 for doc_id in retrieved if qrels.get(doc_id, 0) > 0)
    return relevant / k if k > 0 else 0.0


def recall_at_k(
    retrieved_docs: list[str],
    qrels: dict[str, int],
    k: int,
) -> float:
    """Compute Recall at K.

    Args:
        retrieved_docs: Ranked list of document IDs
        qrels: Query-document relevance judgments
        k: Cutoff position

    Returns:
        Recall@K score between 0.0 and 1.0
    """
    total_relevant = sum(1 for rel in qrels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0

    retrieved = retrieved_docs[:k]
    found_relevant = sum(1 for doc_id in retrieved if qrels.get(doc_id, 0) > 0)
    return found_relevant / total_relevant


def mean_reciprocal_rank(
    retrieved_docs: list[str],
    qrels: dict[str, int],
) -> float:
    """Compute Mean Reciprocal Rank.

    Args:
        retrieved_docs: Ranked list of document IDs
        qrels: Query-document relevance judgments

    Returns:
        MRR score between 0.0 and 1.0
    """
    for i, doc_id in enumerate(retrieved_docs):
        if qrels.get(doc_id, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def average_precision(
    retrieved_docs: list[str],
    qrels: dict[str, int],
) -> float:
    """Compute Average Precision for a single query.

    Args:
        retrieved_docs: Ranked list of document IDs
        qrels: Query-document relevance judgments

    Returns:
        AP score between 0.0 and 1.0
    """
    total_relevant = sum(1 for rel in qrels.values() if rel > 0)
    if total_relevant == 0:
        return 0.0

    precisions = []
    relevant_found = 0

    for i, doc_id in enumerate(retrieved_docs):
        if qrels.get(doc_id, 0) > 0:
            relevant_found += 1
            precisions.append(relevant_found / (i + 1))

    return sum(precisions) / total_relevant if precisions else 0.0


def evaluate_retrieval(
    retrieved_docs: list[str],
    qrels: dict[str, int],
    k_values: list[int] | None = None,
) -> RetrievalMetrics:
    """Evaluate retrieval against ground truth relevance judgments.

    Args:
        retrieved_docs: Ranked list of document IDs
        qrels: Query-document relevance judgments {doc_id: relevance_score}
        k_values: List of k-values to evaluate at (default: STANDARD_K_VALUES)

    Returns:
        RetrievalMetrics with scores at all k-values
    """
    if k_values is None:
        k_values = STANDARD_K_VALUES

    return RetrievalMetrics(
        ndcg={k: ndcg_at_k(retrieved_docs, qrels, k) for k in k_values},
        map={k: average_precision(retrieved_docs[:k], qrels) for k in k_values},
        recall={k: recall_at_k(retrieved_docs, qrels, k) for k in k_values},
        precision={k: precision_at_k(retrieved_docs, qrels, k) for k in k_values},
        mrr=mean_reciprocal_rank(retrieved_docs, qrels),
    )
