"""Retrieval and evolution metrics for benchmark evaluation."""

from siare.benchmarks.metrics.evolution_metrics import (
    EvolutionMetrics,
    benchmark_accuracy,
    benchmark_f1,
    benchmark_partial_match,
    extract_generated_answer,
    normalize_text,
    register_benchmark_metrics,
)
from siare.benchmarks.metrics.integration import (
    RetrievalEvaluator,
    extract_retrieved_docs_from_trace,
)
from siare.benchmarks.metrics.retrieval import (
    STANDARD_K_VALUES,
    RetrievalMetrics,
    average_precision,
    evaluate_retrieval,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from siare.benchmarks.metrics.retrieval_evolution import (
    RETRIEVAL_METRIC_CONFIGS,
    register_retrieval_evolution_metrics,
    retrieval_mrr,
    retrieval_ndcg_at_k,
    retrieval_recall_at_k,
)

__all__ = [
    # Retrieval metrics
    "STANDARD_K_VALUES",
    "RetrievalMetrics",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "average_precision",
    "evaluate_retrieval",
    # Integration
    "RetrievalEvaluator",
    "extract_retrieved_docs_from_trace",
    # Evolution metrics
    "EvolutionMetrics",
    "benchmark_accuracy",
    "benchmark_f1",
    "benchmark_partial_match",
    "extract_generated_answer",
    "normalize_text",
    "register_benchmark_metrics",
    # Retrieval evolution metrics
    "RETRIEVAL_METRIC_CONFIGS",
    "register_retrieval_evolution_metrics",
    "retrieval_recall_at_k",
    "retrieval_ndcg_at_k",
    "retrieval_mrr",
]
