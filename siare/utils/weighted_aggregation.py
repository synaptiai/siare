"""Weighted task aggregation utilities for heterogeneous task sets"""

import logging
from typing import Any

import numpy as np
import numpy.typing as npt

from siare.core.constants import (
    MIN_SAMPLES_VARIANCE,
    WEIGHT_SUM_LOWER_BOUND,
    WEIGHT_SUM_UPPER_BOUND,
)
from siare.core.models import AggregatedMetric, AggregationMethod, EvaluationVector

logger = logging.getLogger(__name__)


def apply_task_weights(
    evaluations: list[EvaluationVector],
    aggregated_metrics: dict[str, AggregatedMetric],
    task_weights: list[float],
) -> dict[str, AggregatedMetric]:
    """
    Re-aggregate metrics with task-specific weights

    Each task contributes differently to the final metric based on its importance.
    Example: High-priority tasks get more weight than low-priority tasks.

    Args:
        evaluations: List of EvaluationVectors (one per task)
        aggregated_metrics: Unweighted aggregation results
        task_weights: Weight for each task (must sum to 1.0)

    Returns:
        Weighted aggregated metrics

    Raises:
        ValueError: If task_weights don't sum to 1.0 or length mismatch
    """
    if len(task_weights) != len(evaluations):
        raise ValueError(
            f"task_weights length ({len(task_weights)}) must match "
            f"evaluations length ({len(evaluations)})"
        )

    weight_sum = sum(task_weights)
    if not (WEIGHT_SUM_LOWER_BOUND <= weight_sum <= WEIGHT_SUM_UPPER_BOUND):
        raise ValueError(f"task_weights must sum to 1.0, got {weight_sum}")

    # Build task-level metric scores
    task_metrics: dict[str, list[float]] = {}

    for eval_vec in evaluations:
        for metric_result in eval_vec.metrics:
            metric_id = metric_result.metricId
            if metric_id not in task_metrics:
                task_metrics[metric_id] = []
            task_metrics[metric_id].append(metric_result.score)

    # Re-aggregate with weights
    weighted_aggregated: dict[str, AggregatedMetric] = {}

    for metric_id, scores in task_metrics.items():
        if len(scores) != len(task_weights):
            logger.warning(
                f"Metric {metric_id} has {len(scores)} scores but "
                f"{len(task_weights)} weights, skipping"
            )
            continue

        weighted_aggregated[metric_id] = _aggregate_with_weights(
            metric_id=metric_id,
            scores=scores,
            weights=task_weights,
        )

    return weighted_aggregated


def _aggregate_with_weights(
    metric_id: str,
    scores: list[float],
    weights: list[float],
) -> AggregatedMetric:
    """
    Aggregate scores with weights

    Uses weighted statistics:
    - Weighted mean: μ_w = Σ(w_i * x_i)
    - Weighted variance: σ²_w = Σ(w_i * (x_i - μ_w)²)
    - Effective sample size: n_eff = (Σw_i)² / Σ(w_i²)

    Args:
        metric_id: Metric identifier
        scores: Score values
        weights: Task weights (must sum to 1.0)

    Returns:
        AggregatedMetric with weighted statistics
    """

    scores_arr = np.array(scores)
    weights_arr = np.array(weights)

    # Weighted statistics
    weighted_mean = float(np.sum(weights_arr * scores_arr))

    # Weighted variance
    weighted_variance = float(np.sum(weights_arr * (scores_arr - weighted_mean) ** 2))
    weighted_std = weighted_variance**0.5

    # Effective sample size (for CI)
    # When weights are equal, n_eff = n
    # When weights are unequal, n_eff < n
    # Formula: n_eff = (Σw_i)² / Σ(w_i²)
    n_eff = (weights_arr.sum() ** 2) / (weights_arr**2).sum()
    weighted_se = weighted_std / (n_eff**0.5)

    # Weighted median (approximate via sorting)
    sorted_indices = np.argsort(scores_arr)
    sorted_scores = scores_arr[sorted_indices]
    sorted_weights = weights_arr[sorted_indices]
    cumsum_weights = np.cumsum(sorted_weights)
    median_idx = np.searchsorted(cumsum_weights, 0.5)
    weighted_median = float(sorted_scores[median_idx])

    # Bootstrap CI for weighted mean
    ci: tuple[float, float] | None = None
    if len(scores) >= MIN_SAMPLES_VARIANCE:
        try:
            # Bootstrap with weights (resample tasks with replacement)
            def weighted_mean_fn(indices: npt.NDArray[np.int_]) -> Any:
                return np.sum(weights_arr[indices] * scores_arr[indices])

            bootstrap_samples: list[float] = []
            rng = np.random.RandomState(42)
            for _ in range(10000):
                sampled_indices = rng.choice(len(scores), size=len(scores), replace=True)
                bootstrap_samples.append(weighted_mean_fn(sampled_indices))

            lower = float(np.percentile(bootstrap_samples, 2.5))
            upper = float(np.percentile(bootstrap_samples, 97.5))
            ci = (lower, upper)

        except Exception as e:
            logger.warning(f"Failed to compute weighted CI for {metric_id}: {e}")

    return AggregatedMetric(
        metricId=metric_id,
        mean=weighted_mean,
        median=weighted_median,
        standardDeviation=weighted_std,
        standardError=weighted_se,
        sampleSize=len(scores),
        confidenceInterval=ci,
        aggregationMethod=AggregationMethod.WEIGHTED,
        rawValues=scores,
    )
