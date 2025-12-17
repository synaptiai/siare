"""Baseline comparison framework for benchmark evaluation."""

from siare.benchmarks.comparison.baselines import (
    BaselineComparison,
    BaselineResult,
    RandomSearchBaseline,
    STATIC_BASELINE_CONFIGS,
    create_no_retrieval_baseline,
    create_static_baseline,
)

__all__ = [
    "BaselineComparison",
    "BaselineResult",
    "RandomSearchBaseline",
    "STATIC_BASELINE_CONFIGS",
    "create_no_retrieval_baseline",
    "create_static_baseline",
]
