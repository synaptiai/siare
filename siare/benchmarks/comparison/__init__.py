"""Baseline comparison framework for benchmark evaluation."""

from siare.benchmarks.comparison.baselines import (
    STATIC_BASELINE_CONFIGS,
    BaselineComparison,
    BaselineResult,
    RandomSearchBaseline,
    create_no_retrieval_baseline,
    create_static_baseline,
)

__all__ = [
    "STATIC_BASELINE_CONFIGS",
    "BaselineComparison",
    "BaselineResult",
    "RandomSearchBaseline",
    "create_no_retrieval_baseline",
    "create_static_baseline",
]
