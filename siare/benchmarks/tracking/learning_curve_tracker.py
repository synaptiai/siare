"""Learning curve tracking for evolution analysis.

Tracks performance metrics across evolution generations to show
learning progress and detect convergence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from siare.utils.statistics import bootstrap_confidence_interval


@dataclass
class LearningCurvePoint:
    """A single point on the learning curve."""

    generation: int
    best_quality: float
    avg_quality: float
    std_quality: float
    bootstrap_ci: tuple[float, float]
    population_size: int
    metric_values: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvergenceInfo:
    """Information about evolution convergence."""

    converged: bool
    convergence_generation: int | None
    reason: str
    final_quality: float
    quality_plateau_length: int


class LearningCurveTracker:
    """Tracks learning curves across evolution generations.

    Usage:
        tracker = LearningCurveTracker()

        # After each generation
        tracker.add_generation(
            generation=0,
            quality_scores=[0.65, 0.62, 0.68],
            metric_values={"accuracy": 0.65, "f1": 0.60}
        )
        tracker.add_generation(
            generation=1,
            quality_scores=[0.72, 0.70, 0.75],
            metric_values={"accuracy": 0.72, "f1": 0.68}
        )

        # Check convergence
        if tracker.detect_convergence():
            print("Evolution has converged!")

        # Get curve data for plotting
        curve_data = tracker.get_curve_data()
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> None:
        """Initialize the tracker.

        Args:
            confidence_level: Confidence level for bootstrap CI
            n_bootstrap: Number of bootstrap samples
        """
        self._points: list[LearningCurvePoint] = []
        self._confidence_level = confidence_level
        self._n_bootstrap = n_bootstrap
        self._metric_history: dict[str, list[float]] = {}

    def add_generation(
        self,
        generation: int,
        quality_scores: list[float],
        metric_values: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LearningCurvePoint:
        """Add a generation's data to the learning curve.

        Args:
            generation: Generation number
            quality_scores: Quality scores for all individuals in population
            metric_values: Optional dict of metric name -> aggregate value
            metadata: Optional additional metadata

        Returns:
            LearningCurvePoint for this generation
        """
        if not quality_scores:
            raise ValueError("quality_scores cannot be empty")

        scores = np.array(quality_scores)
        best = float(np.max(scores))
        avg = float(np.mean(scores))
        std = float(np.std(scores)) if len(scores) > 1 else 0.0

        # Compute bootstrap CI if we have enough samples
        if len(quality_scores) >= 2:
            try:
                ci = bootstrap_confidence_interval(
                    quality_scores,
                    confidence_level=self._confidence_level,
                    n_bootstrap=self._n_bootstrap,
                )
            except ValueError:
                # Not enough samples for bootstrap
                ci = (avg, avg)
        else:
            ci = (avg, avg)

        point = LearningCurvePoint(
            generation=generation,
            best_quality=best,
            avg_quality=avg,
            std_quality=std,
            bootstrap_ci=ci,
            population_size=len(quality_scores),
            metric_values=metric_values or {},
            metadata=metadata or {},
        )

        self._points.append(point)

        # Track individual metrics
        for metric_name, value in (metric_values or {}).items():
            if metric_name not in self._metric_history:
                self._metric_history[metric_name] = []
            self._metric_history[metric_name].append(value)

        return point

    def get_curve_data(self) -> dict[str, list[Any]]:
        """Get learning curve data suitable for plotting.

        Returns:
            Dict with keys:
                - generations: List of generation numbers
                - best_quality: List of best quality scores
                - avg_quality: List of average quality scores
                - ci_lower: List of CI lower bounds
                - ci_upper: List of CI upper bounds
                - metrics: Dict of metric name -> list of values
        """
        if not self._points:
            return {
                "generations": [],
                "best_quality": [],
                "avg_quality": [],
                "ci_lower": [],
                "ci_upper": [],
                "metrics": {},
            }

        # Sort by generation
        sorted_points = sorted(self._points, key=lambda p: p.generation)

        return {
            "generations": [p.generation for p in sorted_points],
            "best_quality": [p.best_quality for p in sorted_points],
            "avg_quality": [p.avg_quality for p in sorted_points],
            "ci_lower": [p.bootstrap_ci[0] for p in sorted_points],
            "ci_upper": [p.bootstrap_ci[1] for p in sorted_points],
            "metrics": {
                metric: [p.metric_values.get(metric, 0.0) for p in sorted_points]
                for metric in self._metric_history.keys()
            },
        }

    def detect_convergence(
        self,
        patience: int = 3,
        min_improvement: float = 0.01,
    ) -> bool:
        """Detect if evolution has converged.

        Convergence is detected when the best quality score hasn't improved
        by more than min_improvement for `patience` consecutive generations.

        Args:
            patience: Number of generations without improvement to consider converged
            min_improvement: Minimum improvement threshold

        Returns:
            True if converged, False otherwise
        """
        if len(self._points) < patience + 1:
            return False

        # Sort by generation
        sorted_points = sorted(self._points, key=lambda p: p.generation)

        # Check last `patience` generations for improvement
        recent_best = [p.best_quality for p in sorted_points[-patience:]]
        prior_best = sorted_points[-(patience + 1)].best_quality

        # Check if any recent generation improved significantly
        max_recent = max(recent_best)
        improvement = max_recent - prior_best

        return improvement < min_improvement

    def get_convergence_info(
        self,
        patience: int = 3,
        min_improvement: float = 0.01,
    ) -> ConvergenceInfo:
        """Get detailed convergence information.

        Args:
            patience: Number of generations without improvement
            min_improvement: Minimum improvement threshold

        Returns:
            ConvergenceInfo with convergence details
        """
        if not self._points:
            return ConvergenceInfo(
                converged=False,
                convergence_generation=None,
                reason="No data",
                final_quality=0.0,
                quality_plateau_length=0,
            )

        sorted_points = sorted(self._points, key=lambda p: p.generation)
        final_quality = sorted_points[-1].best_quality

        # Find when quality plateaued
        plateau_start = None
        plateau_length = 0
        peak_quality = 0.0

        for i, point in enumerate(sorted_points):
            if point.best_quality > peak_quality + min_improvement:
                peak_quality = point.best_quality
                plateau_start = i
                plateau_length = 0
            else:
                plateau_length += 1

        converged = plateau_length >= patience

        if converged:
            reason = f"No improvement > {min_improvement} for {plateau_length} generations"
            convergence_gen = sorted_points[plateau_start].generation if plateau_start is not None else None
        else:
            reason = "Still improving"
            convergence_gen = None

        return ConvergenceInfo(
            converged=converged,
            convergence_generation=convergence_gen,
            reason=reason,
            final_quality=final_quality,
            quality_plateau_length=plateau_length,
        )

    def get_improvement_rate(self) -> float:
        """Calculate average improvement per generation.

        Returns:
            Average improvement in best quality per generation
        """
        if len(self._points) < 2:
            return 0.0

        sorted_points = sorted(self._points, key=lambda p: p.generation)
        first_best = sorted_points[0].best_quality
        last_best = sorted_points[-1].best_quality
        num_gens = sorted_points[-1].generation - sorted_points[0].generation

        if num_gens == 0:
            return 0.0

        return (last_best - first_best) / num_gens

    def get_total_improvement(self) -> tuple[float, float]:
        """Get total improvement from initial to final generation.

        Returns:
            Tuple of (absolute_improvement, percentage_improvement)
        """
        if len(self._points) < 2:
            return (0.0, 0.0)

        sorted_points = sorted(self._points, key=lambda p: p.generation)
        first_best = sorted_points[0].best_quality
        last_best = sorted_points[-1].best_quality

        absolute = last_best - first_best
        percentage = (absolute / first_best * 100) if first_best > 0 else 0.0

        return (absolute, percentage)

    def get_metric_improvement(self, metric_name: str) -> tuple[float, float]:
        """Get improvement for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            Tuple of (absolute_improvement, percentage_improvement)
        """
        if metric_name not in self._metric_history:
            return (0.0, 0.0)

        values = self._metric_history[metric_name]
        if len(values) < 2:
            return (0.0, 0.0)

        first_val = values[0]
        last_val = values[-1]

        absolute = last_val - first_val
        percentage = (absolute / first_val * 100) if first_val > 0 else 0.0

        return (absolute, percentage)

    def get_points(self) -> list[LearningCurvePoint]:
        """Get all learning curve points.

        Returns:
            List of LearningCurvePoint sorted by generation
        """
        return sorted(self._points, key=lambda p: p.generation)

    def get_generation_count(self) -> int:
        """Get number of generations tracked.

        Returns:
            Number of generations
        """
        return len(self._points)

    def get_best_generation(self) -> LearningCurvePoint | None:
        """Get the generation with the best quality score.

        Returns:
            LearningCurvePoint with highest best_quality, or None if empty
        """
        if not self._points:
            return None
        return max(self._points, key=lambda p: p.best_quality)

    def summary(self) -> dict[str, Any]:
        """Get summary statistics for the learning curve.

        Returns:
            Dict with summary statistics
        """
        if not self._points:
            return {
                "generations": 0,
                "initial_quality": 0.0,
                "final_quality": 0.0,
                "best_quality": 0.0,
                "best_generation": None,
                "total_improvement": 0.0,
                "improvement_rate": 0.0,
                "converged": False,
            }

        sorted_points = sorted(self._points, key=lambda p: p.generation)
        best_point = self.get_best_generation()
        abs_improvement, pct_improvement = self.get_total_improvement()
        convergence_info = self.get_convergence_info()

        return {
            "generations": len(self._points),
            "initial_quality": sorted_points[0].best_quality,
            "final_quality": sorted_points[-1].best_quality,
            "best_quality": best_point.best_quality if best_point else 0.0,
            "best_generation": best_point.generation if best_point else None,
            "total_improvement": abs_improvement,
            "improvement_pct": pct_improvement,
            "improvement_rate": self.get_improvement_rate(),
            "converged": convergence_info.converged,
            "convergence_reason": convergence_info.reason,
        }
