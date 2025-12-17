"""Quality Gate Benchmark Suite (Tier 2).

Pre-production validation with Bonferroni correction for multiple comparisons.
Provides statistical rigor for comparing evolved SOPs against baselines before deployment.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np

from siare.benchmarks.metrics.evolution_metrics import (
    extract_generated_answer,
    normalize_text,
)
from siare.benchmarks.reproducibility import ReproducibilityTracker
from siare.utils.statistics import (
    bootstrap_confidence_interval,
    wilcoxon_signed_rank_test,
)


if TYPE_CHECKING:
    from siare.benchmarks.base import BenchmarkDataset
    from siare.core.models import ProcessConfig, PromptGenome
    from siare.services.llm_provider import LLMProvider


logger = logging.getLogger(__name__)


def _empty_float_dict() -> dict[str, float]:
    """Factory for empty float dict."""
    return {}


def _empty_stats_dict() -> dict[str, "MetricStatistics"]:
    """Factory for empty stats dict."""
    return {}


def _empty_test_dict() -> dict[str, "StatisticalTest"]:
    """Factory for empty test dict."""
    return {}


def _empty_baseline_dict() -> dict[str, "BaselineComparison"]:
    """Factory for empty baseline dict."""
    return {}


@dataclass
class MetricStatistics:
    """Statistical summary of a metric across runs.

    Attributes:
        mean: Mean value of the metric
        std: Standard deviation
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "mean": self.mean,
            "std": self.std,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
        }


@dataclass
class StatisticalTest:
    """Result of a statistical hypothesis test.

    Attributes:
        p_value: Raw p-value from the test
        significant: Whether result is significant after correction
        effect_size: Cohen's d or similar effect size measure
        adjusted_alpha: Alpha threshold after Bonferroni correction
    """

    p_value: float
    significant: bool
    effect_size: float
    adjusted_alpha: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "p_value": self.p_value,
            "significant": self.significant,
            "effect_size": self.effect_size,
            "adjusted_alpha": self.adjusted_alpha,
        }


@dataclass
class BaselineComparison:
    """Comparison results between SOP and a baseline.

    Attributes:
        baseline_name: Name of the baseline system
        metrics: Statistical summaries per metric
        statistical_tests: Test results per metric
    """

    baseline_name: str
    metrics: dict[str, MetricStatistics] = field(default_factory=_empty_stats_dict)
    statistical_tests: dict[str, StatisticalTest] = field(default_factory=_empty_test_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "baseline_name": self.baseline_name,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "statistical_tests": {k: v.to_dict() for k, v in self.statistical_tests.items()},
        }


@dataclass
class QualityGateResult:
    """Complete results from quality gate benchmark.

    Attributes:
        sop_metrics: Metrics for the evolved SOP
        baseline_comparisons: Comparisons with each baseline
        dataset_hash: Hash of the benchmark dataset for reproducibility
        n_queries: Number of queries evaluated
        n_runs: Number of independent runs
        confidence_level: Confidence level used for intervals and tests
    """

    sop_metrics: dict[str, MetricStatistics] = field(default_factory=_empty_stats_dict)
    baseline_comparisons: dict[str, BaselineComparison] = field(
        default_factory=_empty_baseline_dict
    )
    dataset_hash: str = ""
    n_queries: int = 0
    n_runs: int = 0
    confidence_level: float = 0.95

    def passes_gate(self, min_improvement: float = 0.0) -> bool:
        """Check if SOP passes the quality gate.

        Passes if evolved SOP is significantly better than ALL baselines
        on at least one metric, and not significantly worse on any metric.

        Args:
            min_improvement: Minimum relative improvement required (0.0 = any improvement)

        Returns:
            True if passes the gate, False otherwise
        """
        if not self.baseline_comparisons:
            logger.warning("No baseline comparisons available")
            return False

        for baseline_name, comparison in self.baseline_comparisons.items():
            has_significant_win = False
            has_significant_loss = False

            for metric_name, test in comparison.statistical_tests.items():
                if test.significant:
                    # Check if evolved SOP is better or worse
                    sop_mean = self.sop_metrics.get(metric_name, MetricStatistics(0, 0, 0, 0)).mean
                    baseline_mean = comparison.metrics.get(
                        metric_name, MetricStatistics(0, 0, 0, 0)
                    ).mean

                    if sop_mean > baseline_mean:
                        # Check minimum improvement threshold
                        if baseline_mean > 0:
                            relative_improvement = (sop_mean - baseline_mean) / baseline_mean
                            if relative_improvement >= min_improvement:
                                has_significant_win = True
                        else:
                            has_significant_win = True
                    else:
                        has_significant_loss = True

            if has_significant_loss:
                logger.info(
                    f"Quality gate FAILED: SOP is significantly worse than {baseline_name}"
                )
                return False

            if not has_significant_win:
                logger.info(
                    f"Quality gate FAILED: No significant improvement over {baseline_name}"
                )
                return False

        return True

    def summary(self) -> dict[str, Any]:
        """Generate summary statistics.

        Returns:
            Dictionary with human-readable summary
        """
        summary_dict: dict[str, Any] = {
            "n_queries": self.n_queries,
            "n_runs": self.n_runs,
            "confidence_level": self.confidence_level,
            "dataset_hash": self.dataset_hash[:16] + "..." if self.dataset_hash else "N/A",
            "sop_metrics": {k: v.to_dict() for k, v in self.sop_metrics.items()},
            "passes_gate": self.passes_gate(),
            "baseline_comparisons": {},
        }

        for baseline_name, comparison in self.baseline_comparisons.items():
            summary_dict["baseline_comparisons"][baseline_name] = comparison.to_dict()

        return summary_dict

    def to_dict(self) -> dict[str, Any]:
        """Convert to full dictionary representation."""
        return {
            "sop_metrics": {k: v.to_dict() for k, v in self.sop_metrics.items()},
            "baseline_comparisons": {
                k: v.to_dict() for k, v in self.baseline_comparisons.items()
            },
            "dataset_hash": self.dataset_hash,
            "n_queries": self.n_queries,
            "n_runs": self.n_runs,
            "confidence_level": self.confidence_level,
        }


class QualityGateBenchmark:
    """Tier 2 Benchmark Suite with Bonferroni correction.

    Pre-production validation benchmark that compares evolved SOPs against
    baselines with statistical rigor. Uses Bonferroni correction for multiple
    comparisons to control family-wise error rate.

    Example:
        >>> benchmark = QualityGateBenchmark(
        ...     dataset=frames_dataset,
        ...     llm_provider=provider,
        ... )
        >>> result = benchmark.run(
        ...     sop=evolved_sop,
        ...     genome=evolved_genome,
        ...     baselines={"naive_rag": (naive_sop, naive_genome)},
        ...     n_runs=30,
        ... )
        >>> if result.passes_gate():
        ...     print("Ready for production!")
    """

    DEFAULT_METRICS: ClassVar[list[str]] = ["benchmark_accuracy", "benchmark_f1"]

    def __init__(
        self,
        dataset: "BenchmarkDataset",
        llm_provider: "LLMProvider",
        metrics: Optional[list[str]] = None,
        tool_adapters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize quality gate benchmark.

        Args:
            dataset: Benchmark dataset to evaluate on
            llm_provider: LLM provider for execution
            metrics: List of metric names to evaluate (default: accuracy and F1)
            tool_adapters: Optional tool adapters for SOP execution
        """
        self._dataset = dataset
        self._llm_provider = llm_provider
        self._metrics = metrics or self.DEFAULT_METRICS
        self._tool_adapters = tool_adapters or {}

    def run(
        self,
        sop: "ProcessConfig",
        genome: "PromptGenome",
        baselines: dict[str, tuple["ProcessConfig", "PromptGenome"]],
        n_runs: int = 30,
        confidence_level: float = 0.95,
        max_samples: Optional[int] = None,
        random_seed: int = 42,
    ) -> QualityGateResult:
        """Run quality gate benchmark.

        Executes the evolved SOP and all baselines multiple times,
        computing statistics and significance tests.

        Args:
            sop: Evolved SOP to evaluate
            genome: Prompt genome for the evolved SOP
            baselines: Dictionary mapping baseline names to (SOP, genome) tuples
            n_runs: Number of independent evaluation runs (default: 30)
            confidence_level: Confidence level for CIs and tests (default: 0.95)
            max_samples: Maximum samples from dataset (None = use all)
            random_seed: Random seed for reproducibility

        Returns:
            QualityGateResult with statistics and comparisons
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Get samples from dataset
        samples = list(self._dataset)
        if max_samples:
            samples = samples[:max_samples]

        n_queries = len(samples)

        # Compute dataset hash for reproducibility
        dataset_hash = ""
        dataset_filepath = getattr(self._dataset, "filepath", None)
        if dataset_filepath:
            try:
                dataset_hash = ReproducibilityTracker.compute_dataset_hash(
                    str(dataset_filepath)
                )
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Could not compute dataset hash: {e}")

        # Run evolved SOP multiple times
        logger.info(f"Running evolved SOP for {n_runs} runs on {n_queries} queries")
        sop_metric_values = self._run_multiple_times(
            sop, genome, samples, n_runs, random_seed
        )

        # Compute statistics for evolved SOP
        sop_metrics: dict[str, MetricStatistics] = {}
        for metric_name, values in sop_metric_values.items():
            sop_metrics[metric_name] = self._compute_statistics(
                values, confidence_level, random_seed
            )

        # Run baselines and compare
        baseline_comparisons: dict[str, BaselineComparison] = {}

        # Bonferroni correction: adjust alpha for number of metrics
        n_metrics = len(self._metrics)
        base_alpha = 1 - confidence_level
        adjusted_alpha = base_alpha / n_metrics

        for baseline_name, (baseline_sop, baseline_genome) in baselines.items():
            logger.info(f"Running baseline '{baseline_name}' for {n_runs} runs")

            baseline_metric_values = self._run_multiple_times(
                baseline_sop, baseline_genome, samples, n_runs, random_seed
            )

            # Compute statistics for baseline
            baseline_metrics: dict[str, MetricStatistics] = {}
            for metric_name, values in baseline_metric_values.items():
                baseline_metrics[metric_name] = self._compute_statistics(
                    values, confidence_level, random_seed
                )

            # Statistical tests comparing evolved vs baseline
            statistical_tests: dict[str, StatisticalTest] = {}
            for metric_name in self._metrics:
                if metric_name in sop_metric_values and metric_name in baseline_metric_values:
                    test_result = self._compare_metrics(
                        sop_metric_values[metric_name],
                        baseline_metric_values[metric_name],
                        adjusted_alpha,
                    )
                    statistical_tests[metric_name] = test_result

            baseline_comparisons[baseline_name] = BaselineComparison(
                baseline_name=baseline_name,
                metrics=baseline_metrics,
                statistical_tests=statistical_tests,
            )

        return QualityGateResult(
            sop_metrics=sop_metrics,
            baseline_comparisons=baseline_comparisons,
            dataset_hash=dataset_hash,
            n_queries=n_queries,
            n_runs=n_runs,
            confidence_level=confidence_level,
        )

    def _run_multiple_times(
        self,
        sop: "ProcessConfig",
        genome: "PromptGenome",
        samples: list[Any],
        n_runs: int,
        random_seed: int,
    ) -> dict[str, list[float]]:
        """Run SOP multiple times and collect metric values.

        Args:
            sop: SOP to execute
            genome: Prompt genome for the SOP
            samples: Benchmark samples to evaluate
            n_runs: Number of runs
            random_seed: Random seed for reproducibility

        Returns:
            Dictionary mapping metric names to lists of per-run values
        """
        from siare.services.execution_engine import ExecutionEngine

        metric_values: dict[str, list[float]] = {m: [] for m in self._metrics}

        for run_idx in range(n_runs):
            run_seed = random_seed + run_idx
            random.seed(run_seed)

            run_correct = 0
            run_total = 0
            run_f1_scores: list[float] = []

            for sample in samples:
                engine = ExecutionEngine(
                    llm_provider=self._llm_provider,
                    tool_adapters=self._tool_adapters,
                )

                task_data = sample.to_task_data()
                task_input = task_data.get("input", task_data)

                try:
                    trace = engine.execute(
                        sop=sop,
                        prompt_genome=genome,
                        task_input=task_input,
                    )

                    # Extract and evaluate answer
                    ground_truth = task_data.get("groundTruth", {}).get("answer", "")
                    generated = extract_generated_answer(trace)

                    if ground_truth and generated:
                        normalized_truth = normalize_text(ground_truth)
                        normalized_generated = normalize_text(generated)

                        # Accuracy (exact or partial match)
                        if (normalized_truth == normalized_generated or
                                normalized_truth in normalized_generated):
                            run_correct += 1

                        # F1 score (token overlap)
                        f1 = self._compute_f1(normalized_truth, normalized_generated)
                        run_f1_scores.append(f1)

                    run_total += 1

                except (ValueError, RuntimeError, KeyError, TimeoutError, OSError) as e:
                    logger.debug(f"Execution failed for sample: {e}")
                    run_total += 1
                    run_f1_scores.append(0.0)

            # Record metrics for this run
            if run_total > 0:
                run_accuracy = run_correct / run_total
                metric_values["benchmark_accuracy"].append(run_accuracy)

                if run_f1_scores:
                    avg_f1 = sum(run_f1_scores) / len(run_f1_scores)
                    metric_values["benchmark_f1"].append(avg_f1)

        return metric_values

    def _compute_f1(self, ground_truth: str, generated: str) -> float:
        """Compute token-level F1 score.

        Args:
            ground_truth: Normalized ground truth text
            generated: Normalized generated text

        Returns:
            F1 score between 0 and 1
        """
        truth_tokens = set(ground_truth.split())
        generated_tokens = set(generated.split())

        if not truth_tokens or not generated_tokens:
            return 0.0

        common = truth_tokens & generated_tokens
        if not common:
            return 0.0

        precision = len(common) / len(generated_tokens)
        recall = len(common) / len(truth_tokens)

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def _compute_statistics(
        self,
        values: list[float],
        confidence_level: float,
        random_seed: int,
    ) -> MetricStatistics:
        """Compute statistics for a list of metric values.

        Args:
            values: List of metric values across runs
            confidence_level: Confidence level for CI
            random_seed: Random seed for bootstrap

        Returns:
            MetricStatistics with mean, std, and CI
        """
        if not values:
            return MetricStatistics(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0)

        values_array = np.array(values)
        mean = float(np.mean(values_array))
        std = float(np.std(values_array, ddof=1)) if len(values) > 1 else 0.0

        # Bootstrap CI
        min_samples_for_ci = 2
        if len(values) >= min_samples_for_ci:
            try:
                ci_lower, ci_upper = bootstrap_confidence_interval(
                    values,
                    confidence_level=confidence_level,
                    n_bootstrap=10000,
                    random_seed=random_seed,
                )
            except ValueError:
                ci_lower = mean
                ci_upper = mean
        else:
            ci_lower = mean
            ci_upper = mean

        return MetricStatistics(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def _compare_metrics(
        self,
        sop_values: list[float],
        baseline_values: list[float],
        adjusted_alpha: float,
    ) -> StatisticalTest:
        """Compare metric values between SOP and baseline.

        Uses Wilcoxon signed-rank test for paired comparisons.

        Args:
            sop_values: Metric values from evolved SOP runs
            baseline_values: Metric values from baseline runs
            adjusted_alpha: Bonferroni-adjusted alpha threshold

        Returns:
            StatisticalTest with p-value and significance
        """
        # Ensure equal lengths for paired test
        min_len = min(len(sop_values), len(baseline_values))
        sop_paired = sop_values[:min_len]
        baseline_paired = baseline_values[:min_len]

        min_samples_for_test = 5
        if min_len < min_samples_for_test:
            # Not enough samples for statistical test
            effect_size = self._compute_cohens_d(sop_paired, baseline_paired)
            return StatisticalTest(
                p_value=1.0,
                significant=False,
                effect_size=effect_size,
                adjusted_alpha=adjusted_alpha,
            )

        try:
            test_result = wilcoxon_signed_rank_test(sop_paired, baseline_paired)
            p_value = test_result.pValue
            significant = p_value < adjusted_alpha
            effect_size = test_result.effectSize if test_result.effectSize else 0.0
        except ValueError as e:
            logger.warning(f"Wilcoxon test failed: {e}")
            effect_size = self._compute_cohens_d(sop_paired, baseline_paired)
            p_value = 1.0
            significant = False

        return StatisticalTest(
            p_value=p_value,
            significant=significant,
            effect_size=effect_size,
            adjusted_alpha=adjusted_alpha,
        )

    def _compute_cohens_d(
        self,
        group_a: list[float],
        group_b: list[float],
    ) -> float:
        """Compute Cohen's d effect size.

        Args:
            group_a: First group of values
            group_b: Second group of values

        Returns:
            Cohen's d effect size
        """
        if not group_a or not group_b:
            return 0.0

        a_array = np.array(group_a)
        b_array = np.array(group_b)

        mean_a = float(np.mean(a_array))
        mean_b = float(np.mean(b_array))

        n_a = len(group_a)
        n_b = len(group_b)

        var_a = float(np.var(a_array, ddof=1)) if n_a > 1 else 0.0
        var_b = float(np.var(b_array, ddof=1)) if n_b > 1 else 0.0

        # Pooled standard deviation
        pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2)
        pooled_std = np.sqrt(pooled_var) if pooled_var > 0 else 0.0

        if pooled_std == 0:
            return 0.0

        return (mean_a - mean_b) / pooled_std
