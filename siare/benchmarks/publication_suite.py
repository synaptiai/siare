"""Publication Benchmark Suite (Tier 3).

Publication-grade benchmark with FDR correction, ablation studies,
learning curves, and power analysis for academic rigor.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Optional

import numpy as np

from siare.benchmarks.metrics.evolution_metrics import (
    extract_generated_answer,
    normalize_text,
)
from siare.benchmarks.quality_gate_suite import MetricStatistics
from siare.benchmarks.reproducibility import (
    EnvironmentSnapshot,
    ReproducibilityTracker,
)
from siare.utils.statistics import (
    bootstrap_confidence_interval,
    wilcoxon_signed_rank_test,
)

if TYPE_CHECKING:
    from siare.benchmarks.base import BenchmarkDataset
    from siare.core.models import ProcessConfig, PromptGenome, SOPGene
    from siare.services.llm_provider import LLMProvider


logger = logging.getLogger(__name__)

# Statistical constants for power analysis fallback
STANDARD_ALPHA_THRESHOLD = 0.05  # Standard alpha for 95% confidence
STANDARD_POWER_THRESHOLD = 0.8  # Standard power (80%)
Z_ALPHA_95 = 1.96  # z-score for 95% CI (two-tailed)
Z_ALPHA_90 = 1.645  # z-score for 90% CI (two-tailed)
Z_BETA_80 = 0.84  # z-score for 80% power
Z_BETA_60 = 0.52  # z-score for 60% power


def _empty_dict_str_any() -> dict[str, Any]:
    """Factory for empty dict[str, Any]."""
    return {}


def _empty_dict_str_float() -> dict[str, float]:
    """Factory for empty dict[str, float]."""
    return {}


def _empty_dict_str_stats() -> dict[str, "PublicationMetricStats"]:
    """Factory for empty dict[str, PublicationMetricStats]."""
    return {}


def _empty_dict_str_test() -> dict[str, "PublicationStatisticalTest"]:
    """Factory for empty dict[str, PublicationStatisticalTest]."""
    return {}


def _empty_dict_str_ablation() -> dict[str, "AblationResult"]:
    """Factory for empty dict[str, AblationResult]."""
    return {}


def _empty_list_dict() -> list[dict[str, Any]]:
    """Factory for empty list of dicts."""
    return []


def _empty_baselines_dict() -> dict[str, list[dict[str, Any]]]:
    """Factory for empty baselines dict."""
    return {}


@dataclass
class PublicationMetricStats:
    """Extended statistical summary for publication-grade reporting.

    Attributes:
        mean: Mean value of the metric
        std: Standard deviation
        ci_lower: Lower bound of confidence interval
        ci_upper: Upper bound of confidence interval
        ci_width: Width of confidence interval (ci_upper - ci_lower)
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    ci_width: float = 0.0

    def __post_init__(self) -> None:
        """Compute CI width if not provided."""
        if self.ci_width == 0.0:
            self.ci_width = self.ci_upper - self.ci_lower

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary representation."""
        return {
            "mean": self.mean,
            "std": self.std,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_width": self.ci_width,
        }

    @classmethod
    def from_metric_statistics(cls, stats: MetricStatistics) -> "PublicationMetricStats":
        """Create from MetricStatistics."""
        return cls(
            mean=stats.mean,
            std=stats.std,
            ci_lower=stats.ci_lower,
            ci_upper=stats.ci_upper,
            ci_width=stats.ci_upper - stats.ci_lower,
        )


@dataclass
class PublicationStatisticalTest:
    """Extended statistical test result for publication-grade reporting.

    Attributes:
        p_value: Raw p-value from the test
        adjusted_p_value: P-value after Benjamini-Hochberg FDR correction
        significant: Whether result is significant after FDR correction
        effect_size: Cohen's d or similar effect size measure
        mean_difference: Absolute difference in means (SOP - baseline)
        relative_improvement_pct: Relative improvement as percentage
    """

    p_value: float
    adjusted_p_value: float
    significant: bool
    effect_size: float
    mean_difference: float
    relative_improvement_pct: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "p_value": self.p_value,
            "adjusted_p_value": self.adjusted_p_value,
            "significant": self.significant,
            "effect_size": self.effect_size,
            "mean_difference": self.mean_difference,
            "relative_improvement_pct": self.relative_improvement_pct,
        }


@dataclass
class AblationResult:
    """Results from ablating (removing) a component from the SOP.

    Attributes:
        component_name: Name of the ablated component
        metrics: Statistical summaries per metric when component is removed
        statistical_tests: Tests comparing full SOP vs ablated SOP
        contribution: Performance drop per metric (full - ablated)
    """

    component_name: str
    metrics: dict[str, PublicationMetricStats] = field(default_factory=_empty_dict_str_stats)
    statistical_tests: dict[str, PublicationStatisticalTest] = field(
        default_factory=_empty_dict_str_test
    )
    contribution: dict[str, float] = field(default_factory=_empty_dict_str_float)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "component_name": self.component_name,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "statistical_tests": {k: v.to_dict() for k, v in self.statistical_tests.items()},
            "contribution": self.contribution,
        }


@dataclass
class LearningCurveData:
    """Data for plotting learning curves across generations.

    Attributes:
        evolved_sop: List of generation data points for evolved SOP
        baselines: Dict mapping baseline names to their (static) generation data
        primary_metric: The primary metric being tracked
    """

    evolved_sop: list[dict[str, Any]] = field(default_factory=_empty_list_dict)
    baselines: dict[str, list[dict[str, Any]]] = field(default_factory=_empty_baselines_dict)
    primary_metric: str = "benchmark_accuracy"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "evolved_sop": self.evolved_sop,
            "baselines": self.baselines,
            "primary_metric": self.primary_metric,
        }


@dataclass
class PowerAnalysisResult:
    """Results from statistical power analysis.

    Attributes:
        primary_metric: The metric used for power calculation
        effect_size: Observed effect size (Cohen's d)
        alpha: Significance level used
        power: Desired statistical power
        required_sample_size: Sample size needed to achieve desired power
        actual_sample_size: Actual sample size in the study
        sufficient: Whether actual sample size is sufficient
    """

    primary_metric: str
    effect_size: float
    alpha: float
    power: float
    required_sample_size: int
    actual_sample_size: int
    sufficient: bool

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "primary_metric": self.primary_metric,
            "effect_size": self.effect_size,
            "alpha": self.alpha,
            "power": self.power,
            "required_sample_size": self.required_sample_size,
            "actual_sample_size": self.actual_sample_size,
            "sufficient": self.sufficient,
        }


@dataclass
class PublicationBenchmarkResult:
    """Complete results from publication-grade benchmark.

    Attributes:
        metadata: Environment and run metadata for reproducibility
        evolved_sop_results: Metrics and statistics for evolved SOP
        baseline_comparisons: Comparisons with each baseline
        ablation_studies: Results from ablation experiments
        learning_curves: Data for plotting learning curves
        power_analysis: Statistical power analysis results
    """

    metadata: dict[str, Any] = field(default_factory=_empty_dict_str_any)
    evolved_sop_results: dict[str, Any] = field(default_factory=_empty_dict_str_any)
    baseline_comparisons: dict[str, dict[str, Any]] = field(default_factory=_empty_dict_str_any)
    ablation_studies: dict[str, AblationResult] = field(default_factory=_empty_dict_str_ablation)
    learning_curves: LearningCurveData | None = None
    power_analysis: PowerAnalysisResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to full dictionary representation."""
        return {
            "metadata": self.metadata,
            "evolved_sop_results": self.evolved_sop_results,
            "baseline_comparisons": self.baseline_comparisons,
            "ablation_studies": {k: v.to_dict() for k, v in self.ablation_studies.items()},
            "learning_curves": self.learning_curves.to_dict() if self.learning_curves else None,
            "power_analysis": self.power_analysis.to_dict() if self.power_analysis else None,
        }

    def summary(self) -> dict[str, Any]:
        """Generate human-readable summary."""
        summary_dict: dict[str, Any] = {
            "timestamp": self.metadata.get("timestamp", "unknown"),
            "n_queries": self.metadata.get("n_queries", 0),
            "n_runs": self.metadata.get("n_runs", 0),
            "confidence_level": self.metadata.get("confidence_level", 0.99),
        }

        # Summarize evolved SOP results
        if self.evolved_sop_results.get("metrics"):
            summary_dict["evolved_sop_metrics"] = self.evolved_sop_results["metrics"]

        # Summarize comparisons
        summary_dict["n_baselines"] = len(self.baseline_comparisons)
        summary_dict["n_ablations"] = len(self.ablation_studies)

        # Power analysis summary
        if self.power_analysis:
            summary_dict["power_sufficient"] = self.power_analysis.sufficient
            summary_dict["required_sample_size"] = self.power_analysis.required_sample_size

        return summary_dict


class PublicationBenchmark:
    """Tier 3 Publication-grade Benchmark Suite.

    Provides publication-ready statistical analysis with:
    - Benjamini-Hochberg FDR correction for multiple comparisons
    - Ablation studies to measure component contributions
    - Learning curves tracking improvement over generations
    - Power analysis for sample size adequacy

    Example:
        >>> benchmark = PublicationBenchmark(
        ...     golden_dataset=frames_dataset,
        ...     llm_provider=provider,
        ... )
        >>> result = benchmark.run_full_suite(
        ...     evolved_sop=best_sop,
        ...     evolved_genome=best_genome,
        ...     evolution_history=history,
        ...     baselines={"naive_rag": (naive_sop, naive_genome)},
        ...     ablations={"no_decomposition": (ablated_sop, ablated_genome)},
        ...     n_runs=30,
        ... )
        >>> print(result.summary())
    """

    FULL_METRICS: ClassVar[list[str]] = [
        "benchmark_accuracy",
        "benchmark_f1",
        "benchmark_partial_match",
    ]

    def __init__(
        self,
        golden_dataset: "BenchmarkDataset",
        llm_provider: "LLMProvider",
        public_dataset: Optional["BenchmarkDataset"] = None,
        metrics: list[str] | None = None,
        tool_adapters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize publication benchmark.

        Args:
            golden_dataset: Primary benchmark dataset for evaluation
            llm_provider: LLM provider for execution
            public_dataset: Optional additional dataset for generalization testing
            metrics: List of metric names to evaluate (default: full metrics)
            tool_adapters: Optional tool adapters for SOP execution
        """
        self._golden_dataset = golden_dataset
        self._public_dataset = public_dataset
        self._llm_provider = llm_provider
        self._metrics = metrics or self.FULL_METRICS
        self._tool_adapters = tool_adapters or {}

    def run_full_suite(
        self,
        evolved_sop: "ProcessConfig",
        evolved_genome: "PromptGenome",
        evolution_history: list["SOPGene"],
        baselines: dict[str, tuple["ProcessConfig", "PromptGenome"]],
        ablations: dict[str, tuple["ProcessConfig", "PromptGenome"]],
        n_runs: int = 30,
        confidence_level: float = 0.99,
        max_samples: int | None = None,
        random_seed: int = 42,
    ) -> PublicationBenchmarkResult:
        """Run complete publication-grade benchmark suite.

        Args:
            evolved_sop: The evolved SOP to evaluate
            evolved_genome: Prompt genome for the evolved SOP
            evolution_history: List of SOPGene objects from evolution
            baselines: Dictionary mapping baseline names to (SOP, genome) tuples
            ablations: Dictionary mapping ablation names to (SOP, genome) tuples
            n_runs: Number of independent evaluation runs (default: 30)
            confidence_level: Confidence level for CIs and tests (default: 0.99)
            max_samples: Maximum samples from dataset (None = use all)
            random_seed: Random seed for reproducibility

        Returns:
            PublicationBenchmarkResult with full analysis
        """
        random.seed(random_seed)
        np.random.seed(random_seed)

        # Capture environment for reproducibility
        env_snapshot = ReproducibilityTracker.capture_environment()

        # Get samples from dataset
        samples = list(self._golden_dataset)
        if max_samples:
            samples = samples[:max_samples]

        n_queries = len(samples)

        # Compute dataset hash
        dataset_hash = ""
        dataset_filepath = getattr(self._golden_dataset, "filepath", None)
        if dataset_filepath:
            try:
                dataset_hash = ReproducibilityTracker.compute_dataset_hash(
                    str(dataset_filepath)
                )
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Could not compute dataset hash: {e}")

        # Build metadata
        metadata: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_queries": n_queries,
            "n_runs": n_runs,
            "confidence_level": confidence_level,
            "random_seed": random_seed,
            "dataset_hash": dataset_hash,
            "environment": self._env_snapshot_to_dict(env_snapshot),
            "metrics": self._metrics,
        }

        # Run evolved SOP multiple times
        logger.info(f"Running evolved SOP for {n_runs} runs on {n_queries} queries")
        evolved_metric_values = self._run_multiple_times(
            evolved_sop, evolved_genome, samples, n_runs, random_seed
        )

        # Compute statistics for evolved SOP
        evolved_metrics_stats: dict[str, PublicationMetricStats] = {}
        for metric_name, values in evolved_metric_values.items():
            stats = self._compute_statistics(values, confidence_level, random_seed)
            evolved_metrics_stats[metric_name] = stats

        evolved_sop_results: dict[str, Any] = {
            "metrics": {k: v.to_dict() for k, v in evolved_metrics_stats.items()},
        }

        # Run baselines and collect p-values for FDR correction
        baseline_raw_results: dict[str, dict[str, Any]] = {}
        all_p_values: list[float] = []
        p_value_map: list[tuple[str, str]] = []  # (baseline_name, metric_name)

        for baseline_name, (baseline_sop, baseline_genome) in baselines.items():
            logger.info(f"Running baseline '{baseline_name}' for {n_runs} runs")

            baseline_metric_values = self._run_multiple_times(
                baseline_sop, baseline_genome, samples, n_runs, random_seed
            )

            # Compute statistics
            baseline_metrics_stats: dict[str, PublicationMetricStats] = {}
            for metric_name, values in baseline_metric_values.items():
                stats = self._compute_statistics(values, confidence_level, random_seed)
                baseline_metrics_stats[metric_name] = stats

            # Compute raw p-values and effect sizes
            raw_tests: dict[str, dict[str, float]] = {}
            for metric_name in self._metrics:
                if (
                    metric_name in evolved_metric_values
                    and metric_name in baseline_metric_values
                ):
                    sop_vals = evolved_metric_values[metric_name]
                    baseline_vals = baseline_metric_values[metric_name]

                    p_value, effect_size = self._compute_test(sop_vals, baseline_vals)
                    mean_diff = evolved_metrics_stats[metric_name].mean - baseline_metrics_stats.get(
                        metric_name, PublicationMetricStats(0, 0, 0, 0)
                    ).mean

                    baseline_mean = baseline_metrics_stats.get(
                        metric_name, PublicationMetricStats(0, 0, 0, 0)
                    ).mean
                    rel_improvement = (
                        (mean_diff / baseline_mean * 100) if baseline_mean > 0 else 0.0
                    )

                    raw_tests[metric_name] = {
                        "p_value": p_value,
                        "effect_size": effect_size,
                        "mean_difference": mean_diff,
                        "relative_improvement_pct": rel_improvement,
                    }

                    all_p_values.append(p_value)
                    p_value_map.append((baseline_name, metric_name))

            baseline_raw_results[baseline_name] = {
                "metrics": baseline_metrics_stats,
                "raw_tests": raw_tests,
            }

        # Apply Benjamini-Hochberg FDR correction
        adjusted_p_values = self._fdr_correction(all_p_values)

        # Build final baseline comparisons with adjusted p-values
        baseline_comparisons: dict[str, dict[str, Any]] = {}
        alpha = 1 - confidence_level

        for baseline_name, raw_data in baseline_raw_results.items():
            tests: dict[str, dict[str, Any]] = {}

            for metric_name, raw_test in raw_data["raw_tests"].items():
                # Find the adjusted p-value for this test
                idx = None
                for i, (bn, mn) in enumerate(p_value_map):
                    if bn == baseline_name and mn == metric_name:
                        idx = i
                        break

                adj_p = adjusted_p_values[idx] if idx is not None else raw_test["p_value"]
                significant = adj_p < alpha

                tests[metric_name] = PublicationStatisticalTest(
                    p_value=raw_test["p_value"],
                    adjusted_p_value=adj_p,
                    significant=significant,
                    effect_size=raw_test["effect_size"],
                    mean_difference=raw_test["mean_difference"],
                    relative_improvement_pct=raw_test["relative_improvement_pct"],
                ).to_dict()

            baseline_comparisons[baseline_name] = {
                "metrics": {k: v.to_dict() for k, v in raw_data["metrics"].items()},
                "statistical_tests": tests,
            }

        # Run ablation studies
        ablation_studies: dict[str, AblationResult] = {}

        if ablations:
            logger.info(f"Running {len(ablations)} ablation studies")

            # Collect p-values for ablation FDR
            ablation_p_values: list[float] = []
            ablation_p_map: list[tuple[str, str]] = []

            ablation_raw: dict[str, dict[str, Any]] = {}

            for ablation_name, (ablation_sop, ablation_genome) in ablations.items():
                logger.info(f"Running ablation '{ablation_name}'")

                ablation_metric_values = self._run_multiple_times(
                    ablation_sop, ablation_genome, samples, n_runs, random_seed
                )

                # Compute statistics
                ablation_metrics_stats: dict[str, PublicationMetricStats] = {}
                for metric_name, values in ablation_metric_values.items():
                    stats = self._compute_statistics(values, confidence_level, random_seed)
                    ablation_metrics_stats[metric_name] = stats

                # Compute contributions (full - ablated) and raw tests
                contributions: dict[str, float] = {}
                raw_abl_tests: dict[str, dict[str, float]] = {}

                for metric_name in self._metrics:
                    if (
                        metric_name in evolved_metric_values
                        and metric_name in ablation_metric_values
                    ):
                        evolved_mean = evolved_metrics_stats[metric_name].mean
                        ablated_mean = ablation_metrics_stats[metric_name].mean
                        contribution = evolved_mean - ablated_mean
                        contributions[metric_name] = contribution

                        # Statistical test: evolved vs ablated
                        p_value, effect_size = self._compute_test(
                            evolved_metric_values[metric_name],
                            ablation_metric_values[metric_name],
                        )

                        rel_improvement = (
                            (contribution / ablated_mean * 100) if ablated_mean > 0 else 0.0
                        )

                        raw_abl_tests[metric_name] = {
                            "p_value": p_value,
                            "effect_size": effect_size,
                            "mean_difference": contribution,
                            "relative_improvement_pct": rel_improvement,
                        }

                        ablation_p_values.append(p_value)
                        ablation_p_map.append((ablation_name, metric_name))

                ablation_raw[ablation_name] = {
                    "metrics": ablation_metrics_stats,
                    "contributions": contributions,
                    "raw_tests": raw_abl_tests,
                }

            # Apply FDR correction to ablation p-values
            adjusted_ablation_p = self._fdr_correction(ablation_p_values)

            # Build final ablation results
            for ablation_name, raw_abl_data in ablation_raw.items():
                abl_tests: dict[str, PublicationStatisticalTest] = {}

                for metric_name, raw_test in raw_abl_data["raw_tests"].items():
                    idx = None
                    for i, (an, mn) in enumerate(ablation_p_map):
                        if an == ablation_name and mn == metric_name:
                            idx = i
                            break

                    adj_p = adjusted_ablation_p[idx] if idx is not None else raw_test["p_value"]
                    significant = adj_p < alpha

                    abl_tests[metric_name] = PublicationStatisticalTest(
                        p_value=raw_test["p_value"],
                        adjusted_p_value=adj_p,
                        significant=significant,
                        effect_size=raw_test["effect_size"],
                        mean_difference=raw_test["mean_difference"],
                        relative_improvement_pct=raw_test["relative_improvement_pct"],
                    )

                ablation_studies[ablation_name] = AblationResult(
                    component_name=ablation_name,
                    metrics=raw_abl_data["metrics"],
                    statistical_tests=abl_tests,
                    contribution=raw_abl_data["contributions"],
                )

        # Compute learning curves from evolution history
        learning_curves = self._compute_learning_curves(
            evolution_history, baselines, samples, n_runs, random_seed
        )

        # Power analysis
        power_analysis = self._compute_power_analysis(
            evolved_metric_values,
            baseline_raw_results,
            n_queries,
            alpha,
        )

        return PublicationBenchmarkResult(
            metadata=metadata,
            evolved_sop_results=evolved_sop_results,
            baseline_comparisons=baseline_comparisons,
            ablation_studies=ablation_studies,
            learning_curves=learning_curves,
            power_analysis=power_analysis,
        )

    def _fdr_correction(self, p_values: list[float]) -> list[float]:
        """Apply Benjamini-Hochberg FDR correction.

        Args:
            p_values: List of raw p-values

        Returns:
            List of adjusted p-values (same order as input)
        """
        if not p_values:
            return []

        n = len(p_values)
        p_array = np.array(p_values)

        # Get sorted indices
        sorted_indices = np.argsort(p_array)
        sorted_p = p_array[sorted_indices]

        # Compute adjusted p-values
        adjusted = np.zeros(n)
        for i in range(n):
            rank = i + 1
            adjusted[sorted_indices[i]] = sorted_p[i] * n / rank

        # Enforce monotonicity (right-to-left)
        for i in range(n - 1, 0, -1):
            adjusted[i - 1] = min(adjusted[i - 1], adjusted[i])

        # Cap at 1.0
        adjusted = np.minimum(adjusted, 1.0)

        return adjusted.tolist()

    def _compute_power_analysis(
        self,
        evolved_metric_values: dict[str, list[float]],
        baseline_raw_results: dict[str, dict[str, Any]],
        n_queries: int,
        alpha: float,
        target_power: float = 0.8,
    ) -> PowerAnalysisResult:
        """Compute statistical power analysis.

        Args:
            evolved_metric_values: Metric values from evolved SOP
            baseline_raw_results: Raw baseline comparison results
            n_queries: Number of queries evaluated
            alpha: Significance level
            target_power: Desired statistical power (default: 0.8)

        Returns:
            PowerAnalysisResult with sample size adequacy assessment
        """
        primary_metric = "benchmark_accuracy"

        # Find the average effect size across baselines for primary metric
        effect_sizes: list[float] = []

        for baseline_data in baseline_raw_results.values():
            if primary_metric in baseline_data["raw_tests"]:
                effect_sizes.append(baseline_data["raw_tests"][primary_metric]["effect_size"])

        avg_effect_size: float
        if effect_sizes:
            avg_effect_size = sum(abs(es) for es in effect_sizes) / len(effect_sizes)
        # Estimate effect size from evolved SOP variance
        elif primary_metric in evolved_metric_values:
            values = evolved_metric_values[primary_metric]
            if values:
                std_val = np.std(values, ddof=1) if len(values) > 1 else 0.1
                avg_effect_size = float(0.5 * std_val)  # Medium effect size estimate
            else:
                avg_effect_size = 0.5
        else:
            avg_effect_size = 0.5

        # Compute required sample size
        required_n = self._compute_required_sample_size(
            effect_size=float(avg_effect_size),
            alpha=alpha,
            power=target_power,
        )

        return PowerAnalysisResult(
            primary_metric=primary_metric,
            effect_size=float(avg_effect_size),
            alpha=alpha,
            power=target_power,
            required_sample_size=required_n,
            actual_sample_size=n_queries,
            sufficient=n_queries >= required_n,
        )

    def _compute_required_sample_size(
        self,
        effect_size: float,
        alpha: float,
        power: float,
    ) -> int:
        """Compute required sample size for desired power.

        Uses the formula: n = 2 * ((z_alpha + z_beta) / d)^2
        where d is effect size, z_alpha and z_beta are normal quantiles.

        Args:
            effect_size: Expected effect size (Cohen's d)
            alpha: Significance level
            power: Desired statistical power

        Returns:
            Required sample size per group
        """
        # Protect against invalid inputs
        if effect_size <= 0:
            return 1000  # Default large sample size

        try:
            from scipy.stats import norm

            z_alpha = float(norm.ppf(1 - alpha / 2))  # type: ignore[reportUnknownMemberType]
            z_beta = float(norm.ppf(power))  # type: ignore[reportUnknownMemberType]
            n = int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2 * 2))
            return max(n, 10)  # Minimum 10 samples
        except ImportError:
            # Fallback approximation if scipy not available
            # Using standard z-scores from constants
            z_alpha = Z_ALPHA_95 if alpha <= STANDARD_ALPHA_THRESHOLD else Z_ALPHA_90
            z_beta = Z_BETA_80 if power >= STANDARD_POWER_THRESHOLD else Z_BETA_60
            n = int(np.ceil(((z_alpha + z_beta) / effect_size) ** 2 * 2))
            return max(n, 10)

    def _compute_learning_curves(
        self,
        evolution_history: list["SOPGene"],
        baselines: dict[str, tuple["ProcessConfig", "PromptGenome"]],
        samples: list[Any],
        n_runs: int,
        random_seed: int,
    ) -> LearningCurveData:
        """Extract learning curve data from evolution history.

        Args:
            evolution_history: List of SOPGene objects from evolution
            baselines: Baseline configurations for comparison
            samples: Benchmark samples
            n_runs: Number of runs
            random_seed: Random seed

        Returns:
            LearningCurveData with generation-by-generation metrics
        """
        primary_metric = "benchmark_accuracy"

        evolved_sop_data: list[dict[str, Any]] = []

        # Extract metrics from evolution history
        for gene in evolution_history:
            generation = gene.generation if gene.generation is not None else 0
            metric_mean = gene.get_metric_mean(primary_metric)

            evolved_sop_data.append({
                "generation": generation,
                primary_metric: metric_mean,
            })

        # For baselines, they don't evolve, so show flat line at their performance
        baselines_data: dict[str, list[dict[str, Any]]] = {}

        if evolution_history:
            max_gen = max(
                (g.generation for g in evolution_history if g.generation is not None),
                default=0,
            )
        else:
            max_gen = 10

        for baseline_name, (baseline_sop, baseline_genome) in baselines.items():
            # Run baseline once to get its performance
            baseline_values = self._run_multiple_times(
                baseline_sop, baseline_genome, samples, n_runs, random_seed
            )

            if primary_metric in baseline_values:
                baseline_mean = sum(baseline_values[primary_metric]) / len(
                    baseline_values[primary_metric]
                )
            else:
                baseline_mean = 0.0

            # Create flat line data for all generations
            baseline_curve: list[dict[str, Any]] = []
            for gen in range(max_gen + 1):
                baseline_curve.append({
                    "generation": gen,
                    primary_metric: baseline_mean,
                })
            baselines_data[baseline_name] = baseline_curve

        return LearningCurveData(
            evolved_sop=evolved_sop_data,
            baselines=baselines_data,
            primary_metric=primary_metric,
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
            run_partial_correct = 0
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

                        # Accuracy (exact match)
                        if normalized_truth == normalized_generated:
                            run_correct += 1
                            run_partial_correct += 1
                        # Partial match (containment)
                        elif (
                            normalized_truth in normalized_generated
                            or normalized_generated in normalized_truth
                        ):
                            run_partial_correct += 1

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
                if "benchmark_accuracy" in metric_values:
                    run_accuracy = run_correct / run_total
                    metric_values["benchmark_accuracy"].append(run_accuracy)

                if "benchmark_f1" in metric_values and run_f1_scores:
                    avg_f1 = sum(run_f1_scores) / len(run_f1_scores)
                    metric_values["benchmark_f1"].append(avg_f1)

                if "benchmark_partial_match" in metric_values:
                    run_partial = run_partial_correct / run_total
                    metric_values["benchmark_partial_match"].append(run_partial)

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
    ) -> PublicationMetricStats:
        """Compute extended statistics for publication.

        Args:
            values: List of metric values across runs
            confidence_level: Confidence level for CI
            random_seed: Random seed for bootstrap

        Returns:
            PublicationMetricStats with full statistical summary
        """
        if not values:
            return PublicationMetricStats(mean=0.0, std=0.0, ci_lower=0.0, ci_upper=0.0)

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

        return PublicationMetricStats(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            ci_width=ci_upper - ci_lower,
        )

    def _compute_test(
        self,
        sop_values: list[float],
        baseline_values: list[float],
    ) -> tuple[float, float]:
        """Compute statistical test between two groups.

        Args:
            sop_values: Metric values from evolved SOP
            baseline_values: Metric values from baseline

        Returns:
            Tuple of (p_value, effect_size)
        """
        min_len = min(len(sop_values), len(baseline_values))
        sop_paired = sop_values[:min_len]
        baseline_paired = baseline_values[:min_len]

        min_samples_for_test = 5
        if min_len < min_samples_for_test:
            effect_size = self._compute_cohens_d(sop_paired, baseline_paired)
            return 1.0, effect_size

        try:
            test_result = wilcoxon_signed_rank_test(sop_paired, baseline_paired)
            p_value = test_result.pValue
            effect_size = test_result.effectSize if test_result.effectSize else 0.0
        except ValueError:
            effect_size = self._compute_cohens_d(sop_paired, baseline_paired)
            p_value = 1.0

        return p_value, effect_size

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

    def _env_snapshot_to_dict(self, snapshot: EnvironmentSnapshot) -> dict[str, Any]:
        """Convert EnvironmentSnapshot to dictionary.

        Args:
            snapshot: Environment snapshot

        Returns:
            Dictionary representation
        """
        return {
            "timestamp": snapshot.timestamp,
            "git_commit": snapshot.git_commit,
            "git_branch": snapshot.git_branch,
            "git_dirty": snapshot.git_dirty,
            "python_version": snapshot.python_version,
            "dependencies": snapshot.dependencies,
            "hardware": snapshot.hardware,
        }

    def generate_report(
        self,
        result: PublicationBenchmarkResult,
        output_path: str,
    ) -> str:
        """Generate publication report and save to file.

        Args:
            result: PublicationBenchmarkResult to report on
            output_path: Path to save report

        Returns:
            Path to saved report
        """
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        # Save full results as JSON
        json_path = output.with_suffix(".json")
        with json_path.open("w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        logger.info(f"Saved publication results to {json_path}")
        return str(json_path)
