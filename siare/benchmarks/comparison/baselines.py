"""Baseline strategies for fair comparison with evolution.

To prove evolution provides value, we must compare against:
1. No retrieval (0% expected) - proves retrieval is necessary
2. Random search (same compute budget) - proves we're not just lucky sampling
3. Static hand-tuned configs - proves evolution beats human intuition
4. Oracle (if known) - shows ceiling performance

This module provides all baseline strategies for rigorous comparison.
"""

import random
from dataclasses import dataclass
from typing import Any

from siare.benchmarks.sops.evolvable_rag import (
    EVOLVABLE_PARAM_BOUNDS,
    create_evolvable_rag_genome,
    create_evolvable_rag_sop,
)
from siare.benchmarks.sops.simple_qa import (
    create_benchmark_genome,
    create_benchmark_sop,
)
from siare.core.models import ProcessConfig, PromptGenome

# Static baseline configurations
STATIC_BASELINE_CONFIGS: dict[str, dict[str, Any]] = {
    "poor": {
        "top_k": 50,
        "similarity_threshold": 0.3,
        "description": "Intentionally poor config (too many docs, low threshold)",
    },
    "reasonable": {
        "top_k": 10,
        "similarity_threshold": 0.6,
        "description": "Hand-tuned reasonable config",
    },
    "aggressive": {
        "top_k": 5,
        "similarity_threshold": 0.8,
        "description": "Aggressive filtering (may miss relevant docs)",
    },
    "oracle": {
        "top_k": 8,
        "similarity_threshold": 0.7,
        "description": "Known-good config (ceiling performance)",
    },
}


def create_no_retrieval_baseline(
    model: str = "llama3.1:8b",
) -> tuple[ProcessConfig, PromptGenome]:
    """Create baseline with no retrieval capability.

    This proves that retrieval is necessary - LLM parametric
    knowledge alone cannot answer knowledge-intensive questions.

    Args:
        model: Model identifier

    Returns:
        Tuple of (SOP, genome) with no tools
    """
    sop = create_benchmark_sop(model=model)
    genome = create_benchmark_genome()
    return sop, genome


def create_static_baseline(
    config_name: str,
    model: str = "llama3.1:8b",
    index_name: str = "frames_corpus",
) -> tuple[ProcessConfig, PromptGenome]:
    """Create static baseline from predefined config.

    Args:
        config_name: One of STATIC_BASELINE_CONFIGS keys
        model: Model identifier
        index_name: Vector index name

    Returns:
        Tuple of (SOP, genome) with static config
    """
    if config_name not in STATIC_BASELINE_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(STATIC_BASELINE_CONFIGS.keys())}")

    config = STATIC_BASELINE_CONFIGS[config_name]

    sop = create_evolvable_rag_sop(
        model=model,
        top_k=config["top_k"],
        similarity_threshold=config["similarity_threshold"],
        index_name=index_name,
    )
    genome = create_evolvable_rag_genome(retrieval_style="generic")

    return sop, genome


class RandomSearchBaseline:
    """Random search baseline for fair comparison.

    Generates N random configurations within the same parameter
    bounds that evolution uses. The best random config should be
    compared against the evolved config to prove evolution does
    better than random sampling.

    Example:
        >>> baseline = RandomSearchBaseline(n_samples=20, seed=42)
        >>> variants = baseline.generate_variants()
        >>> # Run benchmark on all variants, take best
    """

    def __init__(
        self,
        n_samples: int = 20,
        seed: int | None = None,
        param_bounds: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize random search baseline.

        Args:
            n_samples: Number of random configurations to generate
            seed: Random seed for reproducibility
            param_bounds: Parameter bounds (default: EVOLVABLE_PARAM_BOUNDS)
        """
        self.n_samples = n_samples
        self.seed = seed
        self.param_bounds = param_bounds or EVOLVABLE_PARAM_BOUNDS

        if seed is not None:
            random.seed(seed)

    def generate_variants(self) -> list[dict[str, Any]]:
        """Generate random parameter configurations.

        Returns:
            List of configuration dictionaries
        """
        variants = []

        for _ in range(self.n_samples):
            config = {}

            for param_name, bounds in self.param_bounds.items():
                if bounds["type"] == "int":
                    config[param_name] = random.randint(bounds["min"], bounds["max"])
                elif bounds["type"] == "float":
                    config[param_name] = round(
                        random.uniform(bounds["min"], bounds["max"]), 2
                    )

            variants.append(config)

        return variants

    def create_sops(
        self,
        model: str = "llama3.1:8b",
        index_name: str = "frames_corpus",
    ) -> list[tuple[ProcessConfig, PromptGenome, dict[str, Any]]]:
        """Create SOPs for all random variants.

        Args:
            model: Model identifier
            index_name: Vector index name

        Returns:
            List of (SOP, genome, config) tuples
        """
        variants = self.generate_variants()
        sops = []

        for config in variants:
            sop = create_evolvable_rag_sop(
                model=model,
                top_k=config.get("top_k", 10),
                similarity_threshold=config.get("similarity_threshold", 0.5),
                index_name=index_name,
            )
            genome = create_evolvable_rag_genome(retrieval_style="generic")
            sops.append((sop, genome, config))

        return sops


@dataclass
class BaselineResult:
    """Result from a baseline run."""

    name: str
    config: dict[str, Any]
    metrics: dict[str, float]
    runtime_seconds: float


class BaselineComparison:
    """Generates comparison reports across baselines.

    Compares evolved SOP against all baselines and generates
    a report showing improvement over each baseline.

    Example:
        >>> results = {"evolved": {...}, "random_best": {...}, ...}
        >>> comparison = BaselineComparison(results)
        >>> print(comparison.generate_report())
    """

    def __init__(
        self,
        results: dict[str, dict[str, float]],
        evolved_key: str = "evolved",
    ) -> None:
        """Initialize comparison.

        Args:
            results: Dictionary mapping baseline name to metrics
            evolved_key: Key for evolved results
        """
        self.results = results
        self.evolved_key = evolved_key

    def get_evolved_improvement(
        self,
        baseline_name: str,
        metric: str = "accuracy",
    ) -> tuple[float, float]:
        """Get improvement of evolved over baseline.

        Args:
            baseline_name: Baseline to compare against
            metric: Metric to compare

        Returns:
            Tuple of (absolute_improvement, relative_improvement_percent)
        """
        if baseline_name not in self.results:
            return 0.0, 0.0

        evolved = self.results[self.evolved_key].get(metric, 0)
        baseline = self.results[baseline_name].get(metric, 0)

        absolute = evolved - baseline
        relative = (absolute / baseline * 100) if baseline > 0 else 0

        return absolute, relative

    def generate_report(self) -> str:
        """Generate comparison report.

        Returns:
            Markdown-formatted comparison report
        """
        lines = [
            "# Baseline Comparison Report",
            "",
            "## Summary",
            "",
            "| Baseline | Accuracy | vs Evolved |",
            "|----------|----------|------------|",
        ]

        evolved_acc = self.results.get(self.evolved_key, {}).get("accuracy", 0)

        for name, metrics in sorted(self.results.items()):
            if name == self.evolved_key:
                continue

            acc = metrics.get("accuracy", 0)
            abs_imp, rel_imp = self.get_evolved_improvement(name)

            sign = "+" if abs_imp >= 0 else ""
            lines.append(f"| {name} | {acc:.1%} | {sign}{abs_imp:.1%} ({sign}{rel_imp:.1f}%) |")

        lines.append(f"| **{self.evolved_key}** | **{evolved_acc:.1%}** | - |")
        lines.append("")

        # Key findings
        lines.extend([
            "## Key Findings",
            "",
        ])

        # vs no retrieval
        if "no_retrieval" in self.results:
            abs_imp, _ = self.get_evolved_improvement("no_retrieval")
            lines.append(f"- **Retrieval necessity**: Evolution achieves {abs_imp:.1%} improvement over no-retrieval baseline")

        # vs random search
        if "random_search_best" in self.results:
            abs_imp, rel_imp = self.get_evolved_improvement("random_search_best")
            lines.append(f"- **vs Random Search**: Evolution beats best random config by {abs_imp:.1%} ({rel_imp:.1f}% relative)")

        # vs static configs
        for static_name in ["static_poor", "static_reasonable"]:
            if static_name in self.results:
                abs_imp, rel_imp = self.get_evolved_improvement(static_name)
                lines.append(f"- **vs {static_name}**: +{abs_imp:.1%} ({rel_imp:.1f}% relative improvement)")

        return "\n".join(lines)
