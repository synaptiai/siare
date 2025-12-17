"""Self-improvement benchmark for demonstrating SIARE's core value.

This benchmark demonstrates that SIARE can improve RAG performance through
SOP evolution - keeping the model constant while evolving:
1. Prompts (PROMPT_CHANGE, PARAM_TWEAK) - always enabled
2. Topology (ADD_ROLE, REMOVE_ROLE, REWIRE_GRAPH) - when enable_topology_evolution=True

Key Insight: The self-improvement benchmark compares Gen 0 vs Gen N
(same system evolved), NOT System A vs System B. This isolates the effect
of evolution from model differences.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from siare.benchmarks.adapters import benchmark_to_taskset
from siare.benchmarks.metrics import register_benchmark_metrics
from siare.benchmarks.runner import BenchmarkResults, BenchmarkRunner
from siare.benchmarks.tracking import (
    LearningCurveTracker,
    PromptDiffTracker,
)
from siare.core.models import StatisticalTestResult
from siare.utils.statistics import (
    wilcoxon_signed_rank_test,
)

if TYPE_CHECKING:
    from siare.benchmarks.base import BenchmarkDataset
    from siare.core.models import ProcessConfig, PromptGenome, SOPGene
    from siare.services.config_store import ConfigStore
    from siare.services.director import DirectorService
    from siare.services.evaluation_service import EvaluationService
    from siare.services.execution_engine import ExecutionEngine
    from siare.services.gene_pool import GenePool
    from siare.services.llm_provider import LLMProvider
    from siare.services.qd_grid import QDGridManager
    from siare.services.scheduler import EvolutionScheduler

logger = logging.getLogger(__name__)


def _empty_dict() -> dict[str, Any]:
    """Create empty dict for default factory."""
    return {}


def _empty_list() -> list[Any]:
    """Create empty list for default factory."""
    return []


@dataclass
class SelfImprovementConfig:
    """Configuration for self-improvement benchmarking.

    Attributes:
        max_generations: Maximum evolution generations
        population_size: Number of SOPs to evaluate per generation
        model: LLM model for execution
        reasoning_model: Model for Director diagnosis (supports reasoning)
        dataset_tier: Dataset difficulty tier (1=simple, 2=medium, 3=hard)
        max_samples: Maximum benchmark samples
        confidence_level: Statistical confidence level
        prompt_strategy: Prompt optimization strategy (adaptive/textgrad/evoprompt)
        metrics_to_optimize: Metrics to optimize during evolution
        no_early_stop: If True, disable convergence detection and run all generations
        convergence_threshold: Min improvement over convergence_window to continue
        convergence_window: Number of recent generations to check for convergence
        parallel_samples: Number of samples to evaluate concurrently (1=sequential)
        parallel_offspring: If True, evaluate offspring in parallel
    """

    max_generations: int = 10
    population_size: int = 3
    model: str = "llama3.1:8b"
    reasoning_model: str = "deepseek-r1:7b"  # For Director diagnosis
    dataset_tier: int = 1  # 1=BEIR/NQ, 2=HotpotQA, 3=FRAMES
    # Minimum 50 samples for meaningful statistical power (SE ≈ 0.07)
    # With 50 samples, can reliably detect ~14% improvements
    max_samples: int = 50
    confidence_level: float = 0.95
    prompt_strategy: str = "adaptive"
    metrics_to_optimize: list[str] = field(
        default_factory=lambda: ["benchmark_accuracy", "benchmark_f1"]
    )
    # Quick mode for testing
    quick_mode: bool = False
    # Budget constraints
    max_cost: float | None = None
    max_evaluations: int | None = None
    # Convergence control
    no_early_stop: bool = False
    convergence_threshold: float = 0.01
    convergence_window: int = 20
    # Parallelization
    parallel_samples: int = 8  # Default to 8 workers for ~8x speedup
    parallel_offspring: bool = False
    # Crash recovery
    resume: bool = False  # Resume from last checkpoint if available
    output_dir: str = "benchmarks/results/self_improvement"  # Output directory for results and checkpoints
    # Topology evolution - enables ADD_ROLE, REMOVE_ROLE, REWIRE_GRAPH mutations
    enable_topology_evolution: bool = False
    max_roles: int = 8  # Maximum roles when topology evolution is enabled
    mandatory_roles: list[str] = field(
        default_factory=lambda: ["query_decomposer", "synthesizer"]
    )
    # Reproducibility - set to any integer for deterministic benchmark runs
    random_seed: int | None = None


@dataclass
class GenerationSnapshot:
    """Snapshot of performance at a specific generation.

    Attributes:
        generation: Generation number (0-indexed)
        best_quality: Best quality score (weighted_aggregate) in this generation
        avg_quality: Average quality across offspring
        metrics: Individual metric values (accuracy, f1, etc.) for transparency
        prompt_changes: List of prompt modifications in this generation
        timestamp: When this snapshot was captured (ISO format string)
        offspring_details: Per-offspring metrics for debugging selection dynamics.
            Each dict contains:
            - offspring_id (str): Gene ID
            - parent_id (str): Parent gene ID
            - quality (float): Offspring quality score
            - is_selected_as_best (bool): Whether this became the new best
            - metrics (dict[str, float]): All metric values for this offspring
    """

    generation: int
    best_quality: float
    avg_quality: float
    metrics: dict[str, float]
    prompt_changes: list[dict[str, Any]]
    timestamp: str | float = 0.0  # ISO format string or float timestamp
    offspring_details: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class SelfImprovementResult:
    """Results from self-improvement benchmarking.

    The key comparison is initial_metrics vs evolved_metrics:
    both use the SAME model and architecture, differing only in prompts.
    """

    config: SelfImprovementConfig
    dataset_name: str
    # Prompts
    initial_prompts: dict[str, str]
    evolved_prompts: dict[str, str]
    prompt_diffs: dict[str, Any]
    # Evolution trajectory
    generation_snapshots: list[GenerationSnapshot]
    generations_run: int
    # Performance metrics
    initial_metrics: dict[str, float]
    evolved_metrics: dict[str, float]
    # Statistical significance
    significance_tests: dict[str, StatisticalTestResult]
    # Raw results for analysis
    initial_results: BenchmarkResults | None = None
    evolved_results: BenchmarkResults | None = None
    # Timing
    total_time_seconds: float = 0.0
    # Learning curve data
    learning_curve_data: dict[str, list[Any]] = field(default_factory=_empty_dict)
    # Convergence info
    converged: bool = False
    convergence_generation: int | None = None

    def summary(self) -> dict[str, Any]:
        """Return summary of self-improvement results."""
        improvements = {}
        for metric in self.evolved_metrics:
            initial = self.initial_metrics.get(metric, 0.0)
            evolved = self.evolved_metrics.get(metric, 0.0)
            abs_improvement = evolved - initial
            pct_improvement = (abs_improvement / initial * 100) if initial > 0 else 0.0
            test = self.significance_tests.get(metric)
            improvements[metric] = {
                "initial": initial,
                "evolved": evolved,
                "improvement": abs_improvement,
                "improvement_pct": f"{pct_improvement:.1f}%",
                "significant": test.isSignificant if test else False,
                "p_value": test.pValue if test else None,
            }

        return {
            "dataset": self.dataset_name,
            "model": self.config.model,
            "generations": self.generations_run,
            "improvements": improvements,
            "converged": self.converged,
            "convergence_generation": self.convergence_generation,
            "total_time_seconds": self.total_time_seconds,
            "prompt_changes": len([d for d in self.prompt_diffs.values() if d]),
        }


class SelfImprovementBenchmark:
    """Benchmark that demonstrates SIARE's self-improvement capability.

    This benchmark shows that evolved prompts outperform initial prompts
    while keeping the model and architecture constant.

    Usage:
        >>> config = SelfImprovementConfig(
        ...     max_generations=10,
        ...     model="llama3.1:8b",
        ...     reasoning_model="deepseek-r1:7b",
        ...     dataset_tier=1,
        ... )
        >>> benchmark = SelfImprovementBenchmark(
        ...     config=config,
        ...     llm_provider=provider,
        ...     base_sop=multihop_sop,
        ...     base_genome=multihop_genome,
        ... )
        >>> result = benchmark.run()
        >>> print(result.summary())
    """

    def __init__(
        self,
        config: SelfImprovementConfig,
        llm_provider: LLMProvider,
        base_sop: ProcessConfig,
        base_genome: PromptGenome,
        config_store: ConfigStore | None = None,
        gene_pool: GenePool | None = None,
        qd_grid: QDGridManager | None = None,
        execution_engine: ExecutionEngine | None = None,
        evaluation_service: EvaluationService | None = None,
        director_service: DirectorService | None = None,
        tool_adapters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the self-improvement benchmark.

        Args:
            config: Self-improvement configuration
            llm_provider: LLM provider for execution
            base_sop: Baseline SOP to evolve
            base_genome: Baseline prompt genome
            config_store: Config store (created if not provided)
            gene_pool: Gene pool (created if not provided)
            qd_grid: QD grid manager (created if not provided)
            execution_engine: Execution engine (created if not provided)
            evaluation_service: Evaluation service (created if not provided)
            director_service: Director service (created if not provided)
            tool_adapters: Tool adapters for RAG execution
        """
        self._config = config
        self._llm_provider = llm_provider
        self._base_sop = base_sop
        self._base_genome = base_genome
        self._tool_adapters = tool_adapters or {}

        # Services (initialized lazily)
        self._config_store = config_store
        self._gene_pool = gene_pool
        self._qd_grid = qd_grid
        self._execution_engine = execution_engine
        self._evaluation_service = evaluation_service
        self._director_service = director_service
        self._scheduler: EvolutionScheduler | None = None

        # Trackers
        self._prompt_tracker = PromptDiffTracker()
        self._curve_tracker = LearningCurveTracker(
            confidence_level=config.confidence_level,
        )

    def _ensure_services(self) -> None:
        """Ensure all services are initialized."""
        from siare.services.config_store import ConfigStore
        from siare.services.director import DirectorService
        from siare.services.evaluation_service import EvaluationService
        from siare.services.execution_engine import ExecutionEngine
        from siare.services.gene_pool import GenePool
        from siare.services.qd_grid import QDGridManager

        if self._config_store is None:
            self._config_store = ConfigStore()

        if self._gene_pool is None:
            self._gene_pool = GenePool()

        if self._qd_grid is None:
            self._qd_grid = QDGridManager()

        if self._execution_engine is None:
            self._execution_engine = ExecutionEngine(
                llm_provider=self._llm_provider,
                model_fallback_cascade=[self._config.model],
                tool_adapters=self._tool_adapters,
            )

        if self._evaluation_service is None:
            self._evaluation_service = EvaluationService(
                llm_provider=self._llm_provider
            )
            register_benchmark_metrics(self._evaluation_service)

        if self._director_service is None:
            # Use reasoning model for Director
            self._director_service = DirectorService(
                llm_provider=self._llm_provider,
                model=self._config.reasoning_model,
            )

    def _get_checkpoint_path(self) -> Path:
        """Get path to checkpoint file."""
        return Path(self._config.output_dir) / "checkpoint.json"

    def _save_checkpoint(
        self,
        phase: str,
        generation: int,
        initial_results: BenchmarkResults | None,
        initial_prompts: dict[str, str],
        generation_snapshots: list[GenerationSnapshot],
        best_gene_id: str | None = None,
        best_metrics: dict[str, float] | None = None,
    ) -> None:
        """Save checkpoint for crash recovery.

        Args:
            phase: Current phase (e.g., "initial_benchmark", "evolution", "evolved_benchmark")
            generation: Current generation number
            initial_results: Results from initial benchmark (if completed)
            initial_prompts: Initial prompts captured
            generation_snapshots: List of generation snapshots so far
            best_gene_id: ID of best gene found so far
            best_metrics: Best metrics achieved so far
        """
        checkpoint_path = self._get_checkpoint_path()
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "version": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "phase": phase,
            "config": {
                "model": self._config.model,
                "reasoning_model": self._config.reasoning_model,
                "max_generations": self._config.max_generations,
                "max_samples": self._config.max_samples,
                "dataset_tier": self._config.dataset_tier,
                "parallel_samples": self._config.parallel_samples,
                "parallel_offspring": self._config.parallel_offspring,
            },
            "state": {
                "current_generation": generation,
                "initial_results": initial_results.to_dict() if initial_results else None,
                "generation_snapshots": [
                    {
                        "generation": s.generation,
                        "best_quality": s.best_quality,
                        "avg_quality": s.avg_quality,
                        "metrics": s.metrics,
                        "prompt_changes": s.prompt_changes,
                        "timestamp": s.timestamp,
                    }
                    for s in generation_snapshots
                ],
                "best_gene_id": best_gene_id,
                "best_metrics": best_metrics,
            },
            "prompts": {
                "initial": initial_prompts,
            },
        }

        checkpoint_path.write_text(json.dumps(checkpoint, indent=2, default=str))
        logger.info(f"Saved checkpoint at phase={phase}, generation={generation}")

    def _load_checkpoint(self) -> dict[str, Any] | None:
        """Load checkpoint if it exists.

        Returns:
            Checkpoint data or None if no checkpoint exists
        """
        checkpoint_path = self._get_checkpoint_path()
        if not checkpoint_path.exists():
            return None

        try:
            data = json.loads(checkpoint_path.read_text())
            logger.info(
                f"Loaded checkpoint from {checkpoint_path}: "
                f"phase={data.get('phase')}, generation={data.get('state', {}).get('current_generation')}"
            )
            return data
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return None

    def _restore_from_checkpoint(
        self, checkpoint: dict[str, Any]
    ) -> tuple[BenchmarkResults | None, dict[str, str], list[GenerationSnapshot], int]:
        """Restore state from checkpoint.

        Args:
            checkpoint: Checkpoint data

        Returns:
            Tuple of (initial_results, initial_prompts, generation_snapshots, start_generation)
        """
        state = checkpoint.get("state", {})

        # Restore initial results
        initial_results = None
        if state.get("initial_results"):
            initial_results = BenchmarkResults.from_dict(state["initial_results"])

        # Restore prompts
        initial_prompts = checkpoint.get("prompts", {}).get("initial", {})

        # Restore generation snapshots
        generation_snapshots = []
        for snap_data in state.get("generation_snapshots", []):
            snapshot = GenerationSnapshot(
                generation=snap_data["generation"],
                best_quality=snap_data["best_quality"],
                avg_quality=snap_data["avg_quality"],
                metrics=snap_data["metrics"],
                prompt_changes=snap_data.get("prompt_changes", []),
                timestamp=snap_data.get("timestamp", 0.0),
            )
            generation_snapshots.append(snapshot)

        start_generation = state.get("current_generation", 0)

        return initial_results, initial_prompts, generation_snapshots, start_generation

    def _delete_checkpoint(self) -> None:
        """Delete checkpoint file after successful completion."""
        checkpoint_path = self._get_checkpoint_path()
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Deleted checkpoint: {checkpoint_path}")

    def _get_dataset_for_tier(self, tier: int) -> BenchmarkDataset:
        """Get appropriate dataset for the specified tier.

        Args:
            tier: Difficulty tier (1-3)

        Returns:
            BenchmarkDataset instance
        """
        from siare.benchmarks.datasets import BEIRDataset, FRAMESDataset
        from siare.benchmarks.hotpotqa import HotpotQADataset

        max_samples = self._config.max_samples

        if tier == 1:
            # Simple: BEIR Natural Questions
            return BEIRDataset(
                dataset_name="nq",
                max_samples=max_samples,
            )
        if tier == 2:
            # Medium: HotpotQA (multi-hop)
            return HotpotQADataset(
                max_samples=max_samples,
            )
        # Hard: FRAMES (complex multi-hop)
        return FRAMESDataset(
            max_samples=max_samples,
        )

    def _capture_initial_prompts(self) -> dict[str, str]:
        """Capture initial prompts from base genome.

        Returns:
            Dict mapping role_id to prompt text
        """
        self._prompt_tracker.capture_initial(self._base_genome)
        return self._prompt_tracker.get_initial_prompts()

    def _capture_evolved_prompts(self, evolved_genome: PromptGenome) -> dict[str, str]:
        """Capture evolved prompts and compute diffs.

        Args:
            evolved_genome: Evolved genome

        Returns:
            Dict mapping role_id to evolved prompt text
        """
        self._prompt_tracker.capture_evolved(evolved_genome)
        return self._prompt_tracker.get_evolved_prompts()

    def _compute_prompt_diffs(self) -> dict[str, Any]:
        """Compute diffs between initial and evolved prompts.

        Returns:
            Dict mapping role_id to diff data
        """
        diffs = self._prompt_tracker.compute_diffs()
        return {
            role_id: {
                "has_changes": diff.has_changes,
                "lines_added": diff.lines_added,
                "lines_removed": diff.lines_removed,
                "markdown": diff.to_markdown() if diff.has_changes else None,
            }
            for role_id, diff in diffs.items()
        }

    def _track_generation(
        self,
        generation: int,
        quality_scores: list[float],
        metrics: dict[str, float],
    ) -> GenerationSnapshot:
        """Track a generation's performance.

        Args:
            generation: Generation number
            quality_scores: Quality scores from population
            metrics: Aggregate metrics

        Returns:
            GenerationSnapshot for this generation
        """
        import numpy as np

        self._curve_tracker.add_generation(
            generation=generation,
            quality_scores=quality_scores,
            metric_values=metrics,
        )

        return GenerationSnapshot(
            generation=generation,
            best_quality=float(np.max(quality_scores)) if quality_scores else 0.0,
            avg_quality=float(np.mean(quality_scores)) if quality_scores else 0.0,
            metrics=metrics,
            prompt_changes=[],  # Filled in later
            timestamp=time.time(),
        )

    def _run_statistical_analysis(
        self,
        initial_values: dict[str, list[float]],
        evolved_values: dict[str, list[float]],
    ) -> dict[str, StatisticalTestResult]:
        """Run statistical tests comparing initial vs evolved performance.

        Args:
            initial_values: Metric values from initial SOP runs
            evolved_values: Metric values from evolved SOP runs

        Returns:
            Dict of statistical test results per metric
        """
        results: dict[str, StatisticalTestResult] = {}

        for metric in self._config.metrics_to_optimize:
            initial = initial_values.get(metric, [])
            evolved = evolved_values.get(metric, [])

            if len(initial) < 5 or len(evolved) < 5:
                # Not enough samples for Wilcoxon
                logger.warning(
                    f"Insufficient samples for {metric}: "
                    f"initial={len(initial)}, evolved={len(evolved)}"
                )
                # Create a placeholder result
                results[metric] = StatisticalTestResult(
                    testType="insufficient_samples",
                    statistic=0.0,
                    pValue=1.0,
                    isSignificant=False,
                    confidenceLevel=self._config.confidence_level,
                    hypothesis=f"Evolved prompts improve {metric}",
                )
                continue

            try:
                # Use Wilcoxon for paired samples (same queries)
                test_result = wilcoxon_signed_rank_test(
                    sample_a=evolved,  # Evolved first (alternative: greater)
                    sample_b=initial,
                    alternative="greater",  # Test if evolved > initial
                )
                results[metric] = test_result
            except Exception:
                logger.exception("Statistical test failed for %s", metric)
                results[metric] = StatisticalTestResult(
                    testType="error",
                    statistic=0.0,
                    pValue=1.0,
                    isSignificant=False,
                    confidenceLevel=self._config.confidence_level,
                    hypothesis=f"Evolved prompts improve {metric}",
                )

        return results

    def _create_scheduler(self) -> EvolutionScheduler:
        """Create an EvolutionScheduler instance with convergence config from benchmark settings."""
        from siare.core.config import ConvergenceConfig
        from siare.services.scheduler import EvolutionScheduler

        self._ensure_services()

        # Assert services are initialized (for type checker)
        assert self._config_store is not None, "ConfigStore not initialized"
        assert self._gene_pool is not None, "GenePool not initialized"
        assert self._qd_grid is not None, "QDGridManager not initialized"
        assert self._execution_engine is not None, "ExecutionEngine not initialized"
        assert self._evaluation_service is not None, "EvaluationService not initialized"
        assert self._director_service is not None, "DirectorService not initialized"

        # Create convergence config from benchmark settings
        # If no_early_stop is True, set threshold to 0 to disable convergence detection
        convergence_config = ConvergenceConfig(
            convergence_window=self._config.convergence_window,
            convergence_threshold=0.0 if self._config.no_early_stop else self._config.convergence_threshold,
        )

        return EvolutionScheduler(
            config_store=self._config_store,
            gene_pool=self._gene_pool,
            qd_grid=self._qd_grid,
            execution_engine=self._execution_engine,
            evaluation_service=self._evaluation_service,
            director_service=self._director_service,
            convergence_config=convergence_config,
            parallel_offspring=self._config.parallel_offspring,
        )

    def _create_evolution_job(
        self, dataset: BenchmarkDataset
    ) -> Any:  # Returns EvolutionJob
        """Create an EvolutionJob from the benchmark dataset."""
        import uuid

        from siare.core.models import (
            BudgetLimit,
            EvolutionConstraints,
            EvolutionJob,
            EvolutionJobStatus,
            EvolutionPhase,
            MutationType,
            SelectionStrategy,
        )

        task_set = benchmark_to_taskset(dataset)
        max_gens = 3 if self._config.quick_mode else self._config.max_generations

        # Base mutation types (prompt-only evolution)
        mutation_types = [MutationType.PROMPT_CHANGE, MutationType.PARAM_TWEAK]

        # Add topology mutations if enabled
        if self._config.enable_topology_evolution:
            mutation_types.extend([
                MutationType.ADD_ROLE,
                MutationType.REMOVE_ROLE,
                MutationType.REWIRE_GRAPH,
            ])
            logger.info(
                f"Topology evolution enabled with mutation types: "
                f"{[m.value for m in mutation_types]}"
            )

        phases = [
            EvolutionPhase(
                name="explore",
                allowedMutationTypes=mutation_types,
                selectionStrategy=SelectionStrategy.TOURNAMENT,
                parentsPerGeneration=self._config.population_size,
                maxGenerations=max_gens,
            )
        ]

        # Calculate proper evaluation budget based on samples
        # Each generation evaluates population_size offspring × num_samples
        num_samples = len(task_set.tasks)
        default_evaluations = max_gens * self._config.population_size * num_samples

        budget = BudgetLimit(
            maxCost=self._config.max_cost,
            maxEvaluations=self._config.max_evaluations or default_evaluations,
        )

        # Add topology constraints if topology evolution is enabled
        if self._config.enable_topology_evolution:
            constraints = EvolutionConstraints(
                budgetLimit=budget,
                maxRoles=self._config.max_roles,
                mandatoryRoles=self._config.mandatory_roles,
            )
        else:
            constraints = EvolutionConstraints(budgetLimit=budget)

        # Set stop conditions - disable early stopping if no_early_stop is True
        # When no_early_stop=True, set maxGenerationsWithoutImprovement to max_gens
        # to effectively disable it
        stop_config = {
            "stopConditions": {
                "maxTotalGenerations": max_gens,
                "maxGenerationsWithoutImprovement": max_gens if self._config.no_early_stop else max(2, max_gens // 2),
                "targetQualityScore": 0.95,
            }
        }

        weights = {
            metric: 1.0 / len(self._config.metrics_to_optimize)
            for metric in self._config.metrics_to_optimize
        }

        return EvolutionJob(
            id=f"self_improvement_{uuid.uuid4().hex[:8]}",
            domain=dataset.name,
            baseSops=[
                {
                    "sopId": self._base_sop.id,
                    "sopVersion": self._base_sop.version,
                    "promptGenomeId": self._base_genome.id,
                    "promptGenomeVersion": self._base_genome.version,
                }
            ],
            taskSet=task_set,
            metricsToOptimize=self._config.metrics_to_optimize,
            qualityScoreWeights=weights,
            constraints=constraints,
            phases=phases,
            status=EvolutionJobStatus.PENDING,
            config=stop_config,
        )

    def _run_initial_benchmark(
        self, dataset: BenchmarkDataset
    ) -> BenchmarkResults:
        """Run benchmark with initial SOP.

        Args:
            dataset: Benchmark dataset

        Returns:
            BenchmarkResults from initial run
        """
        from siare.core.models import MetricConfig, MetricType

        logger.info(f"Running initial benchmark on {dataset.name}")

        self._ensure_services()
        assert self._evaluation_service is not None, "EvaluationService not initialized"

        runner = BenchmarkRunner(
            sop=self._base_sop,
            genome=self._base_genome,
            llm_provider=self._llm_provider,
            model_fallback_cascade=[self._config.model],
            tool_adapters=self._tool_adapters,
        )

        metric_configs = [
            MetricConfig(
                id=m,
                type=MetricType.PROGRAMMATIC,
                fnRef=m,
                inputs=[],
                weight=1.0,
            )
            for m in self._config.metrics_to_optimize
        ]

        return runner.run_with_evaluation_parallel(
            dataset,
            evaluation_service=self._evaluation_service,
            metric_configs=metric_configs,
            max_samples=self._config.max_samples,
            max_workers=self._config.parallel_samples,
        )

    def _run_evolution(
        self,
        dataset: BenchmarkDataset,
        initial_prompts: dict[str, str],
        initial_results: BenchmarkResults | None = None,
    ) -> tuple[SOPGene, int, list[GenerationSnapshot]]:
        """Run evolution loop.

        Args:
            dataset: Benchmark dataset
            initial_prompts: Initial prompts for checkpointing
            initial_results: Initial benchmark results for checkpointing

        Returns:
            Tuple of (best_gene, generations_run, generation_snapshots)
        """
        from siare.core.models import MetricConfig, MetricType, SOPGene

        logger.info(f"Starting evolution on {dataset.name}")

        # Track snapshots incrementally for checkpointing
        _generation_snapshots: list[GenerationSnapshot] = []

        def _on_generation_complete(generation: int, stats: dict[str, Any]) -> None:
            """Callback to save checkpoint after each generation.

            Captures both weighted_aggregate (best_quality) and individual metrics
            for transparency in learning curve visualization.
            """
            # Extract individual metrics from scheduler stats for transparency
            individual_metrics = stats.get("individual_metrics", {})
            offspring_details = stats.get("offspring_details", [])

            # Extract quality scores from offspring for learning curve tracking
            quality_scores = [
                od.get("quality", 0.0) for od in offspring_details
            ] or [stats.get("best_quality", 0.0)]

            # Track learning curve for visualization
            self._curve_tracker.add_generation(
                generation=generation,
                quality_scores=quality_scores,
                metric_values=individual_metrics,
            )

            # Create snapshot with both aggregate quality and individual metrics
            snapshot = GenerationSnapshot(
                generation=generation,
                best_quality=stats.get("best_quality", 0.0),  # weighted_aggregate
                avg_quality=stats.get("best_quality", 0.0),  # Use best as proxy
                metrics=individual_metrics,  # Individual metrics for transparency
                prompt_changes=[],
                timestamp=datetime.now(timezone.utc).isoformat(),
                offspring_details=offspring_details,
            )
            _generation_snapshots.append(snapshot)

            # Save checkpoint
            self._save_checkpoint(
                phase="evolution",
                generation=generation,
                initial_results=initial_results,
                initial_prompts=initial_prompts,
                generation_snapshots=_generation_snapshots,
            )
            logger.info(f"Checkpoint saved after generation {generation}")

        scheduler = self._create_scheduler()
        job = self._create_evolution_job(dataset)

        # Assert services are initialized (for type checker)
        assert self._config_store is not None, "ConfigStore not initialized"
        assert self._gene_pool is not None, "GenePool not initialized"

        # Store base SOP and genome
        self._config_store.save_sop(self._base_sop)
        self._config_store.save_prompt_genome(self._base_genome)

        # Store metric configs
        for m in self._config.metrics_to_optimize:
            metric_config = MetricConfig(
                id=m,
                type=MetricType.PROGRAMMATIC,
                fnRef=m,
                inputs=[],
                weight=1.0 / len(self._config.metrics_to_optimize),
            )
            self._config_store.save_metric(metric_config)

        # Run evolution with checkpoint callback
        completed_job = scheduler.run_to_completion(
            job, verbose=True, on_generation_complete=_on_generation_complete
        )

        # Trust scheduler's selection - this is not the benchmark's responsibility
        # The scheduler tracks bestSopSoFar which contains sopId, version, quality, metrics
        if not completed_job.bestSopSoFar:
            raise RuntimeError("Evolution completed but no best SOP found")

        best_sop_info = completed_job.bestSopSoFar
        best_gene = self._gene_pool.get_sop_gene(
            best_sop_info["sopId"], best_sop_info.get("version")
        )

        if best_gene is None:
            # Use genome IDs from bestSopSoFar (more reliable than gene pool lookup)
            # These IDs were added to bestSopSoFar in the scheduler for exactly this case
            prompt_genome_id = best_sop_info.get(
                "promptGenomeId", self._base_genome.id
            )
            prompt_genome_version = best_sop_info.get(
                "promptGenomeVersion", self._base_genome.version
            )
            logger.warning(
                f"Could not find best gene {best_sop_info['sopId']}@{best_sop_info.get('version')} "
                f"in gene pool, using genome IDs from bestSopSoFar: "
                f"{prompt_genome_id}@{prompt_genome_version}"
            )
            best_gene = SOPGene(
                sopId=best_sop_info["sopId"],
                version=best_sop_info.get("version", self._base_sop.version),
                promptGenomeId=prompt_genome_id,
                promptGenomeVersion=prompt_genome_version,
                configSnapshot=self._base_sop,
                evaluations=[],
                aggregatedMetrics={},
            )

        # Use snapshots from checkpoint callback (captured during evolution)
        return best_gene, completed_job.currentGeneration, _generation_snapshots

    def _run_evolved_benchmark(
        self,
        dataset: BenchmarkDataset,
        evolved_gene: SOPGene,
    ) -> BenchmarkResults:
        """Run benchmark with evolved SOP.

        Args:
            dataset: Benchmark dataset
            evolved_gene: Best evolved gene

        Returns:
            BenchmarkResults from evolved run
        """
        from siare.core.models import MetricConfig, MetricType

        logger.info(f"Running evolved benchmark on {dataset.name}")

        # Assert services are initialized (for type checker)
        assert self._config_store is not None, "ConfigStore not initialized"
        assert self._evaluation_service is not None, "EvaluationService not initialized"

        # Enhanced logging for genome lookup debugging
        logger.info(
            f"Looking up evolved gene: sopId={evolved_gene.sopId}, version={evolved_gene.version}, "
            f"genomeId={evolved_gene.promptGenomeId}, genomeVersion={evolved_gene.promptGenomeVersion}"
        )

        evolved_sop = self._config_store.get_sop(
            evolved_gene.sopId, evolved_gene.version
        )
        evolved_genome = self._config_store.get_prompt_genome(
            evolved_gene.promptGenomeId, evolved_gene.promptGenomeVersion
        )

        if evolved_sop is None:
            # Log available SOPs for debugging
            available_sops = getattr(self._config_store, "_sops", {})
            available_keys = list(available_sops.keys())[:10]  # First 10 for brevity
            logger.warning(
                f"Evolved SOP not found in config store: "
                f"{evolved_gene.sopId}@{evolved_gene.version}, using baseline. "
                f"Available SOP keys (first 10): {available_keys}"
            )
            evolved_sop = self._base_sop
        if evolved_genome is None:
            # Log available genomes for debugging
            available_genomes = getattr(self._config_store, "_prompt_genomes", {})
            available_keys = list(available_genomes.keys())[:10]  # First 10 for brevity
            logger.warning(
                f"Evolved genome not found in config store: "
                f"{evolved_gene.promptGenomeId}@{evolved_gene.promptGenomeVersion}, "
                f"using baseline. Prompt diffs will not be detected. "
                f"Available genome keys (first 10): {available_keys}"
            )
            evolved_genome = self._base_genome
        else:
            logger.info(
                f"Successfully loaded evolved genome: {evolved_gene.promptGenomeId}@"
                f"{evolved_gene.promptGenomeVersion}"
            )

        # Capture evolved prompts
        self._capture_evolved_prompts(evolved_genome)

        runner = BenchmarkRunner(
            sop=evolved_sop,
            genome=evolved_genome,
            llm_provider=self._llm_provider,
            model_fallback_cascade=[self._config.model],
            tool_adapters=self._tool_adapters,
        )

        metric_configs = [
            MetricConfig(
                id=m,
                type=MetricType.PROGRAMMATIC,
                fnRef=m,
                inputs=[],
                weight=1.0,
            )
            for m in self._config.metrics_to_optimize
        ]

        return runner.run_with_evaluation_parallel(
            dataset,
            evaluation_service=self._evaluation_service,
            metric_configs=metric_configs,
            max_samples=self._config.max_samples,
            max_workers=self._config.parallel_samples,
        )

    def run(self, dataset: BenchmarkDataset | None = None) -> SelfImprovementResult:
        """Run the self-improvement benchmark.

        Supports resume from checkpoint if config.resume=True and a checkpoint exists.

        Args:
            dataset: Optional dataset override. If not provided,
                     uses dataset based on config.dataset_tier

        Returns:
            SelfImprovementResult with comparison data
        """
        # Set random seed for reproducibility if configured
        if self._config.random_seed is not None:
            import random

            import numpy as np

            np.random.seed(self._config.random_seed)
            random.seed(self._config.random_seed)
            logger.info(f"Set random seed to {self._config.random_seed} for reproducibility")

        start_time = time.time()

        # Get dataset
        if dataset is None:
            dataset = self._get_dataset_for_tier(self._config.dataset_tier)

        # Check for checkpoint if resume is enabled
        initial_results: BenchmarkResults | None = None
        initial_prompts: dict[str, str] = {}
        snapshots: list[GenerationSnapshot] = []
        resumed_from_checkpoint = False

        if self._config.resume:
            checkpoint = self._load_checkpoint()
            if checkpoint:
                phase = checkpoint.get("phase", "")
                logger.info(f"Resuming from checkpoint at phase: {phase}")
                initial_results, initial_prompts, snapshots, _ = self._restore_from_checkpoint(checkpoint)
                resumed_from_checkpoint = True

                # If we've already completed initial benchmark, skip to evolution
                if phase in ("evolution", "evolved_benchmark") and initial_results:
                    logger.info("Skipping initial benchmark (already completed)")

        logger.info(
            f"Starting self-improvement benchmark: "
            f"model={self._config.model}, "
            f"reasoning_model={self._config.reasoning_model}, "
            f"generations={self._config.max_generations}"
            + (" (RESUMED)" if resumed_from_checkpoint else "")
        )

        # Phase 1: Capture initial prompts
        if not initial_prompts:
            logger.info("Phase 1: Capturing initial prompts")
            initial_prompts = self._capture_initial_prompts()
        else:
            logger.info("Phase 1: Using initial prompts from checkpoint")
            # Restore prompt tracker state
            self._prompt_tracker.set_initial_prompts(initial_prompts)

        # Phase 2: Run initial benchmark
        if initial_results is None:
            logger.info("Phase 2: Running initial benchmark")
            initial_results = self._run_initial_benchmark(dataset)
            # Save checkpoint after initial benchmark
            self._save_checkpoint(
                phase="initial_benchmark",
                generation=0,
                initial_results=initial_results,
                initial_prompts=initial_prompts,
                generation_snapshots=[],
            )
        else:
            logger.info("Phase 2: Using initial benchmark results from checkpoint")

        # Phase 3: Run evolution
        logger.info("Phase 3: Running evolution loop")
        evolved_gene, generations, snapshots = self._run_evolution(
            dataset, initial_prompts, initial_results
        )

        # Save checkpoint after evolution
        self._save_checkpoint(
            phase="evolution",
            generation=generations,
            initial_results=initial_results,
            initial_prompts=initial_prompts,
            generation_snapshots=snapshots,
            best_gene_id=evolved_gene.sopId if evolved_gene else None,
        )

        # Phase 4: Run evolved benchmark
        logger.info("Phase 4: Running evolved benchmark")
        evolved_results = self._run_evolved_benchmark(dataset, evolved_gene)

        # Phase 5: Compute prompt diffs
        logger.info("Phase 5: Computing prompt diffs")
        prompt_diffs = self._compute_prompt_diffs()
        evolved_prompts = self._prompt_tracker.get_evolved_prompts()

        # Phase 6: Statistical analysis
        logger.info("Phase 6: Running statistical analysis")
        initial_values = {
            m: [r.metrics.get(m, 0.0) for r in initial_results.sample_results]
            for m in self._config.metrics_to_optimize
        }
        evolved_values = {
            m: [r.metrics.get(m, 0.0) for r in evolved_results.sample_results]
            for m in self._config.metrics_to_optimize
        }
        significance_tests = self._run_statistical_analysis(initial_values, evolved_values)

        # Get learning curve data
        curve_data = self._curve_tracker.get_curve_data()
        convergence_info = self._curve_tracker.get_convergence_info()

        total_time = time.time() - start_time

        # Delete checkpoint on successful completion
        self._delete_checkpoint()

        return SelfImprovementResult(
            config=self._config,
            dataset_name=dataset.name,
            initial_prompts=initial_prompts,
            evolved_prompts=evolved_prompts,
            prompt_diffs=prompt_diffs,
            generation_snapshots=snapshots,
            generations_run=generations,
            initial_metrics=initial_results.aggregate_metrics,
            evolved_metrics=evolved_results.aggregate_metrics,
            significance_tests=significance_tests,
            initial_results=initial_results,
            evolved_results=evolved_results,
            total_time_seconds=total_time,
            learning_curve_data=curve_data,
            converged=convergence_info.converged,
            convergence_generation=convergence_info.convergence_generation,
        )

    def run_quick(self, dataset: BenchmarkDataset | None = None) -> SelfImprovementResult:
        """Run a quick version of the benchmark (3 generations).

        Args:
            dataset: Optional dataset override

        Returns:
            SelfImprovementResult
        """
        # Override config for quick mode
        original_max_gen = self._config.max_generations
        original_quick = self._config.quick_mode
        original_samples = self._config.max_samples

        self._config.max_generations = 3
        self._config.quick_mode = True
        self._config.max_samples = min(20, self._config.max_samples)

        try:
            return self.run(dataset)
        finally:
            self._config.max_generations = original_max_gen
            self._config.quick_mode = original_quick
            self._config.max_samples = original_samples
