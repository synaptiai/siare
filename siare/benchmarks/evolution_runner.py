"""Evolution-aware benchmark runner.

Integrates SIARE's evolution loop with the benchmark suite to demonstrate
evolved SOPs vs baseline SOPs.
"""
import logging
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from siare.benchmarks.adapters import benchmark_to_taskset
from siare.benchmarks.metrics import register_benchmark_metrics
from siare.benchmarks.runner import BenchmarkResults, BenchmarkRunner


if TYPE_CHECKING:
    from siare.benchmarks.base import BenchmarkDataset
    from siare.core.models import (
        EvolutionJob,
        ProcessConfig,
        PromptGenome,
        SOPGene,
    )
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


@dataclass
class EvolutionBenchmarkConfig:
    """Configuration for evolution-aware benchmarking.

    Attributes:
        max_generations: Maximum evolution generations (default 5 for quick, 10+ for full)
        population_size: Number of SOPs to evaluate per generation
        mutation_types: Types of mutations to allow
        metrics_to_optimize: Metrics to optimize during evolution
        model: LLM model to use (e.g., "llama3.2:1b")
        quick_mode: If True, runs minimal evolution (2-3 generations)
        max_samples: Maximum benchmark samples to use
    """

    max_generations: int = 5
    population_size: int = 3
    mutation_types: list[str] = field(
        default_factory=lambda: ["PROMPT_CHANGE", "PARAM_TWEAK"]
    )
    metrics_to_optimize: list[str] = field(
        default_factory=lambda: ["benchmark_accuracy", "benchmark_f1"]
    )
    model: str = "llama3.2:1b"
    quick_mode: bool = False
    max_samples: Optional[int] = None
    # Budget constraints
    max_cost: Optional[float] = None
    max_evaluations: Optional[int] = None


@dataclass
class StatisticalComparison:
    """Statistical comparison between baseline and evolved results.

    Attributes:
        metric_name: Name of the metric being compared
        baseline_mean: Mean score for baseline SOP
        evolved_mean: Mean score for evolved SOP
        improvement: Absolute improvement (evolved - baseline)
        improvement_pct: Percentage improvement
        baseline_ci: 95% confidence interval for baseline
        evolved_ci: 95% confidence interval for evolved
        p_value: Statistical significance (if available)
    """

    metric_name: str
    baseline_mean: float
    evolved_mean: float
    improvement: float
    improvement_pct: float
    baseline_ci: tuple[float, float] = (0.0, 0.0)
    evolved_ci: tuple[float, float] = (0.0, 0.0)
    p_value: Optional[float] = None


@dataclass
class GenerationHistoryEntry:
    """Record of a single generation during evolution."""

    generation: int
    best_quality: float
    avg_quality: float
    population_size: int
    metrics: dict[str, float] = field(default_factory=_empty_dict)
    best_sop_id: str = ""
    timestamp: float = 0.0


@dataclass
class EvolutionBenchmarkResult:
    """Results from evolution-aware benchmarking.

    Contains baseline results, evolved results, and statistical comparison.
    """

    dataset_name: str
    config: EvolutionBenchmarkConfig
    baseline_results: BenchmarkResults
    evolved_results: BenchmarkResults
    baseline_sop_id: str
    evolved_sop_id: str
    generations_run: int
    comparisons: list[StatisticalComparison] = field(default_factory=list)
    evolution_trace: dict[str, Any] = field(default_factory=_empty_dict)
    generation_history: list[GenerationHistoryEntry] = field(default_factory=list)
    total_time_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Return summary of evolution benchmark results."""
        return {
            "dataset": self.dataset_name,
            "generations": self.generations_run,
            "baseline": {
                "sop_id": self.baseline_sop_id,
                "completed": self.baseline_results.completed_samples,
                "metrics": self.baseline_results.aggregate_metrics,
            },
            "evolved": {
                "sop_id": self.evolved_sop_id,
                "completed": self.evolved_results.completed_samples,
                "metrics": self.evolved_results.aggregate_metrics,
            },
            "improvements": {
                c.metric_name: {
                    "baseline": c.baseline_mean,
                    "evolved": c.evolved_mean,
                    "improvement": c.improvement,
                    "improvement_pct": f"{c.improvement_pct:.1f}%",
                }
                for c in self.comparisons
            },
            "total_time_seconds": self.total_time_seconds,
        }


class EvolutionBenchmarkRunner:
    """Runs benchmarks with SIARE evolution integration.

    This runner:
    1. Establishes a baseline using an initial SOP
    2. Runs the evolution loop to improve the SOP
    3. Benchmarks the evolved SOP
    4. Compares results statistically

    Example:
        >>> config = EvolutionBenchmarkConfig(max_generations=5, quick_mode=True)
        >>> runner = EvolutionBenchmarkRunner(
        ...     config=config,
        ...     llm_provider=provider,
        ...     base_sop=my_sop,
        ...     base_genome=my_genome,
        ... )
        >>> result = runner.run(FRAMESDataset(max_samples=50))
        >>> print(result.summary())
    """

    def __init__(
        self,
        config: EvolutionBenchmarkConfig,
        llm_provider: "LLMProvider",
        base_sop: "ProcessConfig",
        base_genome: "PromptGenome",
        config_store: Optional["ConfigStore"] = None,
        gene_pool: Optional["GenePool"] = None,
        qd_grid: Optional["QDGridManager"] = None,
        execution_engine: Optional["ExecutionEngine"] = None,
        evaluation_service: Optional["EvaluationService"] = None,
        director_service: Optional["DirectorService"] = None,
        tool_adapters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize the evolution benchmark runner.

        Args:
            config: Evolution benchmark configuration
            llm_provider: LLM provider for execution and evolution
            base_sop: Baseline SOP to start evolution from
            base_genome: Baseline prompt genome
            config_store: Config store (created if not provided)
            gene_pool: Gene pool (created if not provided)
            qd_grid: QD grid manager (created if not provided)
            execution_engine: Execution engine (created if not provided)
            evaluation_service: Evaluation service (created if not provided)
            director_service: Director service (created if not provided)
            tool_adapters: Dictionary mapping tool names to adapter functions
                (e.g., {"vector_search": adapter.execute, "web_search": adapter.execute})
        """
        self._config = config
        self._llm_provider = llm_provider
        self._base_sop = base_sop
        self._base_genome = base_genome
        self._tool_adapters = tool_adapters or {}

        # Initialize services lazily
        self._config_store = config_store
        self._gene_pool = gene_pool
        self._qd_grid = qd_grid
        self._execution_engine = execution_engine
        self._evaluation_service = evaluation_service
        self._director_service = director_service
        self._scheduler: Optional[EvolutionScheduler] = None

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
            # Use configured model as sole fallback for Ollama/local benchmarks
            # This prevents cross-provider fallback to OpenAI/Anthropic models
            self._execution_engine = ExecutionEngine(
                llm_provider=self._llm_provider,
                model_fallback_cascade=[self._config.model],
                tool_adapters=self._tool_adapters,
            )

        if self._evaluation_service is None:
            self._evaluation_service = EvaluationService(
                llm_provider=self._llm_provider
            )
            # Register benchmark metrics
            register_benchmark_metrics(self._evaluation_service)

        if self._director_service is None:
            self._director_service = DirectorService(
                llm_provider=self._llm_provider,
                model=self._config.model,
            )

    def _create_scheduler(self) -> "EvolutionScheduler":
        """Create an EvolutionScheduler instance."""
        from siare.services.scheduler import EvolutionScheduler

        self._ensure_services()

        return EvolutionScheduler(
            config_store=self._config_store,
            gene_pool=self._gene_pool,
            qd_grid=self._qd_grid,
            execution_engine=self._execution_engine,
            evaluation_service=self._evaluation_service,
            director_service=self._director_service,
        )

    def _create_evolution_job(
        self, dataset: "BenchmarkDataset"
    ) -> "EvolutionJob":
        """Create an EvolutionJob from the benchmark dataset.

        Args:
            dataset: Benchmark dataset to use as task set

        Returns:
            EvolutionJob configured for benchmark evolution
        """
        from siare.core.models import (
            BudgetLimit,
            EvolutionConstraints,
            EvolutionJob,
            EvolutionJobStatus,
            EvolutionPhase,
            MutationType,
            SelectionStrategy,
        )

        # Convert dataset to task set
        task_set = benchmark_to_taskset(dataset)

        # Determine max generations based on mode
        max_gens = 3 if self._config.quick_mode else self._config.max_generations

        # Convert mutation type strings to MutationType enum
        mutation_types = [
            MutationType(mt.lower()) if isinstance(mt, str) else mt
            for mt in self._config.mutation_types
        ]

        # Create evolution phases
        phases = [
            EvolutionPhase(
                name="explore",
                allowedMutationTypes=mutation_types,
                selectionStrategy=SelectionStrategy.TOURNAMENT,
                parentsPerGeneration=self._config.population_size,
                maxGenerations=max_gens,
            )
        ]

        # Create budget limits
        budget = BudgetLimit(
            maxCost=self._config.max_cost,
            maxEvaluations=self._config.max_evaluations or (max_gens * self._config.population_size * 10),
        )

        # Create constraints (for safety, budget limits, role limits)
        constraints = EvolutionConstraints(
            budgetLimit=budget,
        )

        # Stop conditions go in config dict
        stop_config = {
            "stopConditions": {
                "maxTotalGenerations": max_gens,
                "maxGenerationsWithoutImprovement": max(2, max_gens // 2),
                "targetQualityScore": 0.95,
            }
        }

        # Create quality score weights (equal weighting)
        weights = {
            metric: 1.0 / len(self._config.metrics_to_optimize)
            for metric in self._config.metrics_to_optimize
        }

        return EvolutionJob(
            id=f"benchmark_evolution_{uuid.uuid4().hex[:8]}",
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

    def _run_baseline(
        self, dataset: "BenchmarkDataset"
    ) -> BenchmarkResults:
        """Run baseline benchmark with initial SOP.

        Args:
            dataset: Benchmark dataset

        Returns:
            BenchmarkResults from baseline run
        """
        from siare.core.models import MetricConfig, MetricType

        logger.info(f"Running baseline benchmark on {dataset.name}")

        # Ensure evaluation service is initialized
        self._ensure_services()

        runner = BenchmarkRunner(
            sop=self._base_sop,
            genome=self._base_genome,
            llm_provider=self._llm_provider,
            model_fallback_cascade=[self._config.model],
            tool_adapters=self._tool_adapters,
        )

        # Create metric configs for evaluation
        # fnRef must match the registered function name in register_benchmark_metrics()
        metric_configs = [
            MetricConfig(
                id=m,
                type=MetricType.PROGRAMMATIC,
                fnRef=m,  # Function reference matches registered name
                inputs=[],  # Not used for programmatic metrics with task_data
                weight=1.0,
            )
            for m in self._config.metrics_to_optimize
        ]

        return runner.run_with_evaluation(
            dataset,
            evaluation_service=self._evaluation_service,
            metric_configs=metric_configs,
            max_samples=self._config.max_samples,
        )

    def _run_evolution(
        self, dataset: "BenchmarkDataset"
    ) -> tuple["SOPGene", int]:
        """Run evolution loop on benchmark dataset.

        Args:
            dataset: Benchmark dataset for evolution

        Returns:
            Tuple of (best evolved SOPGene, generations run)
        """
        logger.info(f"Starting evolution on {dataset.name}")

        # Create scheduler and job
        scheduler = self._create_scheduler()
        job = self._create_evolution_job(dataset)

        # Store base SOP and genome in config store
        self._config_store.save_sop(self._base_sop)
        self._config_store.save_prompt_genome(self._base_genome)

        # Store metric configs for scheduler to retrieve during evolution
        # Without this, scheduler._evaluate_sop() will get None from config_store.get_metric()
        from siare.core.models import MetricConfig, MetricType

        num_metrics = len(self._config.metrics_to_optimize)
        for m in self._config.metrics_to_optimize:
            metric_config = MetricConfig(
                id=m,
                type=MetricType.PROGRAMMATIC,
                fnRef=m,
                inputs=[],
                weight=1.0 / num_metrics if num_metrics > 0 else 1.0,
            )
            self._config_store.save_metric(metric_config)

        # Run evolution
        completed_job = scheduler.run_to_completion(job, verbose=True)

        # Get best SOP from gene pool by finding the one with highest aggregate metrics
        all_genes = self._gene_pool.get_genes_from_recent_generations(
            lookback=completed_job.currentGeneration + 1
        )

        best_gene = None
        best_score = -float("inf")

        for gene in all_genes:
            # Calculate average score across metrics we care about
            score = 0.0
            count = 0
            for metric in self._config.metrics_to_optimize:
                if metric in gene.aggregatedMetrics:
                    score += gene.aggregatedMetrics[metric].mean
                    count += 1
            if count > 0:
                avg_score = score / count
                if avg_score > best_score:
                    best_score = avg_score
                    best_gene = gene

        if best_gene is None:
            # Fall back to baseline if no evolution happened
            from siare.core.models import SOPGene

            best_gene = SOPGene(
                sopId=self._base_sop.id,
                version=self._base_sop.version,
                promptGenomeId=self._base_genome.id,
                promptGenomeVersion=self._base_genome.version,
                configSnapshot=self._base_sop,
                evaluations=[],
                aggregatedMetrics={},
            )

        return best_gene, completed_job.currentGeneration

    def _run_evolved(
        self,
        dataset: "BenchmarkDataset",
        evolved_gene: "SOPGene",
    ) -> BenchmarkResults:
        """Run benchmark with evolved SOP.

        Args:
            dataset: Benchmark dataset
            evolved_gene: Best evolved SOPGene

        Returns:
            BenchmarkResults from evolved SOP
        """
        from siare.core.models import MetricConfig, MetricType

        logger.info(f"Running evolved benchmark on {dataset.name}")

        # Get evolved SOP and genome from config store
        evolved_sop = self._config_store.get_sop(
            evolved_gene.sopId, evolved_gene.version
        )
        evolved_genome = self._config_store.get_prompt_genome(
            evolved_gene.promptGenomeId, evolved_gene.promptGenomeVersion
        )

        # Fall back to base if not found (shouldn't happen)
        if evolved_sop is None:
            evolved_sop = self._base_sop
        if evolved_genome is None:
            evolved_genome = self._base_genome

        runner = BenchmarkRunner(
            sop=evolved_sop,
            genome=evolved_genome,
            llm_provider=self._llm_provider,
            model_fallback_cascade=[self._config.model],
            tool_adapters=self._tool_adapters,
        )

        # Create metric configs for evaluation
        # fnRef must match the registered function name in register_benchmark_metrics()
        metric_configs = [
            MetricConfig(
                id=m,
                type=MetricType.PROGRAMMATIC,
                fnRef=m,  # Function reference matches registered name
                inputs=[],  # Not used for programmatic metrics with task_data
                weight=1.0,
            )
            for m in self._config.metrics_to_optimize
        ]

        return runner.run_with_evaluation(
            dataset,
            evaluation_service=self._evaluation_service,
            metric_configs=metric_configs,
            max_samples=self._config.max_samples,
        )

    def _compare_results(
        self,
        baseline: BenchmarkResults,
        evolved: BenchmarkResults,
    ) -> list[StatisticalComparison]:
        """Compare baseline and evolved results statistically.

        Args:
            baseline: Baseline benchmark results
            evolved: Evolved benchmark results

        Returns:
            List of statistical comparisons for each metric
        """

        comparisons = []

        # Get all metrics from both results
        all_metrics = set(baseline.aggregate_metrics.keys()) | set(
            evolved.aggregate_metrics.keys()
        )

        for metric in all_metrics:
            baseline_mean = baseline.aggregate_metrics.get(metric, 0.0)
            evolved_mean = evolved.aggregate_metrics.get(metric, 0.0)

            improvement = evolved_mean - baseline_mean
            improvement_pct = (
                (improvement / baseline_mean * 100) if baseline_mean > 0 else 0.0
            )

            # Calculate confidence intervals from sample results
            baseline_values = [
                r.metrics.get(metric, 0.0) for r in baseline.sample_results
            ]
            evolved_values = [
                r.metrics.get(metric, 0.0) for r in evolved.sample_results
            ]

            baseline_ci = self._calculate_ci(baseline_values)
            evolved_ci = self._calculate_ci(evolved_values)

            # Simple significance test (Mann-Whitney U if scipy available)
            p_value = self._calculate_p_value(baseline_values, evolved_values)

            comparisons.append(
                StatisticalComparison(
                    metric_name=metric,
                    baseline_mean=baseline_mean,
                    evolved_mean=evolved_mean,
                    improvement=improvement,
                    improvement_pct=improvement_pct,
                    baseline_ci=baseline_ci,
                    evolved_ci=evolved_ci,
                    p_value=p_value,
                )
            )

        return comparisons

    def _calculate_ci(
        self, values: list[float], confidence: float = 0.95
    ) -> tuple[float, float]:
        """Calculate confidence interval for values.

        Args:
            values: List of metric values
            confidence: Confidence level (default 0.95)

        Returns:
            Tuple of (lower, upper) bounds
        """
        import math

        if not values:
            return (0.0, 0.0)

        n = len(values)
        mean = sum(values) / n

        if n < 2:
            return (mean, mean)

        variance = sum((x - mean) ** 2 for x in values) / (n - 1)
        std = math.sqrt(variance)

        # Use 1.96 for 95% CI (approximation)
        z = 1.96 if confidence == 0.95 else 2.576
        margin = z * std / math.sqrt(n)

        return (mean - margin, mean + margin)

    def _calculate_p_value(
        self, baseline: list[float], evolved: list[float]
    ) -> Optional[float]:
        """Calculate p-value for difference between groups.

        Args:
            baseline: Baseline metric values
            evolved: Evolved metric values

        Returns:
            p-value if calculable, None otherwise
        """
        try:
            from scipy import stats

            if len(baseline) < 2 or len(evolved) < 2:
                return None

            # Mann-Whitney U test (non-parametric)
            stat, p_value = stats.mannwhitneyu(
                baseline, evolved, alternative="two-sided"
            )
            return float(p_value)
        except ImportError:
            return None
        except Exception:
            return None

    def run(self, dataset: "BenchmarkDataset") -> EvolutionBenchmarkResult:
        """Run full evolution benchmark.

        Args:
            dataset: Benchmark dataset to evaluate

        Returns:
            EvolutionBenchmarkResult with baseline, evolved, and comparison
        """
        import time

        start_time = time.time()

        # Phase 1: Baseline evaluation
        logger.info("Phase 1: Running baseline evaluation")
        baseline_results = self._run_baseline(dataset)

        # Phase 2: Evolution loop
        logger.info("Phase 2: Running evolution loop")
        evolved_gene, generations = self._run_evolution(dataset)

        # Phase 3: Evolved SOP benchmarking
        logger.info("Phase 3: Running evolved benchmark")
        evolved_results = self._run_evolved(dataset, evolved_gene)

        # Phase 4: Statistical comparison
        logger.info("Phase 4: Computing statistical comparison")
        comparisons = self._compare_results(baseline_results, evolved_results)

        total_time = time.time() - start_time

        return EvolutionBenchmarkResult(
            dataset_name=dataset.name,
            config=self._config,
            baseline_results=baseline_results,
            evolved_results=evolved_results,
            baseline_sop_id=self._base_sop.id,
            evolved_sop_id=evolved_gene.sopId,
            generations_run=generations,
            comparisons=comparisons,
            evolution_trace={
                "generations": generations,
                "evolved_sop_version": evolved_gene.version,
                "evolved_genome_version": evolved_gene.promptGenomeVersion,
            },
            total_time_seconds=total_time,
        )

    def run_quick_validation(
        self, dataset: "BenchmarkDataset"
    ) -> EvolutionBenchmarkResult:
        """Run quick validation (2-3 generations) for testing.

        Args:
            dataset: Benchmark dataset

        Returns:
            EvolutionBenchmarkResult from quick run
        """
        # Override config for quick mode
        original_max_gen = self._config.max_generations
        original_quick = self._config.quick_mode

        self._config.max_generations = 3
        self._config.quick_mode = True

        try:
            return self.run(dataset)
        finally:
            self._config.max_generations = original_max_gen
            self._config.quick_mode = original_quick
