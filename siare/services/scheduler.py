"""Scheduler - Evolution loop orchestrator"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from siare.core.config import ConvergenceConfig
from siare.core.constants import MIN_PARENTS_FOR_CROSSOVER
from siare.core.hooks import HookContext, HookRegistry, fire_evolution_hook
from siare.core.models import (
    AggregatedMetric,
    AggregationMethod,
    EvaluationVector,
    EvolutionJob,
    EvolutionJobStatus,
    MutationType,
    ProcessConfig,
    PromptGenome,
    SelectionStrategy,
    SOPGene,
    TaskSet,
)
from siare.services.config_store import ConfigStore
from siare.services.director import DirectorService
from siare.services.evaluation_service import EvaluationService
from siare.services.execution_engine import ExecutionEngine
from siare.services.gene_pool import GenePool
from siare.services.qd_grid import QDGridManager
from siare.services.retry_handler import RetryHandler
from siare.services.selection.factory import SelectionStrategyFactory
from siare.utils.diff import compute_prompt_diff
from siare.utils.file_utils import atomic_write_json
from siare.utils.graph_viz import generate_ascii_graph
from siare.utils.weighted_aggregation import apply_task_weights


class EvolutionScheduler:
    """
    Orchestrates the evolution loop

    Responsibilities:
    - Manage evolution job lifecycle
    - Select parents for mutation
    - Execute and evaluate offspring
    - Update gene pool and QD grid
    - Track budget and convergence
    - Handle phase transitions
    """

    # Checkpoint version for backward compatibility
    CHECKPOINT_VERSION = "1.0"

    # Default checkpoint interval (save every N generations)
    DEFAULT_CHECKPOINT_INTERVAL = 5

    def __init__(
        self,
        config_store: ConfigStore,
        gene_pool: GenePool,
        qd_grid: QDGridManager,
        execution_engine: ExecutionEngine,
        evaluation_service: EvaluationService,
        director_service: DirectorService,
        checkpoint_dir: str | None = None,
        checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
        retry_handler: RetryHandler | None = None,
        convergence_config: ConvergenceConfig | None = None,
        parallel_offspring: bool = False,
    ):
        """
        Initialize Scheduler

        Args:
            config_store: Configuration storage
            gene_pool: Gene pool for tracking population
            qd_grid: Quality-diversity grid
            execution_engine: SOP execution engine
            evaluation_service: Evaluation service
            director_service: Director for mutations
            checkpoint_dir: Directory for checkpoint storage (None = no checkpointing)
            checkpoint_interval: Save checkpoint every N generations (default: 5)
            retry_handler: Retry handler for file I/O (creates default if None)
            convergence_config: Convergence detection settings (uses defaults if None)
            parallel_offspring: If True, evaluate offspring in parallel (default: False)
        """
        self.config_store = config_store
        self.parallel_offspring = parallel_offspring
        self.gene_pool = gene_pool
        self.qd_grid = qd_grid
        self.execution_engine = execution_engine
        self.evaluation_service = evaluation_service
        self.director_service = director_service

        # Selection strategy factory
        self.selection_factory = SelectionStrategyFactory()

        # Checkpoint configuration
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.checkpoint_interval = checkpoint_interval
        self.retry_handler = retry_handler or RetryHandler()

        # Create checkpoint directory if specified
        if self.checkpoint_dir:
            try:
                self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
            except Exception as e:
                logger.warning(f"Failed to create checkpoint directory: {e}")
                self.checkpoint_dir = None

        # Current evolution job
        self.current_job: EvolutionJob | None = None

        # Job tracking
        self.generation_history: list[dict[str, Any]] = []
        self.best_quality_history: list[float] = []

        # Convergence configuration
        self.convergence_config = convergence_config or ConvergenceConfig()

        # Evaluation cache: key = hash(genome_prompts + task_ids), value = evaluations
        # This avoids re-evaluating identical SOP+genome combinations
        self._evaluation_cache: dict[str, list[EvaluationVector]] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def _fire_hook(self, hook_name: str, ctx: HookContext, *args: Any, **kwargs: Any) -> Any:
        """Fire an evolution hook from sync context.

        Safely runs async hooks from synchronous code. If no hooks are registered,
        returns immediately with zero overhead. Errors are logged but don't propagate.

        Args:
            hook_name: Name of the hook method (e.g., "on_generation_complete").
            ctx: Hook context with correlation ID and metadata.
            *args: Arguments for the hook.
            **kwargs: Keyword arguments for the hook.

        Returns:
            Hook result, or None if no hooks registered or hook failed.
        """
        # Fast path: no hooks registered = no overhead
        if HookRegistry.get_evolution_hooks() is None:
            return None

        try:
            # Try to get existing event loop (may be running in async context)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - create task but don't await
                loop.create_task(fire_evolution_hook(hook_name, ctx, *args, **kwargs))
                return None
            except RuntimeError:
                # No running loop - create new one for sync context
                return asyncio.run(fire_evolution_hook(hook_name, ctx, *args, **kwargs))
        except Exception as e:
            logger.warning(f"Failed to fire hook {hook_name}: {e}")
            return None

    # =========================================================================
    # Checkpoint/Recovery Methods
    # =========================================================================

    def _get_checkpoint_path(self, job_id: str) -> Path:
        """Get checkpoint file path for a job"""
        if not self.checkpoint_dir:
            raise RuntimeError("Checkpointing not enabled (no checkpoint_dir)")
        return self.checkpoint_dir / f"checkpoint_{job_id}.json"

    @staticmethod
    def _load_checkpoint_data(
        checkpoint_path: Path, retry_handler: RetryHandler
    ) -> dict[str, Any]:
        """
        Load and validate checkpoint data from file

        Args:
            checkpoint_path: Path to checkpoint file
            retry_handler: Retry handler for file I/O

        Returns:
            Validated checkpoint data dictionary

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint version mismatch (warning only)
            RuntimeError: If load fails after retries
        """
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        # Load checkpoint with retry
        def _load_json():
            with checkpoint_path.open() as f:
                return json.load(f)

        checkpoint_data = retry_handler.execute_with_retry(
            _load_json,
            retry_config=RetryHandler.CONFIG_RETRY_CONFIG,
            component="Scheduler",
            operation="load_checkpoint",
        )

        # Validate checkpoint version
        checkpoint_version = checkpoint_data.get("version")
        if checkpoint_version != EvolutionScheduler.CHECKPOINT_VERSION:
            logger.warning(
                f"Checkpoint version mismatch: {checkpoint_version} != "
                f"{EvolutionScheduler.CHECKPOINT_VERSION}. Attempting to load anyway..."
            )

        return checkpoint_data

    def _restore_job_state(self, checkpoint_data: dict[str, Any]) -> None:
        """
        Restore job state from checkpoint data

        Args:
            checkpoint_data: Checkpoint data dictionary

        Raises:
            ValueError: If checkpoint is missing required data
        """
        # Restore job state
        job_data = checkpoint_data.get("job")
        if not job_data:
            raise ValueError("Checkpoint missing job data")

        self.current_job = EvolutionJob(**job_data)

        # Restore tracking data
        self.generation_history = checkpoint_data.get("generation_history", [])
        self.best_quality_history = checkpoint_data.get("best_quality_history", [])

    def save_checkpoint(self) -> Path | None:
        """
        Save current job state to checkpoint

        Returns:
            Path to checkpoint file, or None if checkpointing disabled or no active job

        Raises:
            RuntimeError: If checkpoint save fails
        """
        if not self.checkpoint_dir or not self.current_job:
            return None

        try:
            checkpoint_path = self._get_checkpoint_path(self.current_job.id)

            # Prepare checkpoint data
            checkpoint_data = {
                "version": self.CHECKPOINT_VERSION,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "job": self.current_job.model_dump(mode="json"),
                "generation_history": self.generation_history,
                "best_quality_history": self.best_quality_history,
            }

            # Atomic write
            atomic_write_json(
                checkpoint_path, checkpoint_data, self.retry_handler, "Scheduler"
            )

            logger.info(
                f"Checkpoint saved for job {self.current_job.id} "
                f"(generation {self.current_job.currentGeneration})"
            )

            return checkpoint_path

        except Exception:
            logger.exception("Failed to save checkpoint")
            # Don't crash the evolution job on checkpoint failure
            # Just log and continue
            return None

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint_path: str | Path,
        config_store: ConfigStore,
        gene_pool: GenePool,
        qd_grid: QDGridManager,
        execution_engine: ExecutionEngine,
        evaluation_service: EvaluationService,
        director_service: DirectorService,
        retry_handler: RetryHandler | None = None,
    ) -> EvolutionScheduler:
        """
        Load scheduler from checkpoint

        Creates a new EvolutionScheduler instance and restores state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            config_store: Configuration storage
            gene_pool: Gene pool (should contain genes from checkpoint)
            qd_grid: QD grid (should contain cells from checkpoint)
            execution_engine: Execution engine
            evaluation_service: Evaluation service
            director_service: Director service
            retry_handler: Retry handler for file I/O

        Returns:
            EvolutionScheduler with restored state

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is invalid or incompatible
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)
        handler = retry_handler or RetryHandler()

        try:
            # Load and validate checkpoint data
            checkpoint_data = cls._load_checkpoint_data(checkpoint_path, handler)

            # Create scheduler instance
            scheduler = cls(
                config_store=config_store,
                gene_pool=gene_pool,
                qd_grid=qd_grid,
                execution_engine=execution_engine,
                evaluation_service=evaluation_service,
                director_service=director_service,
                checkpoint_dir=str(checkpoint_path.parent),
                retry_handler=handler,
            )

            # Restore job state
            scheduler._restore_job_state(checkpoint_data)

            if scheduler.current_job is None:
                raise RuntimeError("Checkpoint restoration failed: current_job is None")
            logger.info(
                f"Checkpoint loaded: job {scheduler.current_job.id}, "
                f"generation {scheduler.current_job.currentGeneration}"
            )

            return scheduler

        except Exception as e:
            logger.exception(f"Failed to load checkpoint: {checkpoint_path}")
            raise RuntimeError(f"Checkpoint load failed: {e}") from e

    def resume_from_checkpoint(self, checkpoint_path: str | Path) -> None:
        """
        Resume current scheduler from checkpoint

        Updates the current scheduler instance with state from checkpoint.
        Use this when you have an existing scheduler and want to restore its state.

        For creating a new scheduler from checkpoint, use load_checkpoint() class method.

        Args:
            checkpoint_path: Path to checkpoint file

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
            ValueError: If checkpoint is invalid
            RuntimeError: If checkpoint loading fails
        """
        checkpoint_path = Path(checkpoint_path)

        try:
            # Load and validate checkpoint data
            checkpoint_data = self._load_checkpoint_data(checkpoint_path, self.retry_handler)

            # Restore job state
            self._restore_job_state(checkpoint_data)

            if self.current_job is None:
                raise RuntimeError("Checkpoint restoration failed: current_job is None")
            logger.info(
                f"Resumed from checkpoint: job {self.current_job.id}, "
                f"generation {self.current_job.currentGeneration}"
            )

        except Exception as e:
            logger.exception(f"Failed to resume from checkpoint: {checkpoint_path}")
            raise RuntimeError(f"Checkpoint resume failed: {e}") from e

    # =========================================================================
    # Job Lifecycle Methods
    # =========================================================================

    def start_job(self, job: EvolutionJob) -> None:
        """
        Start an evolution job

        Args:
            job: EvolutionJob configuration
        """
        self.current_job = job
        job.status = EvolutionJobStatus.RUNNING
        job.startedAt = datetime.now(timezone.utc).isoformat()
        job.currentGeneration = 0
        job.currentPhaseIndex = 0

        # Bootstrap with base SOPs
        self._bootstrap_population(job)

    def step(self) -> dict[str, Any]:
        """
        Execute one generation step.

        Orchestrates the evolution process through 5 phases:
        1. Selection: Choose parents for mutation
        2. Mutation: Generate offspring via Director service
        3. Evaluation: Execute and evaluate offspring
        4. Update: Update best SOP and statistics
        5. Checkpoint: Save state and check phase/job transitions

        Returns:
            Dictionary with generation statistics
        """
        if not self.current_job or self.current_job.status != EvolutionJobStatus.RUNNING:
            raise RuntimeError("No active evolution job")

        job = self.current_job
        generation_start = time.time()
        cost_at_start = job.budgetUsed.cost

        # Phase 1: Selection
        parents = self._run_selection_phase()
        if not parents:
            return {
                "generation": job.currentGeneration,
                "phase": job.phases[job.currentPhaseIndex].name,
                "status": "no_parents",
                "offspring_count": 0,
                "warning": "Selection returned no parents - check gene pool initialization",
            }

        # Phase 2: Mutation
        offspring, diagnoses = self._run_mutation_phase(parents)

        # Phase 3: Evaluation
        evaluated_offspring = self._run_evaluation_phase(offspring)

        # Phase 4: Update
        generation_stats = self._run_update_phase(
            parents, evaluated_offspring, diagnoses, generation_start, cost_at_start
        )

        # Phase 5: Checkpoint
        self._run_checkpoint_phase()

        return generation_stats

    def _run_selection_phase(self) -> list[tuple[str, str]]:
        """
        Select parents for mutation.

        Returns:
            List of (sopId, version) tuples for selected parents
        """
        job = self.current_job
        if job is None:
            raise RuntimeError("No active job - cannot run selection phase")
        phase = job.phases[job.currentPhaseIndex]
        parents = self._select_parents(phase.selectionStrategy, phase.parentsPerGeneration)

        if not parents:
            logger.warning(
                f"No parents selected for generation {job.currentGeneration}. "
                f"Strategy: {phase.selectionStrategy}. "
                f"Gene pool size: {len(list(self.gene_pool.list_sop_genes()))}"
            )

        return parents

    def _run_mutation_phase(
        self, parents: list[tuple[str, str]]
    ) -> tuple[list[SOPGene], list[dict[str, Any]]]:
        """
        Generate offspring through directed mutations.

        Args:
            parents: List of (sopId, version) tuples

        Returns:
            Tuple of (offspring genes, diagnosis data)
        """
        job = self.current_job
        if job is None:
            raise RuntimeError("No active job - cannot run mutation phase")
        phase = job.phases[job.currentPhaseIndex]

        offspring: list[SOPGene] = []
        diagnoses: list[dict[str, Any]] = []

        for parent_id, parent_version in parents:
            result = self._generate_offspring(
                parent_id, parent_version, phase.allowedMutationTypes, job.constraints
            )

            if result:
                child_gene, diagnosis_info = result
                offspring.append(child_gene)
                diagnoses.append(diagnosis_info)

            if self._is_budget_exceeded(job):
                break

        return offspring, diagnoses

    def _run_evaluation_phase(self, offspring: list[SOPGene]) -> list[SOPGene]:
        """
        Execute offspring and compute metrics.

        Supports parallel evaluation when self.parallel_offspring is True.
        Evaluation is parallelized, but state updates (gene_pool, qd_grid, budget)
        are done sequentially to maintain thread safety.

        Args:
            offspring: List of mutated SOP genes

        Returns:
            List of evaluated offspring with metrics populated
        """
        job = self.current_job
        if job is None:
            raise RuntimeError("No active job - cannot run evaluation phase")

        # Define evaluation function for a single child
        def evaluate_child(
            child_gene: SOPGene,
        ) -> (
            tuple[SOPGene, list[EvaluationVector], dict[str, AggregatedMetric], ProcessConfig, PromptGenome]
            | None
        ):
            """Evaluate a single child gene. Returns None if SOP/genome not found."""
            child_sop = self.config_store.get_sop(child_gene.sopId, child_gene.version)
            child_genome = self.config_store.get_prompt_genome(
                child_gene.promptGenomeId, child_gene.promptGenomeVersion
            )

            if not child_sop or not child_genome:
                return None

            # Evaluate on task set (this is the expensive I/O-bound operation)
            evaluations = self._evaluate_sop(child_sop, child_genome, job.taskSet)
            task_weights = self._extract_task_weights(job.taskSet)
            aggregated = self._aggregate_metrics(
                evaluations, job.metricsToOptimize, job.qualityScoreWeights, task_weights
            )

            return (child_gene, evaluations, aggregated, child_sop, child_genome)

        # Phase 1: Run evaluations (parallel or sequential)
        if self.parallel_offspring and len(offspring) > 1:
            logger.info(f"Evaluating {len(offspring)} offspring in parallel")
            with ThreadPoolExecutor(max_workers=len(offspring)) as executor:
                results = list(executor.map(evaluate_child, offspring))
        else:
            results = [evaluate_child(child) for child in offspring]

        # Phase 2: Sequential state updates (thread-safe)
        evaluated_offspring: list[SOPGene] = []
        for result in results:
            if result is None:
                continue

            child_gene, evaluations, aggregated, child_sop, child_genome = result

            # Update gene
            child_gene.evaluations = evaluations
            child_gene.aggregatedMetrics = aggregated

            # Add to gene pool and QD grid
            self.gene_pool.add_sop_gene(child_gene, generation=job.currentGeneration)
            self.qd_grid.add_sop(child_gene, child_sop, child_genome)

            evaluated_offspring.append(child_gene)

            # Update budget
            job.budgetUsed.evaluations += len(evaluations)
            job.budgetUsed.llmCalls += len(evaluations) * len(child_sop.roles)

            if self._is_budget_exceeded(job):
                break

        return evaluated_offspring

    def _run_update_phase(
        self,
        parents: list[tuple[str, str]],
        evaluated_offspring: list[SOPGene],
        diagnoses: list[dict[str, Any]],
        generation_start: float,
        cost_at_start: float,
    ) -> dict[str, Any]:
        """
        Update evolutionary state and statistics.

        Args:
            parents: Selected parent tuples
            evaluated_offspring: Evaluated offspring genes
            diagnoses: Diagnosis data from mutations
            generation_start: Timestamp when generation started
            cost_at_start: Budget cost at generation start

        Returns:
            Dictionary of generation statistics
        """
        job = self.current_job
        if job is None:
            raise RuntimeError("No active job - cannot run update phase")
        phase = job.phases[job.currentPhaseIndex]

        # Update best SOP
        self._update_best_sop(job, evaluated_offspring)

        # Update Pareto frontier
        self.gene_pool.update_pareto_frontier(
            metric_ids=job.metricsToOptimize,
            maximize_all=True,
        )

        # Increment generation
        job.currentGeneration += 1
        self.gene_pool.increment_generation()

        # Compile statistics
        individual_metrics = {}
        if job.bestSopSoFar and "metrics" in job.bestSopSoFar:
            individual_metrics = job.bestSopSoFar["metrics"]

        generation_cost = job.budgetUsed.cost - cost_at_start

        best_quality: float = job.bestSopSoFar.get("quality", 0.0) if job.bestSopSoFar else 0.0

        # Collect per-offspring details for debugging selection dynamics
        offspring_details: list[dict[str, Any]] = []
        for gene in evaluated_offspring:
            quality = gene.get_metric_mean("weighted_aggregate")
            # Extract parent ID from parent dict if available
            parent_id = ""
            if gene.parent:
                parent_id = gene.parent.get("sopId", "")
            offspring_details.append({
                "offspring_id": str(gene.sopId),
                "parent_id": parent_id,
                "quality": float(quality),
                "is_selected_as_best": (
                    job.bestSopSoFar is not None
                    and job.bestSopSoFar.get("sopId") == gene.sopId
                ),
                "metrics": {
                    metric_id: float(agg.mean)
                    for metric_id, agg in gene.aggregatedMetrics.items()
                },
            })

        generation_stats = {
            "generation": job.currentGeneration,
            "phase": phase.name,
            "phase_index": job.currentPhaseIndex,
            "parents_count": len(parents),
            "offspring_count": len(evaluated_offspring),
            "duration_seconds": time.time() - generation_start,
            "budget_used": job.budgetUsed.model_dump(),
            "generation_cost": generation_cost,
            "best_quality": best_quality,
            "individual_metrics": individual_metrics,
            "offspring_details": offspring_details,
            "pareto_count": self.gene_pool.stats.get("pareto_optimal_count", 0),
            "qd_coverage": self.qd_grid.stats.get("coverage", 0.0),
            "diagnoses": diagnoses,
        }

        self.generation_history.append(generation_stats)
        self.best_quality_history.append(best_quality)  # Track for convergence detection

        # Fire generation complete hook
        hook_ctx = HookContext(
            correlation_id=str(uuid.uuid4()),
            metadata={
                "job_id": str(job.id),
                "phase": phase.name,
            },
        )
        self._fire_hook(
            "on_generation_complete",
            hook_ctx,
            job.currentGeneration,
            len(parents) + len(evaluated_offspring),  # population size
            best_quality,
        )

        return generation_stats

    def _run_checkpoint_phase(self) -> None:
        """
        Save checkpoint and handle phase/job transitions.
        """
        job = self.current_job
        if job is None:
            raise RuntimeError("No active job - cannot run checkpoint phase")
        phase = job.phases[job.currentPhaseIndex]

        # Save periodic checkpoint
        if self.checkpoint_dir and job.currentGeneration % self.checkpoint_interval == 0:
            try:
                self.save_checkpoint()
            except Exception as e:
                logger.warning(f"Checkpoint save failed (non-fatal): {e}")

        # Check phase completion
        if job.currentGeneration >= phase.maxGenerations:
            self._advance_phase(job)

        # Check job completion
        if self._should_stop(job):
            self._complete_job(job)
            if self.checkpoint_dir:
                try:
                    self.save_checkpoint()
                except Exception as e:
                    logger.warning(f"Final checkpoint save failed (non-fatal): {e}")

    def _bootstrap_population(self, job: EvolutionJob) -> None:
        """Bootstrap population with base SOPs"""

        base_sops: list[tuple[ProcessConfig, PromptGenome]] = []

        for base_sop_ref in job.baseSops:
            sop = self.config_store.get_sop(base_sop_ref["sopId"], base_sop_ref.get("sopVersion"))
            genome = self.config_store.get_prompt_genome(
                base_sop_ref["promptGenomeId"], base_sop_ref.get("promptGenomeVersion")
            )

            if not sop or not genome:
                continue

            # Evaluate base SOP
            evaluations = self._evaluate_sop(sop, genome, job.taskSet)

            # Extract task weights from job.taskSet
            task_weights = self._extract_task_weights(job.taskSet)

            # Create gene
            aggregated = self._aggregate_metrics(
                evaluations, job.metricsToOptimize, job.qualityScoreWeights, task_weights
            )

            gene = SOPGene(
                sopId=sop.id,
                version=sop.version,
                promptGenomeId=genome.id,
                promptGenomeVersion=genome.version,
                configSnapshot=sop,
                evaluations=evaluations,
                aggregatedMetrics=aggregated,
            )

            # Add to gene pool and QD grid
            self.gene_pool.add_sop_gene(gene)
            self.qd_grid.add_sop(gene, sop, genome)

            base_sops.append((sop, genome))

            # Update best SOP with baseline (fix: was never being set)
            baseline_quality = gene.get_metric_mean("weighted_aggregate")
            if not job.bestSopSoFar or baseline_quality > job.bestSopSoFar.get(
                "quality", 0.0
            ):
                metrics_dict = {
                    metric_id: agg.mean
                    for metric_id, agg in gene.aggregatedMetrics.items()
                }
                job.bestSopSoFar = {
                    "sopId": gene.sopId,
                    "version": gene.version,
                    "quality": baseline_quality,
                    "metrics": metrics_dict,
                    "generation": 0,  # Baseline is generation 0
                    # Include genome IDs for reliable prompt diff tracking
                    "promptGenomeId": gene.promptGenomeId,
                    "promptGenomeVersion": gene.promptGenomeVersion,
                }

        # Bootstrap QD grid
        if base_sops:
            self.qd_grid.bootstrap(base_sops)

        # Initialize Pareto frontier with baseline evaluations
        self.gene_pool.update_pareto_frontier(
            metric_ids=job.metricsToOptimize,
            maximize_all=True,
        )

    def _select_parents(self, strategy: SelectionStrategy, count: int) -> list[tuple[str, str]]:
        """
        Select parents for mutation using configured selection strategy

        Args:
            strategy: Selection strategy enum
            count: Number of parents to select

        Returns:
            List of (sopId, version) tuples
        """
        # Get strategy config from current phase (if available)
        strategy_config = {}
        if self.current_job:
            phase = self.current_job.phases[self.current_job.currentPhaseIndex]
            config = phase.selectionStrategyConfig

            if config and config.strategyType == strategy:
                # Use detailed config
                if (
                    strategy == SelectionStrategy.QD_CURIOSITY
                    and config.qdCuriosityConfig
                ):
                    strategy_config = config.qdCuriosityConfig.model_dump()
                elif strategy == SelectionStrategy.RECENT and config.recentConfig:
                    strategy_config = config.recentConfig.model_dump()
                elif strategy == SelectionStrategy.HYBRID and config.hybridConfig:
                    strategy_config = config.hybridConfig.model_dump()
                elif config.genericConfig:
                    strategy_config = config.genericConfig
                else:
                    # Use default config
                    strategy_config = {}
            else:
                # Use default config
                strategy_config = {}
        else:
            strategy_config = {}

        # Create strategy instance
        try:
            selector = self.selection_factory.create(strategy, strategy_config)
        except (ValueError, KeyError, TypeError) as e:
            # Catch config-related errors: invalid values, missing keys, wrong types
            logger.warning(f"Failed to create selection strategy {strategy}: {e}. Falling back to QD_QUALITY_WEIGHTED")
            selector = self.selection_factory.create(
                SelectionStrategy.QD_QUALITY_WEIGHTED, {}
            )

        # Execute selection
        try:
            parents = selector.select(self.gene_pool, self.qd_grid, count)

            logger.info(f"Selected {len(parents)} parents using {strategy} strategy")

            return parents

        except (ValueError, RuntimeError, IndexError) as e:
            # Catch selection-specific errors: invalid state, runtime issues, empty pools
            logger.warning(f"Selection strategy {strategy} failed: {e}. Returning empty parent list")
            return []

    def _generate_offspring(
        self,
        parent_id: str,
        parent_version: str,
        mutation_types: list[MutationType],
        constraints: Any,
    ) -> tuple[SOPGene, dict[str, Any]] | None:
        """
        Generate offspring via mutation.

        Returns:
            Tuple of (offspring_gene, diagnosis_info) or None if generation failed.
            diagnosis_info contains:
                - primary_weakness: Director's identified weakness
                - recommendations: List of improvement suggestions
                - mutation_type: Type of mutation applied
                - mutation_rationale: Director's reasoning for the mutation
        """

        # Get parent
        parent_gene = self.gene_pool.get_sop_gene(parent_id, parent_version)
        if not parent_gene:
            # Get available genes for debugging (first 5)
            available_genes = list(self.gene_pool.list_sop_genes())[:5]
            logger.error(
                f"Parent gene not found: {parent_id}@{parent_version}. "
                f"Available genes (first 5): {available_genes}"
            )
            return None

        parent_sop = self.config_store.get_sop(parent_id, parent_version)
        if not parent_sop:
            logger.error(
                f"Parent SOP not found in config store: {parent_id}@{parent_version}"
            )
            return None

        parent_genome = self.config_store.get_prompt_genome(
            parent_gene.promptGenomeId, parent_gene.promptGenomeVersion
        )
        if not parent_genome:
            logger.error(
                f"Parent genome not found: {parent_gene.promptGenomeId}@{parent_gene.promptGenomeVersion}"
            )
            return None

        # Use Director to propose mutation
        diagnosis, mutation = self.director_service.propose_improvements(
            parent_gene,
            parent_sop,
            parent_genome,
            metrics_to_optimize=self.current_job.metricsToOptimize if self.current_job else [],
            mutation_types=mutation_types,
            constraints=constraints.model_dump() if constraints else None,
        )

        # Capture diagnosis info for reporting
        diagnosis_info: dict[str, Any] = {
            "primary_weakness": diagnosis.primaryWeakness,
            "recommendations": diagnosis.recommendations[:3] if diagnosis.recommendations else [],
            "root_cause": diagnosis.rootCauseAnalysis,
            "mutation_type": mutation.mutationType.value,
            "mutation_rationale": mutation.rationale,
        }

        # Save new configs
        self.config_store.save_sop(mutation.newConfig)
        if mutation.newPromptGenome:
            self.config_store.save_prompt_genome(mutation.newPromptGenome)

        # Create offspring gene
        offspring = SOPGene(
            sopId=mutation.newConfig.id,
            version=mutation.newConfig.version,
            parent={"sopId": parent_id, "version": parent_version},
            promptGenomeId=mutation.newPromptGenome.id
            if mutation.newPromptGenome
            else parent_genome.id,
            promptGenomeVersion=mutation.newPromptGenome.version
            if mutation.newPromptGenome
            else parent_genome.version,
            configSnapshot=mutation.newConfig,
            evaluations=[],  # Will be filled by evaluation
            aggregatedMetrics={},
        )

        # Compute detailed changelog for reporting
        changelog = self._compute_mutation_changelog(
            parent_sop=parent_sop,
            parent_genome=parent_genome,
            new_sop=mutation.newConfig,
            new_genome=mutation.newPromptGenome,
            mutation_type=mutation.mutationType,
        )
        diagnosis_info["changelog"] = changelog

        return offspring, diagnosis_info

    def _compute_mutation_changelog(
        self,
        parent_sop: ProcessConfig,
        parent_genome: PromptGenome,
        new_sop: ProcessConfig,
        new_genome: PromptGenome | None,
        mutation_type: MutationType,
    ) -> dict[str, Any]:
        """
        Compute detailed changelog of what changed between parent and offspring.

        Args:
            parent_sop: Parent SOP configuration
            parent_genome: Parent prompt genome
            new_sop: New SOP configuration after mutation
            new_genome: New prompt genome after mutation (may be None)
            mutation_type: Type of mutation applied

        Returns:
            Dictionary containing detailed changes:
            - mutation_type: Type of mutation applied
            - changes: Dict with prompts, roles_added, roles_removed, tools_changed, graph changes
            - sop_snapshot: Current state of roles and graph
        """
        changelog: dict[str, Any] = {
            "mutation_type": mutation_type.value,
            "parent_version": parent_sop.version,
            "new_version": new_sop.version,
            "changes": {
                "prompts": [],
                "roles_added": [],
                "roles_removed": [],
                "tools_changed": [],
                "graph_edges_added": [],
                "graph_edges_removed": [],
                "params_changed": [],
            },
            "sop_snapshot": {
                "roles": [
                    {
                        "id": r.id,
                        "model": r.model,
                        "tools": r.tools or [],
                        "promptRef": r.promptRef,
                    }
                    for r in new_sop.roles
                ],
                "graph": [
                    {"from": e.from_, "to": e.to, "condition": e.condition}
                    for e in new_sop.graph
                ],
            },
        }

        # Compute prompt diffs if genome changed
        if new_genome and parent_genome:
            for role in new_sop.roles:
                prompt_ref = role.promptRef
                if not prompt_ref:
                    continue

                old_prompt_data = parent_genome.rolePrompts.get(prompt_ref)
                new_prompt_data = new_genome.rolePrompts.get(prompt_ref)

                if old_prompt_data and new_prompt_data:
                    old_content = old_prompt_data.content if hasattr(old_prompt_data, "content") else str(old_prompt_data)
                    new_content = new_prompt_data.content if hasattr(new_prompt_data, "content") else str(new_prompt_data)

                    if old_content != new_content:
                        diff = compute_prompt_diff(old_content, new_content)
                        changelog["changes"]["prompts"].append({
                            "role_id": role.id,
                            "prompt_ref": prompt_ref,
                            "diff": diff,
                        })

        # Detect role changes
        old_role_ids = {r.id for r in parent_sop.roles}
        new_role_ids = {r.id for r in new_sop.roles}

        # Roles added
        for role_id in new_role_ids - old_role_ids:
            new_role = next((r for r in new_sop.roles if r.id == role_id), None)
            if new_role:
                changelog["changes"]["roles_added"].append({
                    "id": role_id,
                    "model": new_role.model,
                    "tools": new_role.tools or [],
                    "promptRef": new_role.promptRef,
                })

        # Roles removed
        for role_id in old_role_ids - new_role_ids:
            changelog["changes"]["roles_removed"].append(role_id)

        # Detect tool changes on existing roles
        for new_role in new_sop.roles:
            old_role = next((r for r in parent_sop.roles if r.id == new_role.id), None)
            if old_role:
                old_tools = set(old_role.tools or [])
                new_tools = set(new_role.tools or [])
                if old_tools != new_tools:
                    changelog["changes"]["tools_changed"].append({
                        "role_id": new_role.id,
                        "old_tools": list(old_tools),
                        "new_tools": list(new_tools),
                    })

                # Detect model changes
                if old_role.model != new_role.model:
                    changelog["changes"]["params_changed"].append({
                        "role_id": new_role.id,
                        "param": "model",
                        "old_value": old_role.model,
                        "new_value": new_role.model,
                    })

        # Detect graph edge changes
        # Normalize from_ field which can be str or list[str]
        def normalize_from(from_val: str | list[str]) -> str:
            if isinstance(from_val, list):
                return ",".join(sorted(from_val))
            return from_val

        old_edges = {(normalize_from(e.from_), e.to) for e in parent_sop.graph}
        new_edges = {(normalize_from(e.from_), e.to) for e in new_sop.graph}

        for from_node, to_node in new_edges - old_edges:
            changelog["changes"]["graph_edges_added"].append({
                "from": from_node,
                "to": to_node,
            })

        for from_node, to_node in old_edges - new_edges:
            changelog["changes"]["graph_edges_removed"].append({
                "from": from_node,
                "to": to_node,
            })

        # Add ASCII graph visualization
        changelog["graph_ascii"] = generate_ascii_graph(
            changelog["sop_snapshot"]["graph"],
            changelog["sop_snapshot"]["roles"],
        )

        return changelog

    def _get_sop_cache_key(
        self, genome: PromptGenome, task_set: TaskSet
    ) -> str:
        """Generate a cache key for SOP evaluation.

        The key is based on:
        1. All prompt texts from the genome
        2. Task IDs in the task set

        This allows caching evaluations for identical genome+task combinations.

        Args:
            genome: The prompt genome containing all role prompts
            task_set: The task set being evaluated

        Returns:
            MD5 hash string to use as cache key
        """
        # Collect all prompts from genome in sorted order (for determinism)
        # PromptGenome has rolePrompts: dict[str, RolePrompt] where RolePrompt has content
        prompt_parts: list[str] = []
        for role_id in sorted(genome.rolePrompts.keys()):
            role_prompt = genome.rolePrompts[role_id]
            prompt_parts.append(f"{role_id}:{role_prompt.content}")

        # Add task IDs (in order)
        task_ids: list[str] = [task.id for task in task_set.tasks]
        prompt_parts.extend(task_ids)

        # Create hash
        content = "\n".join(prompt_parts)
        return hashlib.md5(content.encode()).hexdigest()

    def _evaluate_sop(
        self, sop: ProcessConfig, genome: PromptGenome, task_set: TaskSet
    ) -> list[EvaluationVector]:
        """Evaluate SOP on task set with caching.

        Uses a cache to avoid re-evaluating identical genome+task combinations.
        This provides significant speedup when the same prompts are evaluated
        multiple times (e.g., parent SOPs in subsequent generations).

        Args:
            sop: The SOP configuration
            genome: The prompt genome
            task_set: The task set to evaluate on

        Returns:
            List of evaluation vectors, one per task
        """
        # Check cache
        cache_key = self._get_sop_cache_key(genome, task_set)
        if cache_key in self._evaluation_cache:
            self._cache_hits += 1
            logger.debug(
                f"Evaluation cache hit (hits={self._cache_hits}, "
                f"misses={self._cache_misses})"
            )
            return self._evaluation_cache[cache_key]

        self._cache_misses += 1
        logger.debug(
            f"Evaluation cache miss (hits={self._cache_hits}, "
            f"misses={self._cache_misses})"
        )

        evaluations: list[EvaluationVector] = []

        for task in task_set.tasks:
            # Execute
            trace = self.execution_engine.execute(sop, genome, task.input)

            # Track cost from execution trace
            if self.current_job and hasattr(trace, "total_cost"):
                self.current_job.budgetUsed.cost += trace.total_cost

            # Get metrics
            if self.current_job:
                metric_configs = [
                    self.config_store.get_metric(m_id)
                    for m_id in self.current_job.metricsToOptimize
                ]
                metric_configs = [m for m in metric_configs if m is not None]
            else:
                metric_configs = []

            # Evaluate
            evaluation = self.evaluation_service.evaluate(
                trace,
                metric_configs,
                task_data={"input": task.input, "groundTruth": task.groundTruth},
                prompt_genome_id=genome.id,
                prompt_genome_version=genome.version,
            )

            evaluations.append(evaluation)

        # Store in cache
        self._evaluation_cache[cache_key] = evaluations

        return evaluations

    def _extract_task_weights(
        self, task_set: TaskSet | None
    ) -> list[float] | None:
        """
        Extract and normalize task weights from a TaskSet.

        Args:
            task_set: TaskSet containing tasks with optional weights

        Returns:
            Normalized weights (sum to 1.0), or None if no task set or all weights are equal
        """
        if not task_set or not task_set.tasks:
            return None

        raw_weights = [task.weight for task in task_set.tasks]

        # Check if all weights are equal (default case) - no need to apply weighting
        if all(w == raw_weights[0] for w in raw_weights):
            return None

        # Normalize to sum to 1.0
        total = sum(raw_weights)
        if total <= 0:
            return None

        return [w / total for w in raw_weights]

    def _aggregate_metrics(
        self,
        evaluations: list[EvaluationVector],
        metric_ids: list[str],
        weights: dict[str, float],
        task_weights: list[float] | None = None,
    ) -> dict[str, AggregatedMetric]:
        """
        Aggregate metrics across evaluations with statistical rigor

        Args:
            evaluations: List of evaluation vectors
            metric_ids: Metrics to aggregate
            weights: Weights for quality score calculation
            task_weights: Optional weights for each task (must sum to 1.0)

        Returns:
            Dictionary of metric_id -> AggregatedMetric with confidence intervals
        """

        # Step 1: Aggregate each metric statistically
        aggregated = self.evaluation_service.aggregate_all_metrics_statistical(
            evaluations,
            compute_confidence_interval=True,
            detect_outliers=True,
        )

        # Step 1.5: Apply task weights if provided (re-aggregate with task importance)
        if task_weights and len(task_weights) == len(evaluations):
            aggregated = apply_task_weights(evaluations, aggregated, task_weights)

        # Step 2: Compute weighted quality score with propagated uncertainty
        if weights:
            weighted_metric = self._compute_weighted_quality_score(aggregated, weights, metric_ids)
            aggregated["weighted_aggregate"] = weighted_metric

        return aggregated

    def _compute_weighted_quality_score(
        self,
        aggregated: dict[str, AggregatedMetric],
        weights: dict[str, float],
        metric_ids: list[str],
    ) -> AggregatedMetric:
        """
        Compute weighted quality score with propagated uncertainty

        Uses variance propagation: If Z = w_x*X + w_y*Y, then
        Var(Z) = w_x²*Var(X) + w_y²*Var(Y) (assuming independence)

        Args:
            aggregated: Aggregated metrics
            weights: Metric weights
            metric_ids: Metric IDs to include

        Returns:
            AggregatedMetric for weighted aggregate
        """

        from siare.utils.statistics import (
            bootstrap_confidence_interval,  # type: ignore[reportUnknownVariableType]
        )

        weighted_mean = 0.0
        weighted_variance = 0.0
        total_samples = 0
        raw_values: list[float] = []

        # First pass: collect raw values per metric and compute weighted mean/variance
        metric_raw_values: dict[str, list[float]] = {}

        for metric_id in metric_ids:
            if metric_id not in aggregated:
                continue

            agg = aggregated[metric_id]
            weight = weights.get(metric_id, 0.0)

            # Weighted mean
            weighted_mean += weight * agg.mean

            # Propagate variance (assumes independence)
            if agg.standardDeviation is not None:
                weighted_variance += (weight**2) * (agg.standardDeviation**2)

            total_samples = max(total_samples, agg.sampleSize)

            # Collect raw values per metric (for per-sample weighted aggregate)
            if agg.rawValues:
                metric_raw_values[metric_id] = agg.rawValues

        # Compute per-sample weighted aggregates for bootstrap CI
        # This ensures CI is computed on actual weighted aggregate samples,
        # not on individual weighted contributions from each metric
        if metric_raw_values:
            n_samples = min(len(v) for v in metric_raw_values.values())
            for i in range(n_samples):
                sample_weighted = sum(
                    weights.get(m_id, 0.0) * metric_raw_values[m_id][i]
                    for m_id in metric_raw_values
                )
                raw_values.append(sample_weighted)

        weighted_std = weighted_variance**0.5 if weighted_variance > 0 else 0.0
        weighted_se = weighted_std / (total_samples**0.5) if total_samples > 0 else 0.0

        # Bootstrap CI for weighted aggregate
        ci: tuple[float, float] | None = None
        if raw_values and len(raw_values) >= MIN_PARENTS_FOR_CROSSOVER:
            try:
                ci = bootstrap_confidence_interval(raw_values, confidence_level=0.95)
            except Exception as e:
                logger.warning(f"Failed to compute CI for weighted_aggregate: {e}")

        return AggregatedMetric(
            metricId="weighted_aggregate",
            mean=weighted_mean,
            median=weighted_mean,  # Approximation
            confidenceInterval=ci,
            standardDeviation=weighted_std,
            standardError=weighted_se,
            sampleSize=total_samples,
            aggregationMethod=AggregationMethod.WEIGHTED,
            rawValues=raw_values if raw_values else None,
        )

    def _update_best_sop(self, job: EvolutionJob, offspring: list[SOPGene]) -> None:
        """Update best SOP so far.

        Selection criteria:
        1. Higher quality always wins
        2. On quality tie (within quality_tie_threshold), prefer the more recent generation
        3. On quality tie within same generation, prefer newer version
        """
        threshold = self.convergence_config.quality_tie_threshold

        for gene in offspring:
            # Extract mean quality score
            quality = gene.get_metric_mean("weighted_aggregate")

            should_update = False
            if not job.bestSopSoFar:
                should_update = True
            elif quality > job.bestSopSoFar.get("quality", 0.0) + threshold:
                # Strictly better quality
                should_update = True
            elif abs(quality - job.bestSopSoFar.get("quality", 0.0)) <= threshold:
                # Quality tie - prefer more recent generation or newer version
                current_gen = job.bestSopSoFar.get("generation", 0)
                current_version = job.bestSopSoFar.get("version", "0.0.0")
                if job.currentGeneration > current_gen or (job.currentGeneration == current_gen and gene.version > current_version):
                    should_update = True

            if should_update:
                # Store mean values for backward compatibility
                metrics_dict = {
                    metric_id: agg.mean for metric_id, agg in gene.aggregatedMetrics.items()
                }
                job.bestSopSoFar = {
                    "sopId": gene.sopId,
                    "version": gene.version,
                    "quality": quality,
                    "metrics": metrics_dict,
                    "generation": job.currentGeneration,
                    # Include genome IDs for reliable prompt diff tracking
                    "promptGenomeId": gene.promptGenomeId,
                    "promptGenomeVersion": gene.promptGenomeVersion,
                }

    def _is_budget_exceeded(self, job: EvolutionJob) -> bool:
        """Check if budget is exceeded"""

        if not job.constraints or not job.constraints.budgetLimit:
            return False

        limit = job.constraints.budgetLimit

        if limit.maxEvaluations and job.budgetUsed.evaluations >= limit.maxEvaluations:
            return True

        if limit.maxLLMCalls and job.budgetUsed.llmCalls >= limit.maxLLMCalls:
            return True

        return bool(limit.maxCost and job.budgetUsed.cost >= limit.maxCost)

    def _should_stop(self, job: EvolutionJob) -> bool:
        """Check if evolution should stop"""

        # Check budget
        if self._is_budget_exceeded(job):
            return True

        # Check if all phases complete (phase index beyond list)
        if job.currentPhaseIndex >= len(job.phases):
            return True

        # Check if current phase has reached max generations
        # This handles single-phase jobs where _advance_phase doesn't increment
        phase = job.phases[job.currentPhaseIndex]
        is_last_phase = job.currentPhaseIndex == len(job.phases) - 1
        if is_last_phase and job.currentGeneration >= phase.maxGenerations:
            return True

        # Check convergence using configured thresholds
        conv = self.convergence_config
        if len(self.best_quality_history) > conv.convergence_window:
            recent = self.best_quality_history[-conv.convergence_window:]
            if max(recent) - min(recent) < conv.convergence_threshold:  # Very little improvement
                return True

        return False

    def _advance_phase(self, job: EvolutionJob) -> None:
        """Advance to next phase"""

        if job.currentPhaseIndex < len(job.phases) - 1:
            job.currentPhaseIndex += 1
            logger.info(
                f"Advancing to phase {job.currentPhaseIndex + 1}: {job.phases[job.currentPhaseIndex].name}"
            )

    def _complete_job(self, job: EvolutionJob) -> None:
        """Mark job as complete"""

        job.status = EvolutionJobStatus.COMPLETED
        job.completedAt = datetime.now(timezone.utc).isoformat()

        logger.info(f"Evolution job {job.id} completed!")
        logger.info(f"Generations: {job.currentGeneration}")
        logger.info(
            f"Best quality: {job.bestSopSoFar.get('quality') if job.bestSopSoFar else 0.0:.3f}"
        )
        logger.info(f"Budget used: {job.budgetUsed.model_dump()}")

        # Log evaluation cache statistics
        total_lookups = self._cache_hits + self._cache_misses
        if total_lookups > 0:
            hit_rate = self._cache_hits / total_lookups * 100
            logger.info(
                f"Evaluation cache: {self._cache_hits} hits, {self._cache_misses} misses "
                f"({hit_rate:.1f}% hit rate)"
            )

    def run_to_completion(
        self,
        job: EvolutionJob,
        verbose: bool = True,
        on_generation_complete: Callable[[int, dict[str, Any]], None] | None = None,
    ) -> EvolutionJob:
        """
        Run evolution job to completion

        Args:
            job: EvolutionJob to run
            verbose: Print progress
            on_generation_complete: Optional callback called after each generation
                with (generation_number, stats_dict)

        Returns:
            Completed EvolutionJob
        """
        self.start_job(job)

        while job.status == EvolutionJobStatus.RUNNING:
            stats = self.step()

            if verbose:
                logger.info(
                    f"Gen {stats['generation']}: "
                    f"Offspring={stats['offspring_count']}, "
                    f"Best={stats['best_quality']:.3f}, "
                    f"Coverage={stats['qd_coverage']:.2%}"
                )

            # Call checkpoint callback after each generation
            if on_generation_complete is not None:
                on_generation_complete(stats["generation"], stats)

        return job

    def get_statistics(self) -> dict[str, Any]:
        """Get evolution statistics"""

        if not self.current_job:
            return {}

        return {
            "job_id": self.current_job.id,
            "status": self.current_job.status.value,
            "current_generation": self.current_job.currentGeneration,
            "current_phase": self.current_job.currentPhaseIndex,
            "budget_used": self.current_job.budgetUsed.model_dump(),
            "best_sop": self.current_job.bestSopSoFar,
            "gene_pool_stats": self.gene_pool.get_stats(),
            "qd_grid_stats": self.qd_grid.get_stats(),
            "generation_history": self.generation_history[-10:],  # Last 10 generations
        }
