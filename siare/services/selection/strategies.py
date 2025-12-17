"""Selection strategies for evolution

This module implements various selection strategies for choosing parent SOPs
during evolution. Each strategy balances exploration vs exploitation differently.

Available Strategies:
- ParetoSelectionStrategy: Select from Pareto frontier
- QDUniformSelectionStrategy: Uniform sampling from QD grid
- QDQualityWeightedStrategy: Quality-weighted sampling from QD grid
- QDCuriositySelectionStrategy: UCB-based curiosity-driven selection (NEW)
- RecentSelectionStrategy: Momentum-based selection from recent generations (NEW)
- TournamentSelectionStrategy: Tournament selection
- HybridSelectionStrategy: Composite mixing of multiple strategies (NEW)
"""

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from siare.core.models import (
    HybridSelectionConfig,
    QDCuriosityConfig,
    RecentSelectionConfig,
    SOPGene,
)
from siare.utils.sampling import quality_weighted_sample

if TYPE_CHECKING:
    from siare.services.gene_pool import GenePool
    from siare.services.qd_grid import QDGridManager
    from siare.services.selection.factory import SelectionStrategyFactory


logger = logging.getLogger(__name__)


# ============================================================================
# Base Strategy Class
# ============================================================================


class BaseSelectionStrategy(ABC):
    """Abstract base class for selection strategies"""

    @abstractmethod
    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        """
        Select parents for mutation

        Args:
            gene_pool: Gene pool containing SOP population
            qd_grid: Quality-Diversity grid for diversity tracking
            count: Number of parents to select

        Returns:
            List of (sopId, version) tuples
        """

    def _ensure_count(
        self,
        selected: list[tuple[str, str]],
        target_count: int,
        gene_pool: "GenePool",
    ) -> list[tuple[str, str]]:
        """
        Ensure selection has exactly target_count items

        If selection is short, fills remaining with quality-weighted sampling
        from all genes (excluding already selected).

        Args:
            selected: Current selection
            target_count: Desired count
            gene_pool: Gene pool for fallback sampling

        Returns:
            Selection with exactly target_count items (or as many as possible)
        """
        if len(selected) >= target_count:
            return selected[:target_count]

        remaining = target_count - len(selected)
        exclude_ids = set(selected)

        # Get all genes not already selected
        all_genes = gene_pool.list_sop_genes()
        available = [g for g in all_genes if (g.sopId, g.version) not in exclude_ids]

        if not available:
            logger.warning("Cannot fill selection: no genes available")
            return selected

        # Use quality-weighted sampling to fill remaining slots
        filler = self._quality_weighted_sample(available, remaining)

        return selected + filler

    def _quality_weighted_sample(
        self,
        genes: list[SOPGene],
        count: int,
    ) -> list[tuple[str, str]]:
        """
        Sample genes using quality-weighted softmax probabilities

        This method uses quality-weighted sampling with softmax normalization
        to select high-quality genes probabilistically.

        Args:
            genes: List of SOPGene objects to sample from
            count: Number of genes to sample

        Returns:
            List of (sopId, version) tuples for selected genes

        Note:
            Returns empty list if genes is empty
            Returns fewer than count if len(genes) < count
        """

        def get_quality(gene: SOPGene) -> float:
            metric = gene.aggregatedMetrics.get("weighted_aggregate")
            return metric.mean if metric is not None else 0.0

        selected = quality_weighted_sample(genes, count, get_quality)
        return [(g.sopId, g.version) for g in selected]


# ============================================================================
# Existing Strategy Implementations (Refactored)
# ============================================================================


class ParetoSelectionStrategy(BaseSelectionStrategy):
    """Select from Pareto frontier

    Exploitation-focused strategy that selects from the Pareto-optimal set,
    ensuring all parents are non-dominated across all metrics.
    """

    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        pareto_genes = gene_pool.list_sop_genes(pareto_optimal_only=True)

        if not pareto_genes:
            logger.warning("No Pareto optimal genes found")
            return []

        # Random sampling from frontier
        n: int = min(count, len(pareto_genes))
        indices: NDArray[np.intp] = np.random.choice(len(pareto_genes), size=n, replace=False)
        selected: list[SOPGene] = [pareto_genes[i] for i in indices]

        result: list[tuple[str, str]] = [(g.sopId, g.version) for g in selected]

        # Fill remaining if needed
        return self._ensure_count(result, count, gene_pool)


class QDUniformSelectionStrategy(BaseSelectionStrategy):
    """Uniform sampling from QD grid cells

    Exploration-focused strategy that samples uniformly across QD grid,
    ensuring diversity but ignoring quality.
    """

    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        result = qd_grid.sample_for_evolution("uniform", count)
        return self._ensure_count(result, count, gene_pool)


class QDQualityWeightedStrategy(BaseSelectionStrategy):
    """Quality-weighted sampling from QD grid

    Balanced strategy that samples from QD grid with probability proportional
    to quality, maintaining diversity while favoring high-quality solutions.
    """

    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        result = qd_grid.sample_for_evolution("quality_weighted", count)
        return self._ensure_count(result, count, gene_pool)


class TournamentSelectionStrategy(BaseSelectionStrategy):
    """Tournament selection

    Standard genetic algorithm selection: sample K candidates, select winner
    (highest quality). Provides tunable selection pressure via tournament size.
    """

    def __init__(self, tournament_size: int = 3):
        """
        Initialize tournament selection

        Args:
            tournament_size: Number of candidates per tournament (default: 3)
        """
        self.tournament_size = tournament_size

    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        all_genes = gene_pool.list_sop_genes()

        if not all_genes:
            return []

        selected: list[tuple[str, str]] = []
        for _ in range(count):
            # Sample tournament
            tournament_indices: NDArray[np.intp] = np.random.choice(
                len(all_genes),
                size=min(self.tournament_size, len(all_genes)),
                replace=False,
            )
            tournament: list[SOPGene] = [all_genes[i] for i in tournament_indices]

            # Select winner (highest quality)
            def get_tournament_quality(gene: SOPGene) -> float:
                metric = gene.aggregatedMetrics.get("weighted_aggregate")
                if metric is not None:
                    return metric.mean
                return 0.0

            winner: SOPGene = max(tournament, key=get_tournament_quality)

            selected.append((winner.sopId, winner.version))

        return selected


# ============================================================================
# NEW: QD_CURIOSITY Strategy
# ============================================================================


class QDCuriositySelectionStrategy(BaseSelectionStrategy):
    """Curiosity-driven selection using UCB (Upper Confidence Bound)

    Balances exploration (visiting new cells) with exploitation (high-quality cells)
    using multi-armed bandit algorithm.

    UCB formula: score(cell) = quality + C * sqrt(log(total_visits) / cell_visits)
    - High quality, high visits → low score (exploitation penalty)
    - High quality, low visits → high score (promising but under-explored)
    - Low quality, no visits → high score (novelty bonus)
    - Low quality, many visits → low score (known bad region)
    """

    def __init__(self, config: QDCuriosityConfig | None = None):
        """
        Initialize QD curiosity selection

        Args:
            config: Configuration for UCB parameters
        """
        self.config = config or QDCuriosityConfig()

    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        """
        Select cells using UCB-based curiosity

        UCB formula: score(cell) = quality + C * sqrt(log(total_visits) / cell_visits)
        """
        result = qd_grid.sample_by_ucb(
            count=count,
            exploration_constant=self.config.explorationConstant,
            temperature=self.config.temperature,
        )

        logger.info(
            f"QD_CURIOSITY selected {len(result)} parents "
            f"(C={self.config.explorationConstant}, T={self.config.temperature})"
        )

        # Log visit statistics
        visit_stats = qd_grid.get_visit_stats()
        logger.debug(f"Visit stats: {visit_stats}")

        return self._ensure_count(result, count, gene_pool)


# ============================================================================
# NEW: RECENT Strategy
# ============================================================================


class RecentSelectionStrategy(BaseSelectionStrategy):
    """Select from recent generations to capitalize on momentum

    Favors genes from last N generations that meet quality threshold.
    Implements recency bias with adaptive fallback when recent pool is empty.

    Properties:
    - Momentum: Favors lineages with recent success
    - Recency bias: Recent genes more likely to be selected
    - Quality filter: Doesn't waste time on recent failures
    - Adaptive: Automatically adjusts to current search region
    """

    def __init__(self, config: RecentSelectionConfig | None = None):
        """
        Initialize recent selection

        Args:
            config: Configuration for lookback window and quality threshold
        """
        self.config = config or RecentSelectionConfig()

    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        """
        Select from recent high-quality genes
        """
        # Get recent genes
        recent_genes = gene_pool.get_genes_from_recent_generations(
            lookback=self.config.lookbackWindow,
            min_quality=self.config.minQualityThreshold,
        )

        if not recent_genes:
            if self.config.fallbackOnEmpty:
                logger.warning(
                    f"No recent genes meet criteria (window={self.config.lookbackWindow}, "
                    f"quality>={self.config.minQualityThreshold}). "
                    f"Falling back to relaxed criteria."
                )

                # Fallback 1: Remove quality threshold
                recent_genes = gene_pool.get_genes_from_recent_generations(
                    lookback=self.config.lookbackWindow, min_quality=None
                )

                # Fallback 2: Expand lookback
                if not recent_genes:
                    recent_genes = gene_pool.get_genes_from_recent_generations(
                        lookback=self.config.lookbackWindow * 2, min_quality=None
                    )

                # Fallback 3: All genes
                if not recent_genes:
                    recent_genes = gene_pool.list_sop_genes()
            else:
                logger.warning("No recent genes found and fallback disabled")
                return []

        # Sample from recent genes
        result: list[tuple[str, str]]
        if not recent_genes:
            # No genes available after fallback
            result = []
        elif self.config.samplingMethod == "uniform":
            # Uniform sampling
            n: int = min(count, len(recent_genes))
            indices: NDArray[np.intp] = np.random.choice(len(recent_genes), size=n, replace=False)
            selected: list[SOPGene] = [recent_genes[i] for i in indices]
            result = [(g.sopId, g.version) for g in selected]
        else:
            # Quality-weighted sampling
            result = self._quality_weighted_sample(recent_genes, count)

        logger.info(
            f"RECENT selected {len(result)} parents from last {self.config.lookbackWindow} "
            f"generations ({len(recent_genes)} candidates)"
        )

        return self._ensure_count(result, count, gene_pool)


# ============================================================================
# NEW: HYBRID Strategy
# ============================================================================


class HybridSelectionStrategy(BaseSelectionStrategy):
    """Composite strategy mixing multiple selection approaches

    Allocates budget across sub-strategies and merges results.
    Supports automatic deduplication and fallback when budget is underutilized.

    Example configuration:
    - 30% QD_CURIOSITY (exploration)
    - 50% QD_QUALITY_WEIGHTED (exploitation)
    - 20% RECENT (momentum)

    Properties:
    - Multi-objective: Balances multiple selection criteria
    - Flexible: Weights can be tuned for different evolution phases
    - Robust: Fallback strategy ensures budget is fully utilized
    """

    def __init__(
        self,
        config: HybridSelectionConfig | None = None,
        strategy_factory: Optional["SelectionStrategyFactory"] = None,
    ):
        """
        Initialize hybrid selection

        Args:
            config: Configuration for component strategies and weights
            strategy_factory: Factory for creating sub-strategies (required)
        """
        self.config = config or HybridSelectionConfig(components=[])
        self.factory = strategy_factory  # For creating sub-strategies

    def select(
        self,
        gene_pool: "GenePool",
        qd_grid: "QDGridManager",
        count: int,
    ) -> list[tuple[str, str]]:
        """
        Mix multiple strategies with specified weights
        """
        if not self.config.components:
            logger.error("HYBRID strategy has no components configured")
            return []

        if not self.factory:
            logger.error("HYBRID strategy requires factory for sub-strategy creation")
            return []

        selected: set[tuple[str, str]] = set()  # Use set for automatic deduplication

        for component in self.config.components:
            # Allocate budget for this component
            budget = int(count * component.weight)

            if budget == 0:
                continue

            # Create sub-strategy
            try:
                strategy = self.factory.create(
                    strategy_type=component.strategyType, config=component.config or {}
                )
            except Exception:
                logger.exception(f"Failed to create sub-strategy {component.strategyType}")
                continue

            # Select candidates
            try:
                candidates = strategy.select(gene_pool, qd_grid, budget)
                selected.update(candidates)

                logger.debug(
                    f"HYBRID component {component.strategyType} "
                    f"(weight={component.weight:.2f}) selected {len(candidates)} "
                    f"({len(selected)} unique so far)"
                )
            except Exception:
                logger.exception(f"Sub-strategy {component.strategyType} failed")

        # Convert to list
        result: list[tuple[str, str]] = list(selected)

        # Fill remaining budget if deduplication left us short
        if len(result) < count:
            remaining: int = count - len(result)

            logger.info(
                f"HYBRID: Filling {remaining} remaining slots with "
                f"{self.config.fallbackStrategy} strategy"
            )

            # Use fallback strategy for filling (quality-weighted from remaining genes)
            exclude_ids: set[tuple[str, str]] = set(result)
            all_genes: list[SOPGene] = gene_pool.list_sop_genes()
            available: list[SOPGene] = [g for g in all_genes if (g.sopId, g.version) not in exclude_ids]

            if available:
                # Quality-weighted sampling from available genes
                filler: list[tuple[str, str]] = self._quality_weighted_sample(available, remaining)
                result.extend(filler)

        logger.info(
            f"HYBRID selected {len(result)} parents ({len(self.config.components)} components)"
        )

        return result[:count]  # Ensure exact count
