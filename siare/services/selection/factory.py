"""Factory for creating selection strategies"""

import logging
from typing import Any, Optional

from siare.core.models import (
    HybridSelectionConfig,
    QDCuriosityConfig,
    RecentSelectionConfig,
)
from siare.core.models import (
    SelectionStrategy as SelectionStrategyEnum,
)
from siare.services.selection.strategies import (
    BaseSelectionStrategy,
    HybridSelectionStrategy,
    ParetoSelectionStrategy,
    QDCuriositySelectionStrategy,
    QDQualityWeightedStrategy,
    QDUniformSelectionStrategy,
    RecentSelectionStrategy,
    TournamentSelectionStrategy,
)


logger = logging.getLogger(__name__)


class SelectionStrategyFactory:
    """Factory for creating selection strategy instances

    Provides centralized creation of all selection strategies with proper
    configuration handling and error reporting.

    Example usage:
        factory = SelectionStrategyFactory()
        strategy = factory.create(
            SelectionStrategyEnum.QD_CURIOSITY,
            {"exploration_constant": 1.5, "temperature": 2.0}
        )
        parents = strategy.select(gene_pool, qd_grid, count=10)
    """

    def __init__(self):
        """Initialize factory"""
        # Singleton instance for HYBRID strategies to reference
        self._instance = self

    def create(  # noqa: PLR0911 - Multiple returns acceptable in factory pattern
        self,
        strategy_type: SelectionStrategyEnum,
        config: Optional[dict[str, Any]] = None,
    ) -> BaseSelectionStrategy:
        """
        Create a selection strategy instance

        Args:
            strategy_type: Type of strategy to create
            config: Strategy-specific configuration dict

        Returns:
            BaseSelectionStrategy instance

        Raises:
            ValueError: If strategy type is unknown
            ValidationError: If config is invalid for the strategy type
        """
        config = config or {}

        if strategy_type == SelectionStrategyEnum.PARETO:
            return ParetoSelectionStrategy()

        if strategy_type == SelectionStrategyEnum.QD_UNIFORM:
            return QDUniformSelectionStrategy()

        if strategy_type == SelectionStrategyEnum.QD_QUALITY_WEIGHTED:
            return QDQualityWeightedStrategy()

        if strategy_type == SelectionStrategyEnum.QD_CURIOSITY:
            # Parse config with Pydantic validation
            qd_config = QDCuriosityConfig(**config)
            return QDCuriositySelectionStrategy(qd_config)

        if strategy_type == SelectionStrategyEnum.RECENT:
            # Parse config with Pydantic validation
            recent_config = RecentSelectionConfig(**config)
            return RecentSelectionStrategy(recent_config)

        if strategy_type == SelectionStrategyEnum.TOURNAMENT:
            # Extract tournament size (with default)
            tournament_size = config.get("tournament_size", 3)
            return TournamentSelectionStrategy(tournament_size)

        if strategy_type == SelectionStrategyEnum.HYBRID:
            # Parse config with Pydantic validation
            hybrid_config = HybridSelectionConfig(**config)
            # Pass self as factory for sub-strategy creation
            return HybridSelectionStrategy(hybrid_config, strategy_factory=self)

        raise ValueError(f"Unknown selection strategy: {strategy_type}")
