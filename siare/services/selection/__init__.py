"""Selection strategies for parent selection during evolution

This module provides various strategies for selecting parent SOPs during evolution.
Each strategy balances exploration (diversity) vs exploitation (quality) differently.

Available Strategies:
- PARETO: Select from Pareto frontier (exploitation)
- QD_UNIFORM: Uniform sampling from QD grid (exploration)
- QD_QUALITY_WEIGHTED: Quality-weighted QD sampling (balanced)
- QD_CURIOSITY: UCB-based curiosity-driven selection (NEW)
- RECENT: Momentum-based selection from recent generations (NEW)
- TOURNAMENT: Tournament selection (configurable pressure)
- HYBRID: Composite mixing of multiple strategies (NEW)

Usage:
    from siare.services.selection import SelectionStrategyFactory
    from siare.core.models import SelectionStrategy

    factory = SelectionStrategyFactory()
    strategy = factory.create(SelectionStrategy.QD_CURIOSITY, {"exploration_constant": 1.5})
    parents = strategy.select(gene_pool, qd_grid, count=10)
"""

from siare.services.selection.factory import SelectionStrategyFactory
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


__all__ = [
    "BaseSelectionStrategy",
    "HybridSelectionStrategy",
    "ParetoSelectionStrategy",
    "QDCuriositySelectionStrategy",
    "QDQualityWeightedStrategy",
    "QDUniformSelectionStrategy",
    "RecentSelectionStrategy",
    "SelectionStrategyFactory",
    "TournamentSelectionStrategy",
]
