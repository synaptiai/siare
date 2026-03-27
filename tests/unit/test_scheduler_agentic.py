"""Tests for agentic evolution integration in EvolutionScheduler."""

from unittest.mock import MagicMock, patch

import pytest

from siare.core.models import (
    AgenticVariationConfig,
    MutationType,
    SupervisorDirective,
)


# ============================================================================
# Mode Selection Tests
# ============================================================================


class TestAgenticModeSelection:
    """Tests for _should_use_agentic mode determination."""

    def _make_scheduler(self, agentic_config=None, has_director=True):
        """Create a minimal scheduler for testing mode selection."""
        from siare.services.scheduler import EvolutionScheduler

        scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
        scheduler.agentic_config = agentic_config
        scheduler._agentic_director = MagicMock() if has_director else None
        scheduler._supervisor = None
        scheduler._current_directive = None
        scheduler._redirections_this_phase = 0
        scheduler._using_agentic_mode = False
        return scheduler

    def test_no_config_returns_false(self):
        scheduler = self._make_scheduler(agentic_config=None)
        assert not scheduler._should_use_agentic()

    def test_no_director_returns_false(self):
        config = AgenticVariationConfig(mode="agentic")
        scheduler = self._make_scheduler(
            agentic_config=config, has_director=False,
        )
        assert not scheduler._should_use_agentic()

    def test_agentic_mode_returns_true(self):
        config = AgenticVariationConfig(mode="agentic")
        scheduler = self._make_scheduler(agentic_config=config)
        scheduler._using_agentic_mode = True  # Set by __init__
        assert scheduler._should_use_agentic()

    def test_single_turn_mode_returns_false(self):
        config = AgenticVariationConfig(mode="single_turn")
        scheduler = self._make_scheduler(agentic_config=config)
        assert not scheduler._should_use_agentic()

    def test_adaptive_mode_default_false(self):
        config = AgenticVariationConfig(mode="adaptive")
        scheduler = self._make_scheduler(agentic_config=config)
        assert not scheduler._should_use_agentic()

    def test_adaptive_mode_after_escalation(self):
        config = AgenticVariationConfig(mode="adaptive")
        scheduler = self._make_scheduler(agentic_config=config)
        scheduler._using_agentic_mode = True
        assert scheduler._should_use_agentic()


# ============================================================================
# Stagnation Handling Tests
# ============================================================================


class TestStagnationHandling:
    """Tests for _check_and_handle_stagnation."""

    def _make_scheduler(self, quality_history, max_redirections=3):
        from siare.core.config import ConvergenceConfig
        from siare.services.scheduler import EvolutionScheduler

        scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
        scheduler.agentic_config = AgenticVariationConfig(
            mode="adaptive",
            maxRedirectionsPerPhase=max_redirections,
        )
        scheduler._supervisor = MagicMock()
        scheduler._supervisor.analyze_and_redirect.return_value = SupervisorDirective(
            strategy="test strategy",
            focusArea="prompts",
            mutationTypes=[MutationType.PROMPT_CHANGE],
            explorationTarget="test target",
            rationale="test rationale",
        )
        scheduler._agentic_director = MagicMock()
        scheduler._current_directive = None
        scheduler._redirections_this_phase = 0
        scheduler._using_agentic_mode = False
        scheduler.best_quality_history = quality_history
        scheduler.convergence_config = ConvergenceConfig(
            convergence_window=3,
            convergence_threshold=0.01,
        )
        scheduler.current_job = None
        return scheduler

    def test_no_stagnation_returns_false(self):
        # Quality is improving
        scheduler = self._make_scheduler([0.5, 0.6, 0.7, 0.8])
        assert not scheduler._check_and_handle_stagnation()

    def test_stagnation_invokes_supervisor(self):
        # Quality flat for 3+ generations
        scheduler = self._make_scheduler([0.5, 0.65, 0.65, 0.65, 0.65])
        result = scheduler._check_and_handle_stagnation()

        assert result is True
        assert scheduler._current_directive is not None
        assert scheduler._using_agentic_mode is True
        assert scheduler._redirections_this_phase == 1
        scheduler._supervisor.analyze_and_redirect.assert_called_once()

    def test_max_redirections_stops_supervisor(self):
        scheduler = self._make_scheduler(
            [0.5, 0.65, 0.65, 0.65, 0.65],
            max_redirections=0,
        )
        result = scheduler._check_and_handle_stagnation()
        assert result is False

    def test_no_supervisor_returns_false(self):
        scheduler = self._make_scheduler([0.65, 0.65, 0.65, 0.65])
        scheduler._supervisor = None
        assert not scheduler._check_and_handle_stagnation()

    def test_too_few_generations_returns_false(self):
        scheduler = self._make_scheduler([0.65, 0.65])
        assert not scheduler._check_and_handle_stagnation()


# ============================================================================
# Phase Transition Tests
# ============================================================================


class TestPhaseTransitionAgentic:
    """Tests for phase transitions resetting agentic state."""

    def test_advance_phase_resets_agentic_state(self):
        from siare.core.models import (
            BudgetLimit,
            EvolutionConstraints,
            EvolutionJob,
            EvolutionJobStatus,
            EvolutionPhase,
            SelectionStrategy,
            TaskSet,
        )
        from siare.services.scheduler import EvolutionScheduler

        scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
        scheduler.agentic_config = AgenticVariationConfig(mode="adaptive")
        scheduler._using_agentic_mode = True
        scheduler._current_directive = SupervisorDirective(
            strategy="old",
            focusArea="prompts",
            mutationTypes=[MutationType.PROMPT_CHANGE],
            explorationTarget="old target",
            rationale="old",
        )
        scheduler._redirections_this_phase = 2

        job = EvolutionJob(
            id="test",
            domain="test",
            baseSops=[{"sopId": "s", "sopVersion": "1.0.0",
                       "promptGenomeId": "g", "promptGenomeVersion": "1.0.0"}],
            taskSet=TaskSet(
                id="ts", domain="test", version="1.0.0",
                tasks=[],
                evaluationSplit={"train": 0.8, "test": 0.2},
            ),
            metricsToOptimize=["quality"],
            qualityScoreWeights={"quality": 1.0},
            constraints=EvolutionConstraints(),
            phases=[
                EvolutionPhase(
                    name="p1",
                    allowedMutationTypes=[MutationType.PROMPT_CHANGE],
                    selectionStrategy=SelectionStrategy.PARETO,
                    parentsPerGeneration=2,
                    maxGenerations=10,
                ),
                EvolutionPhase(
                    name="p2",
                    allowedMutationTypes=[MutationType.ADD_ROLE],
                    selectionStrategy=SelectionStrategy.PARETO,
                    parentsPerGeneration=2,
                    maxGenerations=10,
                ),
            ],
            status=EvolutionJobStatus.RUNNING,
            currentPhaseIndex=0,
        )

        scheduler._advance_phase(job)

        assert job.currentPhaseIndex == 1
        assert scheduler._redirections_this_phase == 0
        assert scheduler._current_directive is None
        assert scheduler._using_agentic_mode is False


# ============================================================================
# Config Integration Tests
# ============================================================================


class TestAgenticConfigIntegration:
    """Tests for AgenticVariationConfig on the scheduler."""

    def test_scheduler_accepts_none_config(self):
        """Scheduler works without agentic config (backward compatible)."""
        from siare.services.scheduler import EvolutionScheduler

        scheduler = EvolutionScheduler.__new__(EvolutionScheduler)
        scheduler.agentic_config = None
        scheduler._agentic_director = None
        scheduler._supervisor = None
        scheduler._current_directive = None
        scheduler._redirections_this_phase = 0
        scheduler._using_agentic_mode = False

        assert not scheduler._should_use_agentic()
