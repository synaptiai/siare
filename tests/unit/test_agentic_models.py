"""Tests for agentic variation models (Hybrid Evolution)"""

import pytest

from siare.core.models import (
    AgenticVariationConfig,
    EvolutionRunSummary,
    InnerLoopBudget,
    KnowledgeDocument,
    MutationType,
    SupervisorContext,
    SupervisorDirective,
    VariationResult,
    VariationToolSpec,
)


# ============================================================================
# InnerLoopBudget Tests
# ============================================================================


class TestInnerLoopBudget:
    """Tests for InnerLoopBudget model."""

    def test_default_creation(self):
        budget = InnerLoopBudget()
        assert budget.maxLLMCalls == 20
        assert budget.maxDryRuns == 3
        assert budget.maxCostUSD == 1.0
        assert budget.llmCallsUsed == 0
        assert budget.dryRunsUsed == 0
        assert budget.costUsed == 0.0

    def test_not_exhausted_initially(self):
        budget = InnerLoopBudget()
        assert not budget.exhausted()

    def test_exhausted_by_llm_calls(self):
        budget = InnerLoopBudget(maxLLMCalls=2)
        budget.record_llm_call(cost=0.01)
        assert not budget.exhausted()
        budget.record_llm_call(cost=0.01)
        assert budget.exhausted()

    def test_exhausted_by_dry_runs(self):
        budget = InnerLoopBudget(maxDryRuns=1)
        budget.record_dry_run(cost=0.05)
        assert budget.exhausted()

    def test_exhausted_by_cost(self):
        budget = InnerLoopBudget(maxCostUSD=0.10)
        budget.record_llm_call(cost=0.11)
        assert budget.exhausted()

    def test_record_llm_call_accumulates(self):
        budget = InnerLoopBudget()
        budget.record_llm_call(cost=0.05)
        budget.record_llm_call(cost=0.03)
        assert budget.llmCallsUsed == 2
        assert budget.costUsed == pytest.approx(0.08)

    def test_record_dry_run_accumulates(self):
        budget = InnerLoopBudget()
        budget.record_dry_run(cost=0.10)
        budget.record_dry_run(cost=0.20)
        assert budget.dryRunsUsed == 2
        assert budget.costUsed == pytest.approx(0.30)

    def test_usage_summary(self):
        budget = InnerLoopBudget(maxLLMCalls=10, maxDryRuns=2, maxCostUSD=0.50)
        budget.record_llm_call(cost=0.01)
        summary = budget.usage_summary()
        assert summary["llm_calls"] == "1/10"
        assert summary["dry_runs"] == "0/2"
        assert "0.0100" in summary["cost_usd"]

    def test_validation_rejects_negative_max(self):
        with pytest.raises(ValueError):
            InnerLoopBudget(maxLLMCalls=0)

    def test_zero_dry_runs_not_immediately_exhausted(self):
        budget = InnerLoopBudget(maxDryRuns=0)
        assert not budget.exhausted()  # 0 dry runs allowed = skip dry-run check

    def test_custom_budget(self):
        budget = InnerLoopBudget(
            maxLLMCalls=50,
            maxDryRuns=10,
            maxCostUSD=5.0,
        )
        assert budget.maxLLMCalls == 50
        assert budget.maxDryRuns == 10
        assert budget.maxCostUSD == 5.0


# ============================================================================
# VariationToolSpec Tests
# ============================================================================


class TestVariationToolSpec:
    """Tests for VariationToolSpec model."""

    def test_creation(self):
        tool = VariationToolSpec(
            name="dry_run",
            description="Execute candidate SOP on sample task",
            parameters={"config": {"type": "object"}},
        )
        assert tool.name == "dry_run"
        assert tool.description == "Execute candidate SOP on sample task"
        assert "config" in tool.parameters

    def test_empty_parameters(self):
        tool = VariationToolSpec(
            name="inspect_trace",
            description="Inspect a trace",
        )
        assert tool.parameters == {}


# ============================================================================
# VariationResult Tests
# ============================================================================


class TestVariationResult:
    """Tests for VariationResult model."""

    def test_successful_result(self):
        from siare.core.models import ProcessConfig, RoleConfig, SOPMutation

        config = ProcessConfig(
            id="test",
            version="1.0.0",
            models={"m": "gpt-4"},
            tools=[],
            roles=[
                RoleConfig(id="r", model="gpt-4", promptRef="p")
            ],
            graph=[],
        )
        mutation = SOPMutation(
            parentSopId="parent",
            parentVersion="1.0.0",
            newConfig=config,
            rationale="test",
            mutationType=MutationType.PROMPT_CHANGE,
        )
        result = VariationResult(
            mutation=mutation,
            quality=0.85,
            iterationsUsed=3,
            reason="improvement",
        )
        assert result.succeeded
        assert result.quality == 0.85
        assert result.iterationsUsed == 3

    def test_empty_result(self):
        result = VariationResult(reason="no_improvement")
        assert not result.succeeded
        assert result.mutation is None
        assert result.quality is None
        assert result.iterationsUsed == 0

    def test_budget_exhausted_result(self):
        result = VariationResult(
            reason="budget_exhausted",
            iterationsUsed=5,
            innerBudgetUsed={"llm_calls": "20/20"},
        )
        assert not result.succeeded
        assert result.reason == "budget_exhausted"


# ============================================================================
# SupervisorDirective Tests
# ============================================================================


class TestSupervisorDirective:
    """Tests for SupervisorDirective model."""

    def test_creation(self):
        directive = SupervisorDirective(
            strategy="explore minimal topologies",
            focusArea="topology",
            mutationTypes=[MutationType.REMOVE_ROLE, MutationType.REWIRE_GRAPH],
            explorationTarget="low-complexity cells",
            rationale="All high-quality SOPs use 3+ agents",
        )
        assert directive.strategy == "explore minimal topologies"
        assert directive.focusArea == "topology"
        assert len(directive.mutationTypes) == 2
        assert not directive.overrideSelection
        assert directive.suggestedParents is None

    def test_with_parent_override(self):
        directive = SupervisorDirective(
            strategy="breed top performers",
            focusArea="prompts",
            mutationTypes=[MutationType.PROMPT_CHANGE],
            explorationTarget="high-accuracy Pareto region",
            rationale="Crossover of top two might yield breakthrough",
            overrideSelection=True,
            suggestedParents=[
                {"sopId": "sop-1", "version": "2.1.0"},
                {"sopId": "sop-2", "version": "1.3.0"},
            ],
        )
        assert directive.overrideSelection
        assert len(directive.suggestedParents) == 2


# ============================================================================
# SupervisorContext Tests
# ============================================================================


class TestSupervisorContext:
    """Tests for SupervisorContext model."""

    def test_default_creation(self):
        ctx = SupervisorContext()
        assert ctx.qdCoverage == {}
        assert ctx.paretoFrontier == []
        assert ctx.recentGenes == []
        assert ctx.diversityStats == {}
        assert ctx.mutationSuccessRates == {}
        assert ctx.qualityHistory == []

    def test_populated_context(self):
        ctx = SupervisorContext(
            qdCoverage={"total_cells": 25, "occupied": 8},
            paretoFrontier=[{"sopId": "sop-1", "quality": 0.9}],
            mutationSuccessRates={
                "prompt_change": 0.4,
                "add_role": 0.1,
            },
            qualityHistory=[0.5, 0.6, 0.65, 0.65, 0.65],
        )
        assert ctx.qdCoverage["total_cells"] == 25
        assert len(ctx.paretoFrontier) == 1
        assert ctx.mutationSuccessRates["prompt_change"] == 0.4


# ============================================================================
# AgenticVariationConfig Tests
# ============================================================================


class TestAgenticVariationConfig:
    """Tests for AgenticVariationConfig model."""

    def test_default_adaptive_mode(self):
        config = AgenticVariationConfig()
        assert config.mode == "adaptive"
        assert config.maxInnerIterations == 5
        assert config.improvementThreshold == 0.02
        assert config.enableSupervisor
        assert config.enableKnowledgeBase
        assert config.persistRunSummaries
        assert len(config.enabledTools) == 6

    def test_single_turn_mode(self):
        config = AgenticVariationConfig(mode="single_turn")
        assert config.mode == "single_turn"

    def test_agentic_mode(self):
        config = AgenticVariationConfig(mode="agentic")
        assert config.mode == "agentic"

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            AgenticVariationConfig(mode="invalid")  # type: ignore[arg-type]

    def test_custom_inner_budget(self):
        budget = InnerLoopBudget(maxLLMCalls=50, maxDryRuns=10)
        config = AgenticVariationConfig(innerBudget=budget)
        assert config.innerBudget.maxLLMCalls == 50
        assert config.innerBudget.maxDryRuns == 10

    def test_validation_thresholds(self):
        with pytest.raises(ValueError):
            AgenticVariationConfig(improvementThreshold=1.5)
        with pytest.raises(ValueError):
            AgenticVariationConfig(maxInnerIterations=0)

    def test_disabled_tools(self):
        config = AgenticVariationConfig(
            enabledTools=["dry_run", "validate_mutation"],
        )
        assert len(config.enabledTools) == 2

    def test_no_dry_run(self):
        config = AgenticVariationConfig(sampleTasksPerVariation=0)
        assert config.sampleTasksPerVariation == 0


# ============================================================================
# KnowledgeDocument Tests
# ============================================================================


class TestKnowledgeDocument:
    """Tests for KnowledgeDocument model."""

    def test_creation(self):
        doc = KnowledgeDocument(
            content="Use chain-of-thought prompting for complex reasoning.",
            category="prompt_engineering",
        )
        assert doc.content.startswith("Use chain")
        assert doc.category == "prompt_engineering"
        assert doc.id  # auto-generated UUID
        assert doc.relevanceScore is None

    def test_with_relevance(self):
        doc = KnowledgeDocument(
            content="RAG systems benefit from re-ranking.",
            category="rag_patterns",
            relevanceScore=0.92,
            metadata={"source": "builtin"},
        )
        assert doc.relevanceScore == 0.92
        assert doc.metadata["source"] == "builtin"


# ============================================================================
# EvolutionRunSummary Tests
# ============================================================================


class TestEvolutionRunSummary:
    """Tests for EvolutionRunSummary model."""

    def test_creation(self):
        summary = EvolutionRunSummary(
            jobId="job-123",
            domain="customer_support",
            totalGenerations=50,
            effectiveMutations={
                "prompt_change": 15,
                "param_tweak": 8,
                "add_role": 2,
            },
            breakthroughs=[
                {
                    "generation": 12,
                    "qualityJump": 0.15,
                    "mutationType": "add_role",
                    "description": "Added fact-checker role",
                }
            ],
            deadEnds=["remove_role consistently degraded quality"],
            finalParetoSize=5,
            bestQuality=0.92,
            qualityHistory=[0.5, 0.6, 0.7, 0.8, 0.85, 0.92],
        )
        assert summary.jobId == "job-123"
        assert summary.totalGenerations == 50
        assert summary.effectiveMutations["prompt_change"] == 15
        assert len(summary.breakthroughs) == 1
        assert summary.bestQuality == 0.92
        assert summary.completedAt  # auto-generated

    def test_minimal_creation(self):
        summary = EvolutionRunSummary(
            jobId="job-min",
            domain="test",
            totalGenerations=0,
        )
        assert summary.effectiveMutations == {}
        assert summary.breakthroughs == []
        assert summary.deadEnds == []
        assert summary.finalParetoSize == 0

    def test_validation_rejects_negative_generations(self):
        with pytest.raises(ValueError):
            EvolutionRunSummary(
                jobId="j",
                domain="d",
                totalGenerations=-1,
            )
