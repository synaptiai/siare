"""Tests for SupervisorAgent stagnation analysis and redirection."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from siare.core.models import (
    MutationType,
    SupervisorContext,
    SupervisorDirective,
)
from siare.services.llm_provider import LLMMessage, LLMResponse
from siare.services.supervisor import SupervisorAgent


# ============================================================================
# Mock Infrastructure
# ============================================================================


def make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        usage={"prompt_tokens": 200, "completion_tokens": 100, "total_tokens": 300},
        finish_reason="stop",
        cost=0.02,
    )


class MockLLMProvider:
    def __init__(self, response_content: str) -> None:
        self._content = response_content
        self.calls: list[dict[str, Any]] = []

    def complete(self, messages, model, temperature=0.7, **kwargs) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model})
        return make_llm_response(self._content)

    def get_model_name(self, model_ref: str) -> str:
        return model_ref


class MockGenePool:
    def __init__(self) -> None:
        self.genes: list[Any] = []

    def get_pareto_frontier(self, **kwargs):
        return self.genes

    def get_genes_from_recent_generations(self, lookback=10, **kwargs):
        return self.genes

    def get_diversity_stats(self):
        return {"unique_sops": 5, "total_versions": 12}


class MockQDGrid:
    def __init__(self, visit_stats: dict[str, Any] | None = None) -> None:
        self._visit_stats = visit_stats or {
            "total_selections": 50,
            "cells_visited": 8,
            "cells_unvisited": 17,
        }

    def get_visit_stats(self) -> dict[str, Any]:
        return self._visit_stats


# ============================================================================
# Directive Parsing Tests
# ============================================================================


class TestSupervisorDirectiveParsing:
    """Tests for parsing LLM responses into directives."""

    def test_parse_well_formed_response(self):
        response = (
            "STRATEGY: explore minimal topologies\n"
            "FOCUS_AREA: topology\n"
            "MUTATION_TYPES: remove_role, rewire_graph\n"
            "EXPLORATION_TARGET: low-complexity QD cells\n"
            "RATIONALE: Most SOPs have 3+ agents but minimal configs untried"
        )
        provider = MockLLMProvider(response)
        supervisor = SupervisorAgent(
            llm_provider=provider,
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid(),
        )

        directive = supervisor.analyze_and_redirect(
            quality_history=[0.5, 0.6, 0.65, 0.65, 0.65],
        )

        assert directive.strategy == "explore minimal topologies"
        assert directive.focusArea == "topology"
        assert MutationType.REMOVE_ROLE in directive.mutationTypes
        assert MutationType.REWIRE_GRAPH in directive.mutationTypes
        assert "low-complexity" in directive.explorationTarget
        assert "untried" in directive.rationale

    def test_parse_prompt_focused_response(self):
        response = (
            "STRATEGY: refine retriever prompts\n"
            "FOCUS_AREA: prompts\n"
            "MUTATION_TYPES: prompt_change\n"
            "EXPLORATION_TARGET: high-accuracy Pareto region\n"
            "RATIONALE: Retriever prompts lack specificity"
        )
        provider = MockLLMProvider(response)
        supervisor = SupervisorAgent(
            llm_provider=provider,
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid(),
        )

        directive = supervisor.analyze_and_redirect(
            quality_history=[0.7, 0.7, 0.7],
        )

        assert directive.focusArea == "prompts"
        assert directive.mutationTypes == [MutationType.PROMPT_CHANGE]

    def test_fallback_on_unparseable_response(self):
        response = "I'm not sure what to do. The data is unclear."
        provider = MockLLMProvider(response)
        supervisor = SupervisorAgent(
            llm_provider=provider,
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid(),
        )

        directive = supervisor.analyze_and_redirect(
            quality_history=[0.5, 0.5],
        )

        # Defaults applied
        assert directive.strategy == "explore alternative approaches"
        assert directive.focusArea == "prompts"
        assert directive.mutationTypes == [MutationType.PROMPT_CHANGE]

    def test_invalid_focus_area_defaults_to_prompts(self):
        response = (
            "STRATEGY: test\n"
            "FOCUS_AREA: quantum_computing\n"
            "MUTATION_TYPES: prompt_change\n"
            "EXPLORATION_TARGET: test\n"
            "RATIONALE: test"
        )
        provider = MockLLMProvider(response)
        supervisor = SupervisorAgent(
            llm_provider=provider,
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid(),
        )

        directive = supervisor.analyze_and_redirect(
            quality_history=[0.5],
        )
        assert directive.focusArea == "prompts"


# ============================================================================
# Context Gathering Tests
# ============================================================================


class TestSupervisorContextGathering:
    """Tests for gathering evolutionary trajectory evidence."""

    def test_gather_context_with_data(self):
        supervisor = SupervisorAgent(
            llm_provider=MockLLMProvider("STRATEGY: test\nFOCUS_AREA: prompts\nMUTATION_TYPES: prompt_change\nEXPLORATION_TARGET: test\nRATIONALE: test"),
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid({"total_selections": 100, "cells_unvisited": 5}),
        )

        context = supervisor._gather_context(
            quality_history=[0.5, 0.6, 0.65],
            recent_generations=5,
        )

        assert isinstance(context, SupervisorContext)
        assert context.qualityHistory == [0.5, 0.6, 0.65]
        assert context.qdCoverage["cells_unvisited"] == 5
        assert context.diversityStats["unique_sops"] == 5

    def test_gather_context_resilient_to_errors(self):
        class FailingGenePool:
            def get_pareto_frontier(self, **kwargs):
                raise RuntimeError("DB down")

            def get_genes_from_recent_generations(self, **kwargs):
                raise RuntimeError("DB down")

            def get_diversity_stats(self):
                raise RuntimeError("DB down")

        class FailingQDGrid:
            def get_visit_stats(self):
                raise RuntimeError("Grid error")

        supervisor = SupervisorAgent(
            llm_provider=MockLLMProvider("STRATEGY: test\nFOCUS_AREA: prompts\nMUTATION_TYPES: prompt_change\nEXPLORATION_TARGET: test\nRATIONALE: test"),
            gene_pool=FailingGenePool(),
            qd_grid=FailingQDGrid(),
        )

        context = supervisor._gather_context([0.5], 5)
        assert context.qdCoverage == {}
        assert context.paretoFrontier == []
        assert context.recentGenes == []


# ============================================================================
# Fallback Behavior Tests
# ============================================================================


class TestSupervisorFallback:
    """Tests for fallback when LLM is unavailable."""

    def test_fallback_with_unvisited_cells(self):
        supervisor = SupervisorAgent(
            llm_provider=MockLLMProvider(""),
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid({"cells_unvisited": 10}),
        )

        context = SupervisorContext(
            qdCoverage={"cells_unvisited": 10},
        )
        directive = supervisor._fallback_directive(context)

        assert directive.focusArea == "topology"
        assert MutationType.ADD_ROLE in directive.mutationTypes

    def test_fallback_without_unvisited_cells(self):
        supervisor = SupervisorAgent(
            llm_provider=MockLLMProvider(""),
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid({"cells_unvisited": 0}),
        )

        context = SupervisorContext(
            qdCoverage={"cells_unvisited": 0},
        )
        directive = supervisor._fallback_directive(context)

        assert directive.focusArea == "prompts"
        assert directive.mutationTypes == [MutationType.PROMPT_CHANGE]


# ============================================================================
# LLM Integration Tests
# ============================================================================


class TestSupervisorLLMCall:
    """Tests for LLM call behavior."""

    def test_calls_llm_with_correct_temperature(self):
        provider = MockLLMProvider(
            "STRATEGY: test\nFOCUS_AREA: prompts\n"
            "MUTATION_TYPES: prompt_change\n"
            "EXPLORATION_TARGET: test\nRATIONALE: test"
        )
        supervisor = SupervisorAgent(
            llm_provider=provider,
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid(),
            model="gpt-5",
        )

        supervisor.analyze_and_redirect(quality_history=[0.5])

        assert len(provider.calls) == 1
        assert provider.calls[0]["model"] == "gpt-5"

    def test_includes_quality_history_in_prompt(self):
        provider = MockLLMProvider(
            "STRATEGY: test\nFOCUS_AREA: prompts\n"
            "MUTATION_TYPES: prompt_change\n"
            "EXPLORATION_TARGET: test\nRATIONALE: test"
        )
        supervisor = SupervisorAgent(
            llm_provider=provider,
            gene_pool=MockGenePool(),
            qd_grid=MockQDGrid(),
        )

        supervisor.analyze_and_redirect(
            quality_history=[0.5, 0.6, 0.65, 0.65],
        )

        user_message = provider.calls[0]["messages"][-1]
        assert "0.650" in user_message.content
