"""Tests for AgenticDirector multi-turn variation operator."""

from typing import Any

import pytest

from siare.core.models import (
    AggregatedMetric,
    AgenticVariationConfig,
    AggregationMethod,
    GraphEdge,
    InnerLoopBudget,
    MutationType,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
    SOPGene,
    SupervisorDirective,
    VariationResult,
)
from siare.services.agentic_director import AgenticDirector
from siare.services.llm_provider import LLMMessage, LLMResponse
from siare.services.variation_tools import VariationToolRegistry


# ============================================================================
# Test Fixtures
# ============================================================================


def make_llm_response(content: str, cost: float = 0.01) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="test-model",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        finish_reason="stop",
        cost=cost,
    )


class MockLLMProvider:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0
        self.calls: list[dict] = []

    def complete(self, messages, model, temperature=0.7, **kwargs) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model})
        idx = min(self._idx, len(self._responses) - 1)
        self._idx += 1
        return make_llm_response(self._responses[idx])

    def get_model_name(self, model_ref: str) -> str:
        return model_ref


def make_parent_config() -> ProcessConfig:
    return ProcessConfig(
        id="test-sop",
        version="1.0.0",
        models={"gpt-4": "gpt-4"},
        tools=["vector_search"],
        roles=[
            RoleConfig(id="retriever", model="gpt-4", promptRef="retriever_prompt"),
            RoleConfig(id="answerer", model="gpt-4", promptRef="answerer_prompt"),
        ],
        graph=[GraphEdge(from_="retriever", to="answerer")],
    )


def make_parent_genome() -> PromptGenome:
    return PromptGenome(
        id="test-genome",
        version="1.0.0",
        rolePrompts={
            "retriever_prompt": RolePrompt(
                id="retriever_prompt",
                content="You are a document retriever.",
            ),
            "answerer_prompt": RolePrompt(
                id="answerer_prompt",
                content="Answer the question based on context.",
            ),
        },
    )


def make_parent_gene() -> SOPGene:
    return SOPGene(
        sopId="test-sop",
        version="1.0.0",
        promptGenomeId="test-genome",
        promptGenomeVersion="1.0.0",
        configSnapshot=make_parent_config(),
        evaluations=[],
        aggregatedMetrics={
            "factuality": AggregatedMetric(
                metricId="factuality",
                mean=0.7, median=0.7,
                sampleSize=10,
                aggregationMethod=AggregationMethod.MEAN,
            ),
            "weighted_aggregate": AggregatedMetric(
                metricId="weighted_aggregate",
                mean=0.65, median=0.65,
                sampleSize=10,
                aggregationMethod=AggregationMethod.MEAN,
            ),
        },
        generation=5,
    )


# ============================================================================
# Basic Variation Tests
# ============================================================================


class TestAgenticDirectorBasic:
    """Tests for basic variation functionality."""

    def test_successful_prompt_change(self):
        """Agent proposes a valid PROMPT_CHANGE mutation."""
        response = (
            "After analyzing the SOP, the retriever prompt lacks specificity.\n\n"
            "MUTATION_TYPE: prompt_change\n"
            "TARGET_ROLE: retriever\n"
            "CHANGES: Add domain-specific retrieval instructions\n"
            "NEW_CONTENT: You are a specialized document retriever. "
            "Focus on retrieving recent, high-relevance documents.\n"
            "RATIONALE: The generic prompt leads to irrelevant retrieval."
        )
        provider = MockLLMProvider([response])
        director = AgenticDirector(
            llm_provider=provider,
            config=AgenticVariationConfig(maxInnerIterations=1),
        )

        result = director.vary(
            parent_gene=make_parent_gene(),
            parent_config=make_parent_config(),
            parent_genome=make_parent_genome(),
            metrics_to_optimize=["factuality"],
            mutation_types=[MutationType.PROMPT_CHANGE],
        )

        assert result.succeeded
        assert result.mutation is not None
        assert result.mutation.mutationType == MutationType.PROMPT_CHANGE
        assert "generic" in result.mutation.rationale.lower()

    def test_no_valid_mutation_returns_empty(self):
        """Agent fails to produce a valid mutation."""
        provider = MockLLMProvider([
            "I analyzed the SOP but I'm not sure what to change.",
            "Still not sure. The SOP looks fine to me.",
        ])
        director = AgenticDirector(
            llm_provider=provider,
            config=AgenticVariationConfig(maxInnerIterations=2),
        )

        result = director.vary(
            parent_gene=make_parent_gene(),
            parent_config=make_parent_config(),
            parent_genome=make_parent_genome(),
            metrics_to_optimize=["factuality"],
            mutation_types=[MutationType.PROMPT_CHANGE],
        )

        assert not result.succeeded
        assert result.reason == "no_improvement"

    def test_budget_exhaustion(self):
        """Budget exhaustion stops the loop."""
        provider = MockLLMProvider(["No mutation here."] * 5)
        config = AgenticVariationConfig(
            maxInnerIterations=10,
            innerBudget=InnerLoopBudget(maxLLMCalls=2),
        )
        director = AgenticDirector(
            llm_provider=provider,
            config=config,
        )

        result = director.vary(
            parent_gene=make_parent_gene(),
            parent_config=make_parent_config(),
            parent_genome=make_parent_genome(),
            metrics_to_optimize=["factuality"],
            mutation_types=[MutationType.PROMPT_CHANGE],
        )

        assert not result.succeeded
        # Should stop before all 10 iterations due to budget

    def test_disallowed_mutation_type_rejected(self):
        """Mutation types not in allowed list are rejected."""
        response = (
            "MUTATION_TYPE: add_role\n"
            "TARGET_ROLE: fact_checker\n"
            "CHANGES: Add a fact checker\n"
            "NEW_CONTENT: You check facts.\n"
            "RATIONALE: Need more verification."
        )
        provider = MockLLMProvider([response])
        director = AgenticDirector(
            llm_provider=provider,
            config=AgenticVariationConfig(maxInnerIterations=1),
        )

        result = director.vary(
            parent_gene=make_parent_gene(),
            parent_config=make_parent_config(),
            parent_genome=make_parent_genome(),
            metrics_to_optimize=["factuality"],
            mutation_types=[MutationType.PROMPT_CHANGE],  # Only prompt_change allowed
        )

        assert not result.succeeded


# ============================================================================
# Version Bumping Tests
# ============================================================================


class TestVersionBumping:
    """Tests for semantic version bumping based on mutation type."""

    def setup_method(self):
        self.director = AgenticDirector(
            llm_provider=MockLLMProvider([""]),
        )

    def test_prompt_change_bumps_minor(self):
        v = self.director._bump_version("1.0.0", MutationType.PROMPT_CHANGE)
        assert v.startswith("1.1.0-")

    def test_param_tweak_bumps_patch(self):
        v = self.director._bump_version("1.2.3", MutationType.PARAM_TWEAK)
        assert v.startswith("1.2.4-")

    def test_add_role_bumps_major(self):
        v = self.director._bump_version("1.0.0", MutationType.ADD_ROLE)
        assert v.startswith("2.0.0-")

    def test_remove_role_bumps_major(self):
        v = self.director._bump_version("1.0.0", MutationType.REMOVE_ROLE)
        assert v.startswith("2.0.0-")

    def test_rewire_graph_bumps_major(self):
        v = self.director._bump_version("1.0.0", MutationType.REWIRE_GRAPH)
        assert v.startswith("2.0.0-")

    def test_non_semver_gets_suffix(self):
        v = self.director._bump_version("v1", MutationType.PROMPT_CHANGE)
        assert v == "v1-mut"


# ============================================================================
# System Prompt Tests
# ============================================================================


class TestSystemPrompt:
    """Tests for system prompt construction."""

    def test_includes_tools(self):
        registry = VariationToolRegistry(
            enabled_tools=["validate_mutation", "inspect_trace"],
        )
        director = AgenticDirector(
            llm_provider=MockLLMProvider([""]),
            tool_registry=registry,
        )

        prompt = director._build_system_prompt(
            mutation_types=[MutationType.PROMPT_CHANGE],
            constraints=None,
            directive=None,
        )

        assert "validate_mutation" in prompt
        assert "inspect_trace" in prompt

    def test_includes_constraints(self):
        director = AgenticDirector(
            llm_provider=MockLLMProvider([""]),
        )

        prompt = director._build_system_prompt(
            mutation_types=[MutationType.PROMPT_CHANGE],
            constraints={
                "maxRoles": 5,
                "mandatoryRoles": ["retriever"],
            },
            directive=None,
        )

        assert "Max roles: 5" in prompt
        assert "retriever" in prompt

    def test_includes_supervisor_directive(self):
        director = AgenticDirector(
            llm_provider=MockLLMProvider([""]),
        )
        directive = SupervisorDirective(
            strategy="explore minimal topologies",
            focusArea="topology",
            mutationTypes=[MutationType.REMOVE_ROLE],
            explorationTarget="low-complexity cells",
            rationale="Stagnation in high-complexity region",
        )

        prompt = director._build_system_prompt(
            mutation_types=[MutationType.REMOVE_ROLE],
            constraints=None,
            directive=directive,
        )

        assert "Supervisor Directive" in prompt
        assert "explore minimal topologies" in prompt
        assert "low-complexity cells" in prompt


# ============================================================================
# Mutation Parsing Tests
# ============================================================================


class TestMutationParsing:
    """Tests for parsing mutations from LLM responses."""

    def setup_method(self):
        self.director = AgenticDirector(
            llm_provider=MockLLMProvider([""]),
        )
        self.config = make_parent_config()
        self.genome = make_parent_genome()

    def test_parse_prompt_change(self):
        response = (
            "MUTATION_TYPE: prompt_change\n"
            "TARGET_ROLE: retriever\n"
            "CHANGES: Improve retrieval specificity\n"
            "NEW_CONTENT: You are a highly focused retriever.\n"
            "RATIONALE: Current prompt is too generic."
        )
        mutation = self.director._parse_mutation_from_response(
            response, self.config, self.genome,
            [MutationType.PROMPT_CHANGE],
        )

        assert mutation is not None
        assert mutation.mutationType == MutationType.PROMPT_CHANGE
        assert "too generic" in mutation.rationale

    def test_parse_with_multiline_content(self):
        response = (
            "MUTATION_TYPE: prompt_change\n"
            "TARGET_ROLE: answerer\n"
            "CHANGES: Add chain-of-thought\n"
            "NEW_CONTENT: You are an answerer.\n"
            "Think step by step.\n"
            "Cite your sources.\n"
            "RATIONALE: Improves reasoning quality."
        )
        mutation = self.director._parse_mutation_from_response(
            response, self.config, self.genome,
            [MutationType.PROMPT_CHANGE],
        )

        assert mutation is not None
        prompt = mutation.newPromptGenome.rolePrompts.get("answerer_prompt")
        assert prompt is not None
        assert "step by step" in prompt.content

    def test_no_mutation_type_returns_none(self):
        response = "Here is my analysis of the SOP."
        mutation = self.director._parse_mutation_from_response(
            response, self.config, self.genome,
            [MutationType.PROMPT_CHANGE],
        )
        assert mutation is None

    def test_wrong_mutation_type_returns_none(self):
        response = (
            "MUTATION_TYPE: add_role\n"
            "TARGET_ROLE: new_role\n"
            "CHANGES: Add new role\n"
            "NEW_CONTENT: content\n"
            "RATIONALE: test"
        )
        mutation = self.director._parse_mutation_from_response(
            response, self.config, self.genome,
            [MutationType.PROMPT_CHANGE],  # add_role not allowed
        )
        assert mutation is None


# ============================================================================
# Integration-like Tests
# ============================================================================


class TestAgenticDirectorIntegration:
    """Tests for end-to-end variation flow."""

    def test_vary_with_tool_calls(self):
        """Agent uses tools before proposing mutation."""
        responses = [
            # First response: agent calls a tool
            'Let me inspect the trace.\n'
            '<tool_call>\n'
            '{"name": "validate_mutation", "arguments": '
            '{"mutation_type": "prompt_change", "num_roles_after": 2}}\n'
            '</tool_call>',
            # Second response: agent proposes mutation
            "Based on validation, here is my proposal:\n\n"
            "MUTATION_TYPE: prompt_change\n"
            "TARGET_ROLE: retriever\n"
            "CHANGES: Improve specificity\n"
            "NEW_CONTENT: You retrieve documents with high precision.\n"
            "RATIONALE: Validation passed, proceeding with targeted change.",
        ]
        provider = MockLLMProvider(responses)
        director = AgenticDirector(
            llm_provider=provider,
            config=AgenticVariationConfig(maxInnerIterations=1),
        )

        result = director.vary(
            parent_gene=make_parent_gene(),
            parent_config=make_parent_config(),
            parent_genome=make_parent_genome(),
            metrics_to_optimize=["factuality"],
            mutation_types=[MutationType.PROMPT_CHANGE],
        )

        assert result.succeeded
        assert result.mutation is not None

    def test_vary_with_supervisor_directive(self):
        """Directive from supervisor is included in system prompt."""
        response = (
            "Following the supervisor directive to explore topology.\n\n"
            "MUTATION_TYPE: prompt_change\n"
            "TARGET_ROLE: retriever\n"
            "CHANGES: Simplify retrieval prompt\n"
            "NEW_CONTENT: Retrieve relevant documents only.\n"
            "RATIONALE: Supervisor suggests minimizing complexity."
        )
        provider = MockLLMProvider([response])
        directive = SupervisorDirective(
            strategy="simplify",
            focusArea="prompts",
            mutationTypes=[MutationType.PROMPT_CHANGE],
            explorationTarget="minimal prompts",
            rationale="Stagnation at complex configs",
        )
        director = AgenticDirector(
            llm_provider=provider,
            config=AgenticVariationConfig(maxInnerIterations=1),
        )

        result = director.vary(
            parent_gene=make_parent_gene(),
            parent_config=make_parent_config(),
            parent_genome=make_parent_genome(),
            metrics_to_optimize=["factuality"],
            mutation_types=[MutationType.PROMPT_CHANGE],
            directive=directive,
        )

        assert result.succeeded
        # Verify directive was in the system prompt
        system_msg = provider.calls[0]["messages"][0]
        assert "simplify" in system_msg.content
