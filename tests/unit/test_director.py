"""Tests for Director Service constraint validation"""

import pytest

from siare.core.models import (
    MutationType,
    ProcessConfig,
    PromptConstraints,
    PromptGenome,
    RoleConfig,
    RolePrompt,
)
from siare.services.director import Architect
from siare.services.llm_provider import LLMMessage, LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def __init__(self, mock_response: str):
        self.mock_response = mock_response

    def complete(
        self,
        messages: list[LLMMessage],
        model: str = "gpt-5",
        temperature: float = 0.7,
        max_tokens: int = 4000,
    ) -> LLMResponse:
        """Return mock response"""
        return LLMResponse(
            content=self.mock_response,
            model=model,
            usage={"prompt_tokens": 100, "completion_tokens": 50},
            finish_reason="stop",
        )

    def get_model_name(self, model_ref: str) -> str:
        """Return model name"""
        return model_ref


@pytest.fixture
def sample_sop_config():
    """Create a sample SOP config for testing"""
    return ProcessConfig(
        id="test-sop",
        version="1.0.0",
        models={"default": "gpt-5"},
        tools=["search"],
        roles=[
            RoleConfig(
                id="researcher",
                model="gpt-5",
                tools=["search"],
                promptRef="prompt_researcher",
            ),
            RoleConfig(
                id="writer", model="gpt-5", tools=None, promptRef="prompt_writer"
            ),
        ],
        graph=[{"from": "user_input", "to": "researcher"}, {"from": "researcher", "to": "writer"}],
    )


@pytest.fixture
def sample_prompt_genome():
    """Create a sample prompt genome for testing"""
    return PromptGenome(
        id="test-genome",
        version="1.0.0",
        rolePrompts={
            "prompt_researcher": RolePrompt(
                id="prompt_researcher",
                content="You are a researcher. IMPORTANT: Always cite sources.",
                constraints=PromptConstraints(
                    mustNotChange=["IMPORTANT: Always cite sources."],
                    allowedChanges=["Can modify research methodology"],
                ),
            ),
            "prompt_writer": RolePrompt(
                id="prompt_writer",
                content="You are a writer. Format the content clearly.",
                constraints=None,
            ),
        },
    )


class TestPromptConstraintValidation:
    """Test prompt constraint validation"""

    def test_valid_prompt_change_without_constraints(self, sample_sop_config, sample_prompt_genome):
        """Test that prompt changes without constraints are allowed"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        # Change prompt for writer (no constraints)
        old_prompt = "You are a writer. Format the content clearly."
        new_prompt = "You are an expert writer. Format content beautifully."

        # Should not raise
        architect._validate_prompt_constraints(old_prompt, new_prompt, None)

    def test_valid_prompt_change_with_constraints_preserved(
        self, sample_sop_config, sample_prompt_genome
    ):
        """Test that prompt changes preserving mustNotChange are allowed"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = PromptConstraints(
            mustNotChange=["IMPORTANT: Always cite sources."],
            allowedChanges=["Can modify research methodology"],
        )

        old_prompt = "You are a researcher. IMPORTANT: Always cite sources."
        new_prompt = (
            "You are an expert researcher. Use deep analysis. IMPORTANT: Always cite sources."
        )

        # Should not raise - constraint is preserved
        architect._validate_prompt_constraints(old_prompt, new_prompt, constraints)

    def test_invalid_prompt_change_removes_must_not_change(
        self, sample_sop_config, sample_prompt_genome
    ):
        """Test that removing mustNotChange text raises ValueError"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = PromptConstraints(
            mustNotChange=["IMPORTANT: Always cite sources."],
            allowedChanges=["Can modify research methodology"],
        )

        old_prompt = "You are a researcher. IMPORTANT: Always cite sources."
        new_prompt = "You are an expert researcher. Use deep analysis."

        # Should raise - required text removed
        with pytest.raises(ValueError, match="Constraint violation: Required text"):
            architect._validate_prompt_constraints(old_prompt, new_prompt, constraints)

    def test_invalid_prompt_change_modifies_must_not_change(
        self, sample_sop_config, sample_prompt_genome
    ):
        """Test that modifying mustNotChange text raises ValueError"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = PromptConstraints(
            mustNotChange=["IMPORTANT: Always cite sources."],
        )

        old_prompt = "You are a researcher. IMPORTANT: Always cite sources."
        new_prompt = "You are a researcher. IMPORTANT: Always cite references."  # Modified

        # Should raise - required text modified
        with pytest.raises(ValueError, match="Constraint violation: Required text"):
            architect._validate_prompt_constraints(old_prompt, new_prompt, constraints)

    def test_multiple_must_not_change_constraints(self, sample_sop_config, sample_prompt_genome):
        """Test validation with multiple mustNotChange constraints"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = PromptConstraints(
            mustNotChange=["IMPORTANT: Always cite sources.", "Use peer-reviewed sources only."],
        )

        old_prompt = "You are a researcher. IMPORTANT: Always cite sources. Use peer-reviewed sources only."
        new_prompt = "You are a researcher. IMPORTANT: Always cite sources."

        # Should raise - one required text missing
        with pytest.raises(ValueError, match="Use peer-reviewed sources only"):
            architect._validate_prompt_constraints(old_prompt, new_prompt, constraints)


class TestEvolutionConstraintValidation:
    """Test evolution constraint validation"""

    def test_disallowed_mutation_type(self, sample_sop_config):
        """Test that disallowed mutation types raise ValueError"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = {"disallowed_mutation_types": [MutationType.REMOVE_ROLE]}

        # Should raise - REMOVE_ROLE is disallowed
        with pytest.raises(ValueError, match="Constraint violation: Mutation type"):
            architect._validate_evolution_constraints(
                MutationType.REMOVE_ROLE, "researcher", sample_sop_config, constraints
            )

    def test_allowed_mutation_type(self, sample_sop_config):
        """Test that allowed mutation types pass validation"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = {"disallowed_mutation_types": [MutationType.REMOVE_ROLE]}

        # Should not raise - PROMPT_CHANGE is allowed
        architect._validate_evolution_constraints(
            MutationType.PROMPT_CHANGE, "researcher", sample_sop_config, constraints
        )

    def test_max_roles_constraint_add_role(self, sample_sop_config):
        """Test max_roles constraint for ADD_ROLE mutation"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        # SOP has 2 roles, max is 2
        constraints = {"max_roles": 2}

        # Should raise - already at max
        with pytest.raises(ValueError, match="Maximum roles.*already reached"):
            architect._validate_evolution_constraints(
                MutationType.ADD_ROLE, None, sample_sop_config, constraints
            )

    def test_max_roles_constraint_allows_add_when_under_limit(self, sample_sop_config):
        """Test max_roles allows adding when under limit"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        # SOP has 2 roles, max is 5
        constraints = {"max_roles": 5}

        # Should not raise - under limit
        architect._validate_evolution_constraints(
            MutationType.ADD_ROLE, None, sample_sop_config, constraints
        )

    def test_mandatory_roles_constraint_remove_role(self, sample_sop_config):
        """Test mandatory_roles constraint for REMOVE_ROLE mutation"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = {"mandatory_roles": ["researcher", "writer"]}

        # Should raise - researcher is mandatory
        with pytest.raises(ValueError, match="Cannot remove role.*mandatory"):
            architect._validate_evolution_constraints(
                MutationType.REMOVE_ROLE, "researcher", sample_sop_config, constraints
            )

    def test_mandatory_roles_allows_remove_non_mandatory(self, sample_sop_config):
        """Test mandatory_roles allows removing non-mandatory roles"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = {"mandatory_roles": ["researcher"]}

        # Should not raise - writer is not mandatory
        architect._validate_evolution_constraints(
            MutationType.REMOVE_ROLE, "writer", sample_sop_config, constraints
        )

    def test_no_constraints(self, sample_sop_config):
        """Test that mutations work without constraints"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        # Should not raise - no constraints
        architect._validate_evolution_constraints(
            MutationType.REMOVE_ROLE, "researcher", sample_sop_config, None
        )

    def test_empty_constraints_dict(self, sample_sop_config):
        """Test that mutations work with empty constraints dict"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        # Should not raise - empty constraints
        architect._validate_evolution_constraints(
            MutationType.REMOVE_ROLE, "researcher", sample_sop_config, {}
        )


class TestIntegratedConstraintValidation:
    """Test integrated constraint validation in _apply_mutation"""

    def test_prompt_change_with_constraints_integration(
        self, sample_sop_config, sample_prompt_genome
    ):
        """Test full integration of prompt constraint validation"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        # Valid change - preserves mustNotChange
        new_content = "You are an expert researcher. IMPORTANT: Always cite sources."

        # Should not raise
        new_config, new_genome = architect._apply_mutation(
            MutationType.PROMPT_CHANGE,
            "researcher",
            new_content,
            sample_sop_config,
            sample_prompt_genome,
            constraints=None,
        )

        # Verify the change was applied
        assert new_genome.rolePrompts["prompt_researcher"].content == new_content

    def test_prompt_change_violates_constraints_integration(
        self, sample_sop_config, sample_prompt_genome
    ):
        """Test that constraint violations in _apply_mutation raise ValueError"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        # Invalid change - removes mustNotChange text
        new_content = "You are an expert researcher."

        # Should raise
        with pytest.raises(ValueError, match="Constraint violation: Required text"):
            architect._apply_mutation(
                MutationType.PROMPT_CHANGE,
                "researcher",
                new_content,
                sample_sop_config,
                sample_prompt_genome,
                constraints=None,
            )

    def test_add_role_violates_max_roles_integration(self, sample_sop_config, sample_prompt_genome):
        """Test that ADD_ROLE respects max_roles constraint"""
        mock_response = """
ROLE_ID: analyzer
MODEL: gpt-5
TOOLS: search
PROMPT: You are an analyzer.
EDGES_FROM: user_input
EDGES_TO: researcher
"""
        architect = Architect(llm_provider=MockLLMProvider(mock_response))

        constraints = {"max_roles": 2}  # Already at max

        # Should raise
        with pytest.raises(ValueError, match="Maximum roles.*already reached"):
            architect._apply_mutation(
                MutationType.ADD_ROLE,
                None,
                mock_response,
                sample_sop_config,
                sample_prompt_genome,
                constraints=constraints,
            )

    def test_remove_role_violates_mandatory_constraint_integration(
        self, sample_sop_config, sample_prompt_genome
    ):
        """Test that REMOVE_ROLE respects mandatory_roles constraint"""
        architect = Architect(llm_provider=MockLLMProvider("test"))

        constraints = {"mandatory_roles": ["researcher"]}

        # Should raise
        with pytest.raises(ValueError, match="Cannot remove role.*mandatory"):
            architect._apply_mutation(
                MutationType.REMOVE_ROLE,
                "researcher",
                "researcher",
                sample_sop_config,
                sample_prompt_genome,
                constraints=constraints,
            )
