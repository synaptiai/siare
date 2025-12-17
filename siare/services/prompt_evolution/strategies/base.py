"""
Base Prompt Optimization Strategy

Abstract base class for all prompt optimization strategies.
Follows the pattern established in siare/services/selection/strategies.py.
"""

from abc import ABC, abstractmethod
from typing import Any, Union

from siare.core.models import (
    Diagnosis,
    EvoPromptConfig,
    MetaPromptConfig,
    ParsedPrompt,
    ProcessConfig,
    PromptEvolutionResult,
    PromptFeedback,
    PromptGenome,
    PromptOptimizationStrategyType,
    TextGradConfig,
)


class BasePromptOptimizationStrategy(ABC):
    """
    Abstract base class for prompt optimization strategies.

    All strategies must implement:
    - name: Strategy identifier
    - optimize(): Main optimization method
    - requires_population(): Whether strategy needs multiple candidates

    Strategies receive:
    - Current SOP config and prompt genome
    - Structured feedback from LLM critic
    - Diagnosis from Diagnostician
    - Constraints to respect (mustNotChange, etc.)

    Strategies return:
    - PromptEvolutionResult with new genome and metadata
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Strategy name for logging and identification.

        Returns:
            Strategy name (e.g., "textgrad", "evoprompt", "metaprompt")
        """

    @property
    @abstractmethod
    def strategy_type(self) -> PromptOptimizationStrategyType:
        """
        Strategy type enum value.

        Returns:
            PromptOptimizationStrategyType enum
        """

    @abstractmethod
    def optimize(
        self,
        sop_config: ProcessConfig,
        prompt_genome: PromptGenome,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        parsed_prompts: dict[str, ParsedPrompt] | None = None,
        constraints: dict[str, Any] | None = None,
    ) -> PromptEvolutionResult:
        """
        Apply optimization strategy to evolve prompts.

        Args:
            sop_config: Current SOP configuration
            prompt_genome: Current prompt genome to optimize
            feedback: Structured feedback from LLM critic
            diagnosis: Diagnosis from Diagnostician
            parsed_prompts: Optional pre-parsed prompts (role_id -> ParsedPrompt)
            constraints: Evolution constraints (mustNotChange, etc.)

        Returns:
            PromptEvolutionResult with new genome and metadata
        """

    @abstractmethod
    def requires_population(self) -> bool:
        """
        Whether this strategy needs multiple candidates.

        EvoPrompt: True (maintains population of variants)
        TextGrad, MetaPrompt: False (single candidate optimization)

        Returns:
            True if strategy needs population management
        """

    def validate_constraints(
        self,
        original_content: str,
        new_content: str,
        must_not_change: list[str] | None = None,
    ) -> list[str]:
        """
        Validate that new content respects constraints.

        Args:
            original_content: Original prompt content
            new_content: Proposed new content
            must_not_change: Protected text segments

        Returns:
            List of constraint violations (empty if valid)
        """
        violations: list[str] = []
        if must_not_change:
            for protected_text in must_not_change:
                if protected_text in original_content and protected_text not in new_content:
                    violations.append(f"Removed protected text: '{protected_text[:50]}...'")
        return violations


# Type alias for strategy configuration
StrategyConfig = Union[TextGradConfig, EvoPromptConfig, MetaPromptConfig]
