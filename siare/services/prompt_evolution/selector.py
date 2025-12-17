"""
Adaptive Strategy Selector

Auto-selects the optimal prompt optimization strategy based on:
- Failure patterns in feedback
- Diagnosis characteristics
- Historical performance data

Strategy selection rules:
- Hallucination/reasoning errors → TextGrad (precise surgical fixes)
- Need diversity/exploration → EvoPrompt (population-based search)
- Quick targeted fix → MetaPrompt (fast single-shot improvement)
"""

from collections import Counter, defaultdict
from typing import Any

from siare.core.models import (
    Diagnosis,
    EvoPromptConfig,
    FailurePattern,
    MetaPromptConfig,
    ParsedPrompt,
    ProcessConfig,
    PromptEvolutionResult,
    PromptFeedback,
    PromptGenome,
    PromptOptimizationStrategyType,
    TextGradConfig,
)
from siare.services.llm_provider import LLMProvider
from siare.services.prompt_evolution.strategies.base import (
    BasePromptOptimizationStrategy,
)
from siare.services.prompt_evolution.strategies.evoprompt import EvoPromptStrategy
from siare.services.prompt_evolution.strategies.metaprompt import MetaPromptStrategy
from siare.services.prompt_evolution.strategies.textgrad import TextGradStrategy

# Constants
MAX_SELECTION_HISTORY = 100

# Strategy recommendations based on failure patterns
FAILURE_PATTERN_TO_STRATEGY: dict[FailurePattern, PromptOptimizationStrategyType] = {
    # TextGrad: For precise, surgical fixes
    FailurePattern.HALLUCINATION: PromptOptimizationStrategyType.TEXTGRAD,
    FailurePattern.REASONING_ERROR: PromptOptimizationStrategyType.TEXTGRAD,
    FailurePattern.SAFETY_VIOLATION: PromptOptimizationStrategyType.TEXTGRAD,

    # EvoPrompt: For exploration and diverse failures
    FailurePattern.INCOMPLETE: PromptOptimizationStrategyType.EVOPROMPT,
    FailurePattern.IRRELEVANT: PromptOptimizationStrategyType.EVOPROMPT,
    FailurePattern.CONTEXT_LOSS: PromptOptimizationStrategyType.EVOPROMPT,

    # MetaPrompt: For quick, targeted fixes
    FailurePattern.FORMAT_ERROR: PromptOptimizationStrategyType.METAPROMPT,
    FailurePattern.TIMEOUT: PromptOptimizationStrategyType.METAPROMPT,
    FailurePattern.TOOL_MISUSE: PromptOptimizationStrategyType.METAPROMPT,
}


class AdaptiveStrategySelector(BasePromptOptimizationStrategy):
    """
    Adaptive selector that chooses the optimal strategy per mutation.

    Selection criteria:
    1. Analyze failure patterns in feedback
    2. Consider diagnosis characteristics
    3. Apply weighted voting based on pattern severity
    4. Return the strategy with highest vote count

    Can also be used as a strategy itself (wraps the selected strategy).
    """

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        textgrad_config: TextGradConfig | None = None,
        evoprompt_config: EvoPromptConfig | None = None,
        metaprompt_config: MetaPromptConfig | None = None,
        default_strategy: PromptOptimizationStrategyType = PromptOptimizationStrategyType.EVOPROMPT,
    ):
        """
        Initialize the adaptive selector.

        Args:
            llm_provider: LLM provider passed to strategies
            textgrad_config: Config for TextGrad strategy
            evoprompt_config: Config for EvoPrompt strategy
            metaprompt_config: Config for MetaPrompt strategy
            default_strategy: Default strategy when selection is ambiguous
        """
        self.llm_provider = llm_provider
        self.default_strategy = default_strategy

        # Initialize all strategies
        self._strategies: dict[PromptOptimizationStrategyType, BasePromptOptimizationStrategy] = {
            PromptOptimizationStrategyType.TEXTGRAD: TextGradStrategy(
                llm_provider=llm_provider,
                config=textgrad_config,
            ),
            PromptOptimizationStrategyType.EVOPROMPT: EvoPromptStrategy(
                llm_provider=llm_provider,
                config=evoprompt_config,
            ),
            PromptOptimizationStrategyType.METAPROMPT: MetaPromptStrategy(
                llm_provider=llm_provider,
                config=metaprompt_config,
            ),
        }

        # Track selection history for analysis
        self._selection_history: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "adaptive"

    @property
    def strategy_type(self) -> PromptOptimizationStrategyType:
        return self.default_strategy

    def requires_population(self) -> bool:
        return False

    def select_strategy(
        self,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
    ) -> PromptOptimizationStrategyType:
        """
        Select the optimal strategy based on feedback and diagnosis.

        Selection algorithm:
        1. Count failure patterns in feedback
        2. Weight by pattern severity and confidence
        3. Map patterns to recommended strategies
        4. Return strategy with highest weighted votes

        Args:
            feedback: Feedback from LLM critic
            diagnosis: Diagnosis from Diagnostician

        Returns:
            Selected strategy type
        """
        if not feedback:
            # No feedback - use default
            return self.default_strategy

        # Count weighted votes per strategy
        votes: dict[PromptOptimizationStrategyType, float] = defaultdict(float)

        for fb in feedback:
            if fb.failure_pattern:
                recommended = FAILURE_PATTERN_TO_STRATEGY.get(
                    fb.failure_pattern,
                    self.default_strategy,
                )
                # Weight by confidence
                votes[recommended] += fb.confidence

        # Add diagnosis-based votes
        diagnosis_votes = self._analyze_diagnosis(diagnosis)
        for strategy, weight in diagnosis_votes.items():
            votes[strategy] += weight

        if not votes:
            return self.default_strategy

        # Return strategy with highest votes
        selected = max(votes.keys(), key=lambda k: votes[k])

        # Record selection for history
        self._record_selection(feedback, diagnosis, selected, dict(votes))

        return selected

    def _analyze_diagnosis(
        self,
        diagnosis: Diagnosis,
    ) -> dict[PromptOptimizationStrategyType, float]:
        """
        Analyze diagnosis for additional strategy hints.

        Returns weighted votes based on diagnosis characteristics.
        """
        votes: dict[PromptOptimizationStrategyType, float] = {}

        primary = diagnosis.primaryWeakness.lower()
        root_cause = diagnosis.rootCauseAnalysis.lower()

        # TextGrad indicators
        textgrad_keywords = ["hallucination", "fabricat", "accuracy", "reasoning", "logic", "safety"]
        if any(kw in primary or kw in root_cause for kw in textgrad_keywords):
            votes[PromptOptimizationStrategyType.TEXTGRAD] = 0.5

        # EvoPrompt indicators
        evoprompt_keywords = ["exploration", "diverse", "coverage", "variation", "creative"]
        if any(kw in primary or kw in root_cause for kw in evoprompt_keywords):
            votes[PromptOptimizationStrategyType.EVOPROMPT] = 0.5

        # MetaPrompt indicators
        metaprompt_keywords = ["format", "structure", "simple", "minor", "quick", "tool"]
        if any(kw in primary or kw in root_cause for kw in metaprompt_keywords):
            votes[PromptOptimizationStrategyType.METAPROMPT] = 0.5

        return votes

    def _record_selection(
        self,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
        selected: PromptOptimizationStrategyType,
        votes: dict[PromptOptimizationStrategyType, float],
    ) -> None:
        """Record selection for history and analysis."""
        self._selection_history.append({
            "feedback_count": len(feedback),
            "failure_patterns": [f.failure_pattern.value if f.failure_pattern else None for f in feedback],
            "primary_weakness": diagnosis.primaryWeakness[:100],
            "selected": selected.value,
            "votes": {k.value: v for k, v in votes.items()},
        })

        # Keep only last selections
        if len(self._selection_history) > MAX_SELECTION_HISTORY:
            self._selection_history = self._selection_history[-MAX_SELECTION_HISTORY:]

    def get_strategy(
        self,
        strategy_type: PromptOptimizationStrategyType,
    ) -> BasePromptOptimizationStrategy:
        """Get a specific strategy by type."""
        return self._strategies[strategy_type]

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
        Automatically select and apply the optimal strategy.

        This method allows AdaptiveStrategySelector to be used as a strategy itself.

        Args:
            sop_config: Current SOP configuration
            prompt_genome: Current prompt genome
            feedback: Structured feedback from LLM critic
            diagnosis: Diagnosis from Diagnostician
            parsed_prompts: Pre-parsed prompts (optional)
            constraints: Evolution constraints

        Returns:
            PromptEvolutionResult from the selected strategy
        """
        # Select optimal strategy
        strategy_type = self.select_strategy(feedback, diagnosis)
        strategy = self._strategies[strategy_type]

        # Apply the strategy
        result = strategy.optimize(
            sop_config=sop_config,
            prompt_genome=prompt_genome,
            feedback=feedback,
            diagnosis=diagnosis,
            parsed_prompts=parsed_prompts,
            constraints=constraints,
        )

        # Add selection metadata
        result.strategy_metadata["adaptive_selection"] = {
            "selected_strategy": strategy_type.value,
            "selection_reason": self._get_selection_reason(feedback, diagnosis),
        }

        return result

    def _get_selection_reason(
        self,
        feedback: list[PromptFeedback],
        diagnosis: Diagnosis,
    ) -> str:
        """Generate human-readable reason for strategy selection."""
        if not feedback:
            return f"No feedback provided, using default ({self.default_strategy.value})"

        # Count patterns
        pattern_counts: Counter[str] = Counter()
        for fb in feedback:
            if fb.failure_pattern:
                pattern_counts[fb.failure_pattern.value] += 1

        if not pattern_counts:
            return f"No failure patterns identified, using default ({self.default_strategy.value})"

        top_pattern = pattern_counts.most_common(1)[0][0]
        return f"Dominant pattern: {top_pattern} ({pattern_counts[top_pattern]} occurrences)"

    def get_selection_history(self) -> list[dict[str, Any]]:
        """Get recent selection history for analysis."""
        return list(self._selection_history)

    def get_strategy_stats(self) -> dict[str, int]:
        """Get statistics on strategy usage."""
        stats: Counter[str] = Counter()
        for record in self._selection_history:
            stats[record["selected"]] += 1
        return dict(stats)


class PromptOptimizationFactory:
    """
    Factory for creating prompt optimization strategies.

    Follows the pattern established in siare/services/selection/factory.py.
    """

    @staticmethod
    def create(
        strategy_type: str,
        llm_provider: LLMProvider | None = None,
        config: dict[str, Any] | None = None,
    ) -> BasePromptOptimizationStrategy:
        """
        Create a prompt optimization strategy.

        Args:
            strategy_type: Strategy type string ("textgrad", "evoprompt", "metaprompt", "adaptive")
            llm_provider: LLM provider for strategy
            config: Strategy-specific configuration dict

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy type is unknown
        """
        config = config or {}

        if strategy_type == "textgrad":
            return TextGradStrategy(
                llm_provider=llm_provider,
                config=TextGradConfig(**config) if config else None,
            )

        if strategy_type == "evoprompt":
            return EvoPromptStrategy(
                llm_provider=llm_provider,
                config=EvoPromptConfig(**config) if config else None,
            )

        if strategy_type == "metaprompt":
            return MetaPromptStrategy(
                llm_provider=llm_provider,
                config=MetaPromptConfig(**config) if config else None,
            )

        if strategy_type == "adaptive":
            return AdaptiveStrategySelector(
                llm_provider=llm_provider,
                textgrad_config=TextGradConfig(**config.get("textgrad", {})) if config.get("textgrad") else None,
                evoprompt_config=EvoPromptConfig(**config.get("evoprompt", {})) if config.get("evoprompt") else None,
                metaprompt_config=MetaPromptConfig(**config.get("metaprompt", {})) if config.get("metaprompt") else None,
            )

        raise ValueError(
            f"Unknown strategy type: {strategy_type}. "
            f"Available: textgrad, evoprompt, metaprompt, adaptive"
        )

    @staticmethod
    def available_strategies() -> list[str]:
        """Get list of available strategy types."""
        return ["textgrad", "evoprompt", "metaprompt", "adaptive"]
