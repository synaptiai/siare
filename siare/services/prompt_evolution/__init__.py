"""
Prompt Evolution Package

Plugin-based prompt optimization system supporting multiple strategies:
- TextGrad: Textual gradient descent with LLM-generated critiques
- EvoPrompt: Evolutionary algorithms (GA/DE) for prompt populations
- MetaPrompt: LLM meta-analysis for targeted improvements
- Adaptive: Auto-selects optimal strategy based on failure patterns

Based on research:
- DSPy: https://github.com/stanfordnlp/dspy
- TextGrad: https://github.com/zou-group/textgrad (Nature, 2024)
- EvoPrompt: https://github.com/beeevita/EvoPrompt (ICLR 2024)
"""

from siare.services.prompt_evolution.constraint_validator import (
    BaseConstraintValidator,
    ConstraintValidator,
)
from siare.services.prompt_evolution.critic import (
    BaseLLMCritic,
    TraceFeedbackExtractor,
)
from siare.services.prompt_evolution.feedback_extractor import (
    BaseFeedbackExtractor,
    FeedbackArtifactExtractor,
)
from siare.services.prompt_evolution.feedback_injector import FeedbackInjector
from siare.services.prompt_evolution.orchestrator import PromptEvolutionOrchestrator
from siare.services.prompt_evolution.parser import (
    BasePromptSectionParser,
    LLMSectionParser,
    MarkdownSectionParser,
)
from siare.services.prompt_evolution.section_mutator import (
    BaseSectionMutator,
    SectionBasedPromptMutator,
)
from siare.services.prompt_evolution.selector import (
    AdaptiveStrategySelector,
    PromptOptimizationFactory,
)
from siare.services.prompt_evolution.strategies.base import (
    BasePromptOptimizationStrategy,
)
from siare.services.prompt_evolution.strategies.evoprompt import EvoPromptStrategy
from siare.services.prompt_evolution.strategies.metaprompt import MetaPromptStrategy
from siare.services.prompt_evolution.strategies.textgrad import TextGradStrategy

__all__ = [
    "AdaptiveStrategySelector",
    "BaseConstraintValidator",
    "BaseFeedbackExtractor",
    "BaseLLMCritic",
    "BasePromptOptimizationStrategy",
    "BasePromptSectionParser",
    "BaseSectionMutator",
    "ConstraintValidator",
    "EvoPromptStrategy",
    "FeedbackArtifactExtractor",
    "FeedbackInjector",
    "LLMSectionParser",
    "MarkdownSectionParser",
    "MetaPromptStrategy",
    "PromptEvolutionOrchestrator",
    "PromptOptimizationFactory",
    "SectionBasedPromptMutator",
    "TextGradStrategy",
    "TraceFeedbackExtractor",
]
