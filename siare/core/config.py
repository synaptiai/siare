"""Configuration classes for SIARE evolution parameters

These dataclasses externalize magic numbers from the codebase to enable:
- Runtime configuration without code changes
- A/B testing of different parameter values
- Domain-specific tuning
- Clear documentation of parameter meanings and defaults
"""

from dataclasses import dataclass, field


@dataclass
class ConvergenceConfig:
    """Settings for evolution convergence detection.

    Controls when evolution should stop based on quality improvements.
    Used by EvolutionScheduler._should_stop() and _update_best_sop().

    Attributes:
        quality_tie_threshold: Minimum quality difference to prefer newer SOP.
            If two SOPs differ by less than this, prefer the more recent one.
        convergence_window: Number of recent generations to examine for
            convergence detection. Larger values are more patient with plateaus.
        convergence_threshold: Maximum improvement rate required to continue.
            Evolution stops if recent window shows less improvement than this.
    """

    quality_tie_threshold: float = 0.001
    convergence_window: int = 20
    convergence_threshold: float = 0.01


@dataclass
class TextGradConfig:
    """Settings for TextGrad prompt evolution strategy.

    Controls how textual gradients are computed and applied.
    Used by TextGradStrategy.

    Attributes:
        severity_weights: Multipliers for different failure pattern types.
            Higher weight = more important to fix. Default values:
            - safety_violation: 2.0 (highest - critical to address)
            - hallucination: 1.5 (high - factual accuracy)
            - reasoning_error: 1.3 (moderate-high)
            - tool_misuse: 1.2 (moderate)
            - context_loss: 1.1 (low-moderate)
            - incomplete: 1.0 (baseline)
            - irrelevant: 1.0 (baseline)
            - timeout: 0.8 (penalty - infrastructure issue)
            - format_error: 0.7 (penalty - often fixable easily)
        llm_temperature: Temperature for gradient generation LLM calls.
            Lower = more deterministic, higher = more creative.
    """

    severity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "hallucination": 1.5,
            "safety_violation": 2.0,
            "reasoning_error": 1.3,
            "incomplete": 1.0,
            "irrelevant": 1.0,
            "timeout": 0.8,
            "tool_misuse": 1.2,
            "format_error": 0.7,
            "context_loss": 1.1,
        }
    )
    llm_temperature: float = 0.5


@dataclass
class QDGridConfig:
    """Settings for Quality-Diversity grid.

    Controls how SOPs are mapped to grid cells based on complexity and diversity.
    Used by QDGridManager.

    Attributes:
        complexity_weights: Relative importance of each complexity factor.
            Must sum to 1.0 for normalized complexity score.
            - num_roles: Weight for number of roles (default 0.4)
            - avg_depth: Weight for average DAG depth (default 0.3)
            - num_edges: Weight for number of graph edges (default 0.2)
            - avg_prompt: Weight for average prompt length (default 0.1)
        normalization_factors: Expected maximum values for each component.
            Used to normalize raw values to [0, 1] range.
        embedding_range: Expected range of PCA-reduced embeddings.
            Used for binning embeddings into grid cells.
        complexity_bins: Number of bins for complexity dimension.
        embedding_bins: Number of bins for each embedding dimension.
    """

    complexity_weights: dict[str, float] = field(
        default_factory=lambda: {
            "num_roles": 0.4,
            "avg_depth": 0.3,
            "num_edges": 0.2,
            "avg_prompt": 0.1,
        }
    )
    normalization_factors: dict[str, float] = field(
        default_factory=lambda: {
            "max_roles": 10.0,
            "max_depth": 5.0,
            "max_edges": 20.0,
            "max_prompt_length": 2000.0,
        }
    )
    embedding_range: tuple[float, float] = (-3.0, 3.0)
    complexity_bins: int = 10
    embedding_bins: int = 10


@dataclass
class EvoPromptConfig:
    """Settings for EvoPrompt evolution strategy.

    Controls genetic algorithm parameters for prompt evolution.
    Used by EvoPromptStrategy.

    Attributes:
        elite_ratio: Fraction of population to keep as elites (no mutation).
            Higher = more exploitation, lower = more exploration.
        fitness_decay: Multiplier applied to parent fitness for offspring.
            Less than 1.0 penalizes inherited fitness to encourage exploration.
        crossover_probability: Probability of crossover vs mutation.
            Higher = more recombination of existing prompts.
        llm_crossover_temperature: Temperature for crossover LLM calls.
        llm_mutation_temperature: Temperature for mutation LLM calls.
            Slightly higher than crossover for more variation.
    """

    elite_ratio: float = 0.25
    fitness_decay: float = 0.9
    crossover_probability: float = 0.5
    llm_crossover_temperature: float = 0.7
    llm_mutation_temperature: float = 0.8
