"""
SIARE - Self-Improving Agentic RAG Engine

A Python library for building self-evolving multi-agent RAG systems using
Quality-Diversity optimization and evolutionary algorithms.

Quick Start:
    >>> from siare.builders import pipeline, role, edge, task
    >>>
    >>> config, genome = pipeline(
    ...     "my-rag",
    ...     roles=[
    ...         role("retriever", "gpt-4o-mini", "You are a retriever...", tools=["vector_search"]),
    ...         role("answerer", "gpt-4o-mini", "You answer questions..."),
    ...     ],
    ...     edges=[edge("retriever", "answerer")],
    ... )

For advanced usage, use the core models directly:
    >>> from siare import ProcessConfig, RoleConfig, GraphEdge, PromptGenome
"""

from siare.builders import edge, pipeline, role, task
from siare.core.hooks import (
    EvaluationHooks,
    EvolutionHooks,
    ExecutionHooks,
    HookContext,
    HookRegistry,
    HookRunner,
    LLMHooks,
    StorageHooks,
)
from siare.core.models import (
    Diagnosis,
    EvaluationVector,
    GraphEdge,
    MetricConfig,
    MutationType,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
    SOPGene,
    Task,
)

__version__ = "1.0.0"
__author__ = "Daniel Bentes"
__license__ = "MIT"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Builders (recommended for new users)
    "pipeline",
    "role",
    "edge",
    "task",
    # Core models (for advanced usage)
    "ProcessConfig",
    "RoleConfig",
    "RolePrompt",
    "GraphEdge",
    "PromptGenome",
    "Task",
    "EvaluationVector",
    "Diagnosis",
    "MutationType",
    "MetricConfig",
    "SOPGene",
    # Hooks
    "HookContext",
    "HookRegistry",
    "HookRunner",
    "EvolutionHooks",
    "ExecutionHooks",
    "EvaluationHooks",
    "StorageHooks",
    "LLMHooks",
]
