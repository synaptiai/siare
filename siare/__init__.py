"""
SIARE - Self-Improving Agentic RAG Engine

A Python library for building self-evolving multi-agent RAG systems using
Quality-Diversity optimization and evolutionary algorithms.

Example:
    >>> from siare import ProcessConfig, DirectorService, ExecutionEngine
    >>>
    >>> # Create a simple RAG pipeline configuration
    >>> config = ProcessConfig(name="my-rag", roles=[...], graph=[...])
    >>>
    >>> # Run evolution to optimize the pipeline
    >>> director = DirectorService(llm_provider)
    >>> improved_config = await director.mutate_sop(config, diagnosis)
"""

from siare.core.models import (
    Diagnosis,
    EvaluationVector,
    ExecutionTrace,
    GraphEdge,
    MetricConfig,
    MutationType,
    ProcessConfig,
    PromptGenome,
    Role,
    RoleOutput,
    SOPGene,
    Task,
)
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

__version__ = "1.0.0"
__author__ = "Daniel Bentes"
__license__ = "MIT"

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core models
    "ProcessConfig",
    "Role",
    "GraphEdge",
    "PromptGenome",
    "Task",
    "EvaluationVector",
    "ExecutionTrace",
    "RoleOutput",
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
