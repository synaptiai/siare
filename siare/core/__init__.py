"""SIARE Core - Data models, hooks, and configuration."""

from siare.core.hooks import (
    EvaluationHooks,
    EvolutionHooks,
    ExecutionHooks,
    HookContext,
    HookRegistry,
    HookRunner,
    LLMHooks,
    StorageHooks,
    fire_evaluation_hook,
    fire_evolution_hook,
    fire_execution_hook,
    fire_llm_hook,
    fire_storage_hook,
)
from siare.core.models import (
    Diagnosis,
    EvaluationVector,
    GraphEdge,
    MetricConfig,
    MutationType,
    ProcessConfig,
    PromptGenome,
    Role,
    SOPGene,
    Task,
)

__all__ = [
    # Models
    "ProcessConfig",
    "Role",
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
    "fire_evolution_hook",
    "fire_execution_hook",
    "fire_evaluation_hook",
    "fire_storage_hook",
    "fire_llm_hook",
]
