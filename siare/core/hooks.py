"""
SIARE Hooks System - Extension points for customization and enterprise features.

This module provides the hooks architecture that allows external code to observe
and modify SIARE's behavior without modifying core code. Hooks are optional -
SIARE works fully without any hooks registered.

Design Principles:
    1. Core is unaware of cloud/enterprise - hooks are optional
    2. Zero performance penalty when unused - NoOp by default
    3. Async by default, support both - sync hooks work too
    4. Errors don't stop execution - log and continue
    5. Composable - multiple hook implementations can chain

Example Usage:
    >>> from siare.core.hooks import HookRegistry, HookContext
    >>>
    >>> class MyEvolutionHooks:
    ...     async def on_mutation_complete(self, ctx, original, mutated, mutation_type):
    ...         print(f"Mutation completed: {mutation_type}")
    >>>
    >>> HookRegistry.set_evolution_hooks(MyEvolutionHooks())
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from siare.core.models import (
        Diagnosis,
        EvaluationVector,
        MutationType,
        ProcessConfig,
        SOPGene,
    )
    from siare.services.execution_engine import ExecutionTrace


logger = logging.getLogger(__name__)


@dataclass
class HookContext:
    """Context passed to all hooks for correlation and metadata.

    Attributes:
        correlation_id: Unique ID for tracing related operations.
        timestamp: Unix timestamp when the context was created.
        metadata: Arbitrary key-value pairs for custom data.
        tenant_id: Optional tenant identifier for multi-tenant scenarios.
    """
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    tenant_id: str | None = None

    def with_metadata(self, **kwargs: Any) -> HookContext:
        """Create a new context with additional metadata."""
        return HookContext(
            correlation_id=self.correlation_id,
            timestamp=self.timestamp,
            metadata={**self.metadata, **kwargs},
            tenant_id=self.tenant_id,
        )


# =============================================================================
# Hook Protocols - Define contracts for hook implementations
# =============================================================================


@runtime_checkable
class EvolutionHooks(Protocol):
    """Hooks for the evolutionary optimization process.

    These hooks fire during SOP mutation and generation cycles,
    enabling audit logging, approval gates, and custom logic.
    """

    async def on_mutation_start(
        self,
        ctx: HookContext,
        sop: ProcessConfig,
        diagnosis: Diagnosis,
    ) -> ProcessConfig | None:
        """Called before a mutation is applied.

        Args:
            ctx: Hook context with correlation ID and metadata.
            sop: The SOP about to be mutated.
            diagnosis: The diagnosis driving the mutation.

        Returns:
            Modified SOP to mutate instead, or None to proceed normally.
            Return the original SOP to block mutation.
        """
        ...

    async def on_mutation_complete(
        self,
        ctx: HookContext,
        original: ProcessConfig,
        mutated: ProcessConfig,
        mutation_type: MutationType,
    ) -> None:
        """Called after a mutation is successfully applied.

        Args:
            ctx: Hook context with correlation ID and metadata.
            original: The SOP before mutation.
            mutated: The SOP after mutation.
            mutation_type: The type of mutation that was applied.
        """
        ...

    async def on_generation_complete(
        self,
        ctx: HookContext,
        generation: int,
        population: list[SOPGene],
        best_fitness: float,
    ) -> bool:
        """Called after each generation of evolution completes.

        Args:
            ctx: Hook context with correlation ID and metadata.
            generation: The generation number (1-indexed).
            population: Current population of SOPs.
            best_fitness: Best fitness score in this generation.

        Returns:
            True to continue evolution, False to stop early.
        """
        ...


@runtime_checkable
class ExecutionHooks(Protocol):
    """Hooks for the execution engine.

    These hooks fire during SOP execution, enabling usage tracking,
    billing, and custom instrumentation.
    """

    async def on_execution_start(
        self,
        ctx: HookContext,
        sop: ProcessConfig,
        task_data: dict[str, Any],
    ) -> None:
        """Called when an SOP execution begins.

        Args:
            ctx: Hook context with correlation ID and metadata.
            sop: The SOP being executed.
            task_data: Input data for the task.
        """
        ...

    async def on_execution_complete(
        self,
        ctx: HookContext,
        sop: ProcessConfig,
        trace: ExecutionTrace,
        duration_ms: float,
    ) -> None:
        """Called when an SOP execution completes.

        Args:
            ctx: Hook context with correlation ID and metadata.
            sop: The SOP that was executed.
            trace: The full execution trace.
            duration_ms: Execution duration in milliseconds.
        """
        ...

    async def on_role_complete(
        self,
        ctx: HookContext,
        role_name: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        duration_ms: float,
    ) -> None:
        """Called when an individual role completes execution.

        Args:
            ctx: Hook context with correlation ID and metadata.
            role_name: Name of the role that completed.
            inputs: Inputs provided to the role.
            outputs: Outputs produced by the role.
            duration_ms: Role execution duration in milliseconds.
        """
        ...


@runtime_checkable
class EvaluationHooks(Protocol):
    """Hooks for the evaluation service.

    These hooks fire during metric evaluation, enabling custom metrics,
    alerting, and result aggregation.
    """

    async def on_evaluation_start(
        self,
        ctx: HookContext,
        trace: ExecutionTrace,
        metrics: list[str],
    ) -> None:
        """Called when evaluation of a trace begins.

        Args:
            ctx: Hook context with correlation ID and metadata.
            trace: The execution trace to evaluate.
            metrics: List of metric names to compute.
        """
        ...

    async def on_evaluation_complete(
        self,
        ctx: HookContext,
        trace: ExecutionTrace,
        evaluation: EvaluationVector,
    ) -> None:
        """Called when evaluation completes.

        Args:
            ctx: Hook context with correlation ID and metadata.
            trace: The evaluated execution trace.
            evaluation: The computed evaluation vector.
        """
        ...


@runtime_checkable
class StorageHooks(Protocol):
    """Hooks for SOP storage operations.

    These hooks fire during SOP persistence, enabling multi-tenant
    isolation, caching, and audit logging.
    """

    async def on_sop_saved(
        self,
        ctx: HookContext,
        sop: ProcessConfig,
        version: str,
    ) -> None:
        """Called when an SOP is saved.

        Args:
            ctx: Hook context with correlation ID and metadata.
            sop: The SOP that was saved.
            version: The version string assigned.
        """
        ...

    async def on_sop_loaded(
        self,
        ctx: HookContext,
        sop_id: str,
        version: str | None,
        sop: ProcessConfig,
    ) -> None:
        """Called when an SOP is loaded.

        Args:
            ctx: Hook context with correlation ID and metadata.
            sop_id: The SOP identifier.
            version: The version loaded, or None for latest.
            sop: The loaded SOP.
        """
        ...

    async def on_sop_deleted(
        self,
        ctx: HookContext,
        sop_id: str,
        version: str | None,
    ) -> None:
        """Called when an SOP is deleted.

        Args:
            ctx: Hook context with correlation ID and metadata.
            sop_id: The SOP identifier.
            version: The version deleted, or None for all versions.
        """
        ...


@runtime_checkable
class LLMHooks(Protocol):
    """Hooks for LLM provider interactions.

    These hooks fire during LLM API calls, enabling token tracking,
    cost billing, and request/response logging.
    """

    async def on_llm_request(
        self,
        ctx: HookContext,
        provider: str,
        model: str,
        messages: list[dict[str, Any]],
    ) -> None:
        """Called before an LLM request is made.

        Args:
            ctx: Hook context with correlation ID and metadata.
            provider: LLM provider name (openai, anthropic, etc.).
            model: Model identifier.
            messages: The messages being sent.
        """
        ...

    async def on_llm_response(
        self,
        ctx: HookContext,
        provider: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
    ) -> None:
        """Called after an LLM response is received.

        Args:
            ctx: Hook context with correlation ID and metadata.
            provider: LLM provider name.
            model: Model identifier.
            tokens_in: Input token count.
            tokens_out: Output token count.
            latency_ms: Request latency in milliseconds.
        """
        ...


# =============================================================================
# Hook Runner - Safely executes hooks with error handling
# =============================================================================


class HookRunner:
    """Safely runs hook functions with sync/async handling and error isolation.

    This class ensures that hook failures never break core functionality.
    All hook errors are logged but not propagated.

    Example:
        >>> result = await HookRunner.run(
        ...     hooks.on_mutation_complete,
        ...     ctx, original, mutated, mutation_type
        ... )
    """

    @staticmethod
    async def run(
        hook_fn: Any | None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """Run a hook function safely, handling sync/async and errors.

        Args:
            hook_fn: The hook function to call, or None to skip.
            *args: Positional arguments to pass to the hook.
            **kwargs: Keyword arguments to pass to the hook.

        Returns:
            The hook's return value, or None if hook is None or fails.
        """
        if hook_fn is None:
            return None

        try:
            result = hook_fn(*args, **kwargs)
            if asyncio.iscoroutine(result):
                return await result
            return result
        except Exception as e:
            fn_name = getattr(hook_fn, "__name__", str(hook_fn))
            logger.warning(f"Hook {fn_name} failed: {e}", exc_info=True)
            return None

    @staticmethod
    async def run_all(
        hook_fns: list[Any],
        *args: Any,
        **kwargs: Any,
    ) -> list[Any]:
        """Run multiple hook functions, collecting all results.

        Args:
            hook_fns: List of hook functions to call.
            *args: Positional arguments to pass to each hook.
            **kwargs: Keyword arguments to pass to each hook.

        Returns:
            List of results from hooks that succeeded.
        """
        results = []
        for hook_fn in hook_fns:
            result = await HookRunner.run(hook_fn, *args, **kwargs)
            if result is not None:
                results.append(result)
        return results


# =============================================================================
# Hook Registry - Global registration point for hooks
# =============================================================================


class HookRegistry:
    """Central registry for hook implementations.

    This class provides a global registration point for hooks. Enterprise
    features register their hooks here during initialization.

    Example:
        >>> from siare.core.hooks import HookRegistry
        >>>
        >>> class AuditHooks:
        ...     async def on_mutation_complete(self, ctx, original, mutated, mutation_type):
        ...         await audit_log.record("mutation", mutation_type)
        >>>
        >>> HookRegistry.set_evolution_hooks(AuditHooks())

    Thread Safety:
        Hook registration should happen during application startup.
        Once registered, hooks are read-only during operation.
    """

    _evolution_hooks: EvolutionHooks | None = None
    _execution_hooks: ExecutionHooks | None = None
    _evaluation_hooks: EvaluationHooks | None = None
    _storage_hooks: StorageHooks | None = None
    _llm_hooks: LLMHooks | None = None

    @classmethod
    def set_evolution_hooks(cls, hooks: EvolutionHooks) -> None:
        """Register evolution hooks."""
        cls._evolution_hooks = hooks

    @classmethod
    def get_evolution_hooks(cls) -> EvolutionHooks | None:
        """Get registered evolution hooks."""
        return cls._evolution_hooks

    @classmethod
    def set_execution_hooks(cls, hooks: ExecutionHooks) -> None:
        """Register execution hooks."""
        cls._execution_hooks = hooks

    @classmethod
    def get_execution_hooks(cls) -> ExecutionHooks | None:
        """Get registered execution hooks."""
        return cls._execution_hooks

    @classmethod
    def set_evaluation_hooks(cls, hooks: EvaluationHooks) -> None:
        """Register evaluation hooks."""
        cls._evaluation_hooks = hooks

    @classmethod
    def get_evaluation_hooks(cls) -> EvaluationHooks | None:
        """Get registered evaluation hooks."""
        return cls._evaluation_hooks

    @classmethod
    def set_storage_hooks(cls, hooks: StorageHooks) -> None:
        """Register storage hooks."""
        cls._storage_hooks = hooks

    @classmethod
    def get_storage_hooks(cls) -> StorageHooks | None:
        """Get registered storage hooks."""
        return cls._storage_hooks

    @classmethod
    def set_llm_hooks(cls, hooks: LLMHooks) -> None:
        """Register LLM hooks."""
        cls._llm_hooks = hooks

    @classmethod
    def get_llm_hooks(cls) -> LLMHooks | None:
        """Get registered LLM hooks."""
        return cls._llm_hooks

    @classmethod
    def clear_all(cls) -> None:
        """Clear all registered hooks (useful for testing)."""
        cls._evolution_hooks = None
        cls._execution_hooks = None
        cls._evaluation_hooks = None
        cls._storage_hooks = None
        cls._llm_hooks = None


# =============================================================================
# Convenience functions for firing hooks
# =============================================================================


async def fire_evolution_hook(
    hook_name: str,
    ctx: HookContext,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Fire an evolution hook by name.

    Args:
        hook_name: Name of the hook method (e.g., "on_mutation_complete").
        ctx: Hook context.
        *args: Arguments for the hook.
        **kwargs: Keyword arguments for the hook.

    Returns:
        Hook result, or None if no hooks registered.
    """
    hooks = HookRegistry.get_evolution_hooks()
    if hooks is None:
        return None
    hook_fn = getattr(hooks, hook_name, None)
    return await HookRunner.run(hook_fn, ctx, *args, **kwargs)


async def fire_execution_hook(
    hook_name: str,
    ctx: HookContext,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Fire an execution hook by name."""
    hooks = HookRegistry.get_execution_hooks()
    if hooks is None:
        return None
    hook_fn = getattr(hooks, hook_name, None)
    return await HookRunner.run(hook_fn, ctx, *args, **kwargs)


async def fire_evaluation_hook(
    hook_name: str,
    ctx: HookContext,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Fire an evaluation hook by name."""
    hooks = HookRegistry.get_evaluation_hooks()
    if hooks is None:
        return None
    hook_fn = getattr(hooks, hook_name, None)
    return await HookRunner.run(hook_fn, ctx, *args, **kwargs)


async def fire_storage_hook(
    hook_name: str,
    ctx: HookContext,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Fire a storage hook by name."""
    hooks = HookRegistry.get_storage_hooks()
    if hooks is None:
        return None
    hook_fn = getattr(hooks, hook_name, None)
    return await HookRunner.run(hook_fn, ctx, *args, **kwargs)


async def fire_llm_hook(
    hook_name: str,
    ctx: HookContext,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Fire an LLM hook by name."""
    hooks = HookRegistry.get_llm_hooks()
    if hooks is None:
        return None
    hook_fn = getattr(hooks, hook_name, None)
    return await HookRunner.run(hook_fn, ctx, *args, **kwargs)
