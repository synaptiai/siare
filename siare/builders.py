"""
SIARE Builders - Simplified API for creating pipelines

This module provides a clean, developer-friendly API for building SIARE pipelines.
It wraps the internal models with sensible defaults and intuitive parameter names.

Example:
    >>> from siare.builders import pipeline, role, edge, task
    >>>
    >>> config, genome = pipeline(
    ...     "my-rag",
    ...     roles=[
    ...         role("retriever", "gpt-4o-mini", "You are a retriever...", tools=["vector_search"]),
    ...         role("answerer", "gpt-4o-mini", "You are an answerer..."),
    ...     ],
    ...     edges=[
    ...         edge("retriever", "answerer"),
    ...     ],
    ... )
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
    Task,
)


def role(
    name: str,
    model: str,
    prompt: str,
    *,
    tools: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[RoleConfig, RolePrompt]:
    """Create a role with its prompt in one call.

    Args:
        name: Unique identifier for this role (e.g., "retriever", "answerer")
        model: Model to use (e.g., "gpt-4o-mini", "gpt-4o")
        prompt: System prompt for this role
        tools: Optional list of tool names this role can use
        params: Optional parameters (e.g., temperature)

    Returns:
        Tuple of (RoleConfig, RolePrompt) for internal use by pipeline()

    Example:
        >>> retriever = role(
        ...     "retriever",
        ...     "gpt-4o-mini",
        ...     "You are a document retrieval specialist...",
        ...     tools=["vector_search"],
        ... )
    """
    prompt_ref = f"{name}_prompt"

    role_config = RoleConfig(
        id=name,
        model=model,
        tools=tools,
        promptRef=prompt_ref,
        params=params,
    )

    role_prompt = RolePrompt(
        id=prompt_ref,
        content=prompt,
    )

    return role_config, role_prompt


def edge(
    source: str,
    target: str,
    *,
    condition: str | None = None,
) -> GraphEdge:
    """Create a graph edge connecting two roles.

    Args:
        source: Name of the source role (or "user_input" for entry point)
        target: Name of the target role
        condition: Optional condition for conditional execution
                   (e.g., "'search' in output")

    Returns:
        GraphEdge instance

    Example:
        >>> # Simple edge
        >>> edge("retriever", "answerer")

        >>> # Conditional edge
        >>> edge("router", "searcher", condition="'search' in output")
    """
    return GraphEdge(
        from_=source,
        to=target,
        condition=condition,
    )


def pipeline(
    name: str,
    roles: list[tuple[RoleConfig, RolePrompt]],
    edges: list[GraphEdge],
    *,
    version: str = "1.0.0",
    description: str | None = None,
    models: dict[str, str] | None = None,
    tools: list[str] | None = None,
    entry_point: str | None = None,
) -> tuple[ProcessConfig, PromptGenome]:
    """Create a complete pipeline configuration.

    Args:
        name: Pipeline identifier (e.g., "customer-support-rag")
        roles: List of roles created with role()
        edges: List of edges created with edge()
        version: Semantic version (default: "1.0.0")
        description: Optional description
        models: Optional model aliases (e.g., {"fast": "gpt-4o-mini"})
        tools: Optional list of all tools used in pipeline
        entry_point: First role to receive input (default: first role in list)

    Returns:
        Tuple of (ProcessConfig, PromptGenome)

    Example:
        >>> config, genome = pipeline(
        ...     "my-rag",
        ...     roles=[
        ...         role("retriever", "gpt-4o-mini", "You are...", tools=["vector_search"]),
        ...         role("answerer", "gpt-4o-mini", "You are..."),
        ...     ],
        ...     edges=[
        ...         edge("retriever", "answerer"),
        ...     ],
        ... )
    """
    # Extract role configs and prompts
    role_configs = [r[0] for r in roles]
    role_prompts = {r[1].id: r[1] for r in roles}

    # Collect all tools if not explicitly provided
    if tools is None:
        tools = []
        for rc in role_configs:
            if rc.tools:
                tools.extend(t for t in rc.tools if t not in tools)

    # Set default models if not provided
    if models is None:
        # Extract unique models from roles
        unique_models = set(rc.model for rc in role_configs)
        models = {"default": list(unique_models)[0]} if unique_models else {}

    # Add entry point edge if needed
    final_edges = list(edges)
    if entry_point is None and role_configs:
        entry_point = role_configs[0].id

    # Check if entry edge already exists
    has_entry = any(e.from_ == "user_input" for e in final_edges)
    if not has_entry and entry_point:
        final_edges.insert(0, GraphEdge(from_="user_input", to=entry_point))

    config = ProcessConfig(
        id=name,
        version=version,
        description=description,
        models=models,
        tools=tools,
        roles=role_configs,
        graph=final_edges,
    )

    genome = PromptGenome(
        id=f"{name}_genome",
        version=version,
        rolePrompts=role_prompts,
    )

    return config, genome


def task(
    query: str,
    *,
    id: str | None = None,
    expected: str | dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Task:
    """Create a task for pipeline execution.

    Args:
        query: The user query or input
        id: Optional task ID (auto-generated if not provided)
        expected: Optional expected answer (for evaluation)
        metadata: Optional metadata

    Returns:
        Task instance

    Example:
        >>> t = task("How do I reset my password?")

        >>> # With expected answer for evaluation
        >>> t = task(
        ...     "How do I reset my password?",
        ...     expected="Go to Settings > Security > Reset Password",
        ... )
    """
    task_id = id or str(uuid4())[:8]

    ground_truth = None
    if expected is not None:
        if isinstance(expected, str):
            ground_truth = {"answer": expected}
        else:
            ground_truth = expected

    return Task(
        id=task_id,
        input={"query": query},
        groundTruth=ground_truth,
    )


__all__ = [
    "role",
    "edge",
    "pipeline",
    "task",
]
