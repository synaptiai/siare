"""Variation tools for the agentic director.

These tools are available to the AgenticDirector during multi-turn
variation sessions. Each tool provides a specific capability:
inspecting traces, comparing SOPs, querying the gene pool, dry-running
candidates, validating mutations, and querying the knowledge base.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from siare.core.models import InnerLoopBudget, VariationToolSpec
from siare.services.agent_session import ToolExecutor

if TYPE_CHECKING:
    from siare.services.execution_engine import ExecutionEngine
    from siare.services.gene_pool import GenePool
    from siare.services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


MAX_TOOL_OUTPUT_CHARS = 4000


class VariationTool(ABC):
    """Base class for tools available during agentic variation."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in tool_call JSON."""

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description for the LLM."""

    @property
    def parameters(self) -> dict[str, Any]:
        """JSON Schema for tool parameters."""
        return {}

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> str:
        """Execute the tool and return a string result."""

    def to_spec(self) -> VariationToolSpec:
        """Convert to VariationToolSpec for AgentSession."""
        return VariationToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.parameters,
        )


# ============================================================================
# Tool Implementations
# ============================================================================


class InspectTraceTool(VariationTool):
    """Inspect a specific execution trace in detail."""

    def __init__(self, traces: dict[str, Any] | None = None) -> None:
        self._traces = traces or {}

    @property
    def name(self) -> str:
        return "inspect_trace"

    @property
    def description(self) -> str:
        return (
            "Inspect execution trace details: node outputs, timing, "
            "errors, tool calls. Use to understand why a pipeline "
            "performed poorly."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "trace_id": {
                "type": "string",
                "description": "The run_id of the trace to inspect",
            },
            "focus": {
                "type": "string",
                "enum": ["errors", "outputs", "timing", "full"],
                "description": "Aspect to focus on (default: full)",
            },
        }

    def set_traces(self, traces: dict[str, Any]) -> None:
        """Update available traces."""
        self._traces = traces

    def execute(self, arguments: dict[str, Any]) -> str:
        trace_id = arguments.get("trace_id", "")
        focus = arguments.get("focus", "full")

        if trace_id not in self._traces:
            return f"Trace '{trace_id}' not found. Available: {list(self._traces.keys())[:5]}"

        trace_data = self._traces[trace_id]
        if isinstance(trace_data, dict):
            data = trace_data
        elif hasattr(trace_data, "model_dump"):
            data = trace_data.model_dump()
        else:
            return f"Trace '{trace_id}' has unsupported format."

        if focus == "errors":
            errors = data.get("errors", [])
            if not errors:
                return "No errors in this trace."
            return json.dumps(errors, indent=2, default=str)
        elif focus == "timing":
            nodes = data.get("node_executions", [])
            timing = [
                {
                    "role": n.get("role_id", "?"),
                    "duration_ms": n.get("duration_ms", 0),
                }
                for n in nodes
            ]
            return json.dumps(timing, indent=2)
        elif focus == "outputs":
            return json.dumps(
                data.get("final_outputs", {}), indent=2, default=str
            )
        else:
            return json.dumps(data, indent=2, default=str)[:4000]


class CompareSOPsTool(VariationTool):
    """Compare two SOP configurations to understand differences."""

    def __init__(self, gene_pool: GenePool | None = None) -> None:
        self._gene_pool = gene_pool

    @property
    def name(self) -> str:
        return "compare_sops"

    @property
    def description(self) -> str:
        return (
            "Compare two SOP configurations. Shows structural diffs "
            "(roles, edges), prompt diffs, and performance differences."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "sop_a_id": {"type": "string"},
            "sop_a_version": {"type": "string"},
            "sop_b_id": {"type": "string"},
            "sop_b_version": {"type": "string"},
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        if self._gene_pool is None:
            return "Gene pool not available."

        a_id = arguments.get("sop_a_id", "")
        a_ver = arguments.get("sop_a_version")
        b_id = arguments.get("sop_b_id", "")
        b_ver = arguments.get("sop_b_version")

        try:
            gene_a = self._gene_pool.get_sop_gene(a_id, a_ver)
            gene_b = self._gene_pool.get_sop_gene(b_id, b_ver)
        except Exception as e:
            return f"Failed to retrieve SOPs: {e}"

        if gene_a is None or gene_b is None:
            return "One or both SOPs not found."

        config_a = gene_a.configSnapshot
        config_b = gene_b.configSnapshot

        diff_parts: list[str] = []
        diff_parts.append(
            f"Comparing {a_id}@{gene_a.version} vs {b_id}@{gene_b.version}"
        )

        roles_a = {r.id for r in config_a.roles}
        roles_b = {r.id for r in config_b.roles}
        added = roles_b - roles_a
        removed = roles_a - roles_b
        if added:
            diff_parts.append(f"Roles added: {added}")
        if removed:
            diff_parts.append(f"Roles removed: {removed}")
        if not added and not removed:
            diff_parts.append(f"Same roles: {roles_a}")

        edges_a = {(e.from_, e.to) for e in config_a.graph}
        edges_b = {(e.from_, e.to) for e in config_b.graph}
        new_edges = edges_b - edges_a
        del_edges = edges_a - edges_b
        if new_edges:
            diff_parts.append(f"Edges added: {new_edges}")
        if del_edges:
            diff_parts.append(f"Edges removed: {del_edges}")

        qa = gene_a.get_metric_mean("weighted_aggregate")
        qb = gene_b.get_metric_mean("weighted_aggregate")
        diff_parts.append(f"Quality A: {qa:.3f}, Quality B: {qb:.3f}")

        return "\n".join(diff_parts)


class QueryGenePoolTool(VariationTool):
    """Query the gene pool for evolutionary patterns."""

    def __init__(self, gene_pool: GenePool | None = None) -> None:
        self._gene_pool = gene_pool

    @property
    def name(self) -> str:
        return "query_gene_pool"

    @property
    def description(self) -> str:
        return (
            "Query the gene pool for patterns: Pareto frontier, "
            "top performers, diversity stats, lineage, or unexplored "
            "regions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "query_type": {
                "type": "string",
                "enum": [
                    "pareto_frontier",
                    "top_performers",
                    "diversity_stats",
                    "generation_stats",
                ],
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results (default: 5)",
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        if self._gene_pool is None:
            return "Gene pool not available."

        query_type = arguments.get("query_type", "diversity_stats")
        top_k = arguments.get("top_k", 5)

        try:
            if query_type == "pareto_frontier":
                frontier = self._gene_pool.get_pareto_frontier(
                    metric_ids=["weighted_aggregate"]
                )
                items = [
                    {
                        "sopId": g.sopId,
                        "version": g.version,
                        "quality": g.get_metric_mean("weighted_aggregate"),
                    }
                    for g in frontier[:top_k]
                ]
                return json.dumps(
                    {"pareto_frontier": items, "total": len(frontier)},
                    indent=2,
                )

            elif query_type == "top_performers":
                genes = self._gene_pool.list_sop_genes(
                    pareto_optimal_only=False
                )
                sorted_genes = sorted(
                    genes,
                    key=lambda g: g.get_metric_mean("weighted_aggregate"),
                    reverse=True,
                )
                items = [
                    {
                        "sopId": g.sopId,
                        "version": g.version,
                        "quality": g.get_metric_mean("weighted_aggregate"),
                        "generation": g.generation,
                    }
                    for g in sorted_genes[:top_k]
                ]
                return json.dumps({"top_performers": items}, indent=2)

            elif query_type == "diversity_stats":
                stats = self._gene_pool.get_diversity_stats()
                return json.dumps(stats, indent=2, default=str)

            elif query_type == "generation_stats":
                stats = self._gene_pool.get_generation_stats()
                return json.dumps(stats, indent=2, default=str)

            else:
                return f"Unknown query type: {query_type}"

        except Exception as e:
            return f"Query failed: {e}"


class DryRunTool(VariationTool):
    """Execute a candidate SOP on a sample task for quick evaluation."""

    def __init__(
        self,
        execution_engine: ExecutionEngine | None = None,
        budget: InnerLoopBudget | None = None,
    ) -> None:
        self._engine = execution_engine
        self._budget = budget

    @property
    def name(self) -> str:
        return "dry_run"

    @property
    def description(self) -> str:
        return (
            "Execute a candidate SOP on a sample task and return "
            "the output. Budget-limited — use sparingly. Call "
            "validate_mutation first to catch obvious issues."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "config_json": {
                "type": "string",
                "description": "ProcessConfig as JSON",
            },
            "genome_json": {
                "type": "string",
                "description": "PromptGenome as JSON",
            },
            "task_input": {
                "type": "object",
                "description": "Task input data",
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        if self._engine is None:
            return "Execution engine not available."
        if self._budget and self._budget.dryRunsUsed >= self._budget.maxDryRuns:
            return "Dry-run budget exhausted."

        try:
            from siare.core.models import ProcessConfig, PromptGenome

            config = ProcessConfig.model_validate_json(
                arguments.get("config_json", "{}")
            )
            genome = PromptGenome.model_validate_json(
                arguments.get("genome_json", "{}")
            )
            task_input = arguments.get("task_input", {"query": "test"})

            trace = self._engine.execute(config, genome, task_input)

            if self._budget:
                self._budget.record_dry_run(cost=trace.total_cost)

            return json.dumps({
                "status": trace.status,
                "outputs": trace.final_outputs,
                "duration_ms": (
                    (trace.end_time - trace.start_time).total_seconds() * 1000
                    if trace.end_time
                    else None
                ),
                "errors": [
                    e.get("error", "unknown") for e in trace.errors
                ],
                "cost": trace.total_cost,
            }, indent=2, default=str)

        except Exception as e:
            return f"Dry-run failed: {e}"


class ValidateMutationTool(VariationTool):
    """Pre-validate a proposed mutation against constraints."""

    def __init__(
        self,
        constraints: dict[str, Any] | None = None,
    ) -> None:
        self._constraints = constraints or {}

    @property
    def name(self) -> str:
        return "validate_mutation"

    @property
    def description(self) -> str:
        return (
            "Validate a mutation against evolution constraints before "
            "dry-running. Checks: max roles, mandatory roles, disallowed "
            "mutation types, DAG integrity."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "mutation_type": {"type": "string"},
            "target_role": {"type": "string"},
            "num_roles_after": {"type": "integer"},
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        violations: list[str] = []
        mutation_type = arguments.get("mutation_type", "")
        target_role = arguments.get("target_role", "")
        num_roles = arguments.get("num_roles_after", 0)

        disallowed = self._constraints.get("disallowedMutationTypes", [])
        if mutation_type in disallowed:
            violations.append(
                f"Mutation type '{mutation_type}' is disallowed."
            )

        max_roles = self._constraints.get("maxRoles")
        if max_roles and num_roles > max_roles:
            violations.append(
                f"Would exceed max roles ({num_roles} > {max_roles})."
            )

        mandatory = self._constraints.get("mandatoryRoles", [])
        if mutation_type == "remove_role" and target_role in mandatory:
            violations.append(
                f"Cannot remove mandatory role '{target_role}'."
            )

        if violations:
            return "INVALID: " + "; ".join(violations)
        return "VALID: Mutation passes all constraint checks."


class QueryKnowledgeBaseTool(VariationTool):
    """Query the knowledge base for domain-specific guidance."""

    def __init__(
        self, knowledge_base: KnowledgeBase | None = None
    ) -> None:
        self._kb = knowledge_base

    @property
    def name(self) -> str:
        return "query_knowledge_base"

    @property
    def description(self) -> str:
        return (
            "Query the knowledge base for RAG patterns, prompt "
            "engineering techniques, prior evolution run learnings, "
            "or domain-specific guidance."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "query": {"type": "string", "description": "Search query"},
            "category": {
                "type": "string",
                "enum": [
                    "rag_patterns",
                    "prompt_engineering",
                    "prior_runs",
                    "domain",
                ],
                "description": "Category filter (optional)",
            },
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        if self._kb is None:
            return "Knowledge base not available."

        query = arguments.get("query", "")
        category = arguments.get("category")

        docs = self._kb.query(query, category=category, top_k=3)
        if not docs:
            return "No relevant knowledge found."

        parts: list[str] = []
        for doc in docs:
            score = f" (relevance: {doc.relevanceScore:.2f})" if doc.relevanceScore else ""
            parts.append(f"[{doc.category}]{score}: {doc.content}")

        return "\n\n".join(parts)


# ============================================================================
# Tool Registry
# ============================================================================


class VariationToolRegistry(ToolExecutor):
    """Registry and executor for variation tools.

    Implements the ToolExecutor protocol used by AgentSession.

    Usage:
        registry = VariationToolRegistry(
            gene_pool=gene_pool,
            knowledge_base=knowledge_base,
        )
        tools = registry.get_tool_specs()
        # Pass to AgentSession as tool_executor=registry
    """

    def __init__(
        self,
        gene_pool: GenePool | None = None,
        execution_engine: ExecutionEngine | None = None,
        knowledge_base: KnowledgeBase | None = None,
        constraints: dict[str, Any] | None = None,
        budget: InnerLoopBudget | None = None,
        enabled_tools: list[str] | None = None,
    ) -> None:
        self._tools: dict[str, VariationTool] = {}

        all_tools: list[VariationTool] = [
            InspectTraceTool(),
            CompareSOPsTool(gene_pool),
            QueryGenePoolTool(gene_pool),
            DryRunTool(execution_engine, budget),
            ValidateMutationTool(constraints),
            QueryKnowledgeBaseTool(knowledge_base),
        ]

        enabled = set(enabled_tools) if enabled_tools else None
        for tool in all_tools:
            if enabled is None or tool.name in enabled:
                self._tools[tool.name] = tool

    def get_tool_specs(self) -> list[VariationToolSpec]:
        """Get tool specifications for AgentSession."""
        return [t.to_spec() for t in self._tools.values()]

    def get_tool(self, name: str) -> VariationTool | None:
        """Get a specific tool by name."""
        return self._tools.get(name)

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool by name (implements ToolExecutor protocol)."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return f"Unknown tool: {tool_name}. Available: {list(self._tools.keys())}"
        result = tool.execute(arguments)
        if len(result) > MAX_TOOL_OUTPUT_CHARS:
            return result[:MAX_TOOL_OUTPUT_CHARS] + "\n... (truncated)"
        return result

    def set_traces(self, traces: dict[str, Any]) -> None:
        """Update trace data for InspectTraceTool."""
        inspect_tool = self._tools.get("inspect_trace")
        if isinstance(inspect_tool, InspectTraceTool):
            inspect_tool.set_traces(traces)
