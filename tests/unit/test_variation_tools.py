"""Tests for variation tools and tool registry."""

import json
from typing import Any

import pytest

from siare.core.models import InnerLoopBudget, VariationToolSpec
from siare.services.knowledge_base import KnowledgeBase
from siare.services.variation_tools import (
    CompareSOPsTool,
    DryRunTool,
    InspectTraceTool,
    QueryGenePoolTool,
    QueryKnowledgeBaseTool,
    ValidateMutationTool,
    VariationToolRegistry,
)


# ============================================================================
# InspectTraceTool Tests
# ============================================================================


class TestInspectTraceTool:
    """Tests for trace inspection."""

    def setup_method(self):
        self.traces = {
            "trace-1": {
                "run_id": "trace-1",
                "status": "completed",
                "node_executions": [
                    {"role_id": "retriever", "duration_ms": 500},
                    {"role_id": "answerer", "duration_ms": 200},
                ],
                "errors": [],
                "final_outputs": {"answer": "Paris is the capital."},
            },
            "trace-2": {
                "run_id": "trace-2",
                "status": "failed",
                "node_executions": [
                    {"role_id": "retriever", "duration_ms": 100},
                ],
                "errors": [
                    {"role_id": "answerer", "error": "Context too long"},
                ],
                "final_outputs": {},
            },
        }
        self.tool = InspectTraceTool(traces=self.traces)

    def test_spec(self):
        spec = self.tool.to_spec()
        assert spec.name == "inspect_trace"
        assert "trace" in spec.description.lower()

    def test_full_inspection(self):
        result = self.tool.execute({"trace_id": "trace-1", "focus": "full"})
        assert "retriever" in result
        assert "answerer" in result

    def test_errors_focus(self):
        result = self.tool.execute({"trace_id": "trace-2", "focus": "errors"})
        assert "Context too long" in result

    def test_timing_focus(self):
        result = self.tool.execute({"trace_id": "trace-1", "focus": "timing"})
        parsed = json.loads(result)
        assert len(parsed) == 2
        assert parsed[0]["duration_ms"] == 500

    def test_outputs_focus(self):
        result = self.tool.execute({"trace_id": "trace-1", "focus": "outputs"})
        assert "Paris" in result

    def test_no_errors_message(self):
        result = self.tool.execute({"trace_id": "trace-1", "focus": "errors"})
        assert "No errors" in result

    def test_missing_trace(self):
        result = self.tool.execute({"trace_id": "nonexistent"})
        assert "not found" in result


# ============================================================================
# ValidateMutationTool Tests
# ============================================================================


class TestValidateMutationTool:
    """Tests for mutation validation."""

    def test_valid_mutation(self):
        tool = ValidateMutationTool(constraints={"maxRoles": 5})
        result = tool.execute({
            "mutation_type": "prompt_change",
            "num_roles_after": 3,
        })
        assert result.startswith("VALID")

    def test_disallowed_mutation_type(self):
        tool = ValidateMutationTool(constraints={
            "disallowedMutationTypes": ["remove_role"],
        })
        result = tool.execute({"mutation_type": "remove_role"})
        assert result.startswith("INVALID")
        assert "disallowed" in result

    def test_exceeds_max_roles(self):
        tool = ValidateMutationTool(constraints={"maxRoles": 3})
        result = tool.execute({
            "mutation_type": "add_role",
            "num_roles_after": 4,
        })
        assert result.startswith("INVALID")
        assert "max roles" in result.lower()

    def test_remove_mandatory_role(self):
        tool = ValidateMutationTool(constraints={
            "mandatoryRoles": ["retriever", "answerer"],
        })
        result = tool.execute({
            "mutation_type": "remove_role",
            "target_role": "retriever",
        })
        assert result.startswith("INVALID")
        assert "mandatory" in result

    def test_no_constraints(self):
        tool = ValidateMutationTool()
        result = tool.execute({"mutation_type": "add_role", "num_roles_after": 10})
        assert result.startswith("VALID")


# ============================================================================
# QueryKnowledgeBaseTool Tests
# ============================================================================


class TestQueryKnowledgeBaseTool:
    """Tests for knowledge base querying tool."""

    def test_query_returns_results(self):
        kb = KnowledgeBase()
        kb.load_builtin_knowledge()
        tool = QueryKnowledgeBaseTool(knowledge_base=kb)

        result = tool.execute({"query": "hallucination faithfulness"})
        assert "hallucination" in result.lower() or "faithfulness" in result.lower()

    def test_query_with_category(self):
        kb = KnowledgeBase()
        kb.load_builtin_knowledge()
        tool = QueryKnowledgeBaseTool(knowledge_base=kb)

        result = tool.execute({
            "query": "chain of thought",
            "category": "prompt_engineering",
        })
        assert len(result) > 0

    def test_no_knowledge_base(self):
        tool = QueryKnowledgeBaseTool(knowledge_base=None)
        result = tool.execute({"query": "test"})
        assert "not available" in result

    def test_no_matches(self):
        kb = KnowledgeBase()
        tool = QueryKnowledgeBaseTool(knowledge_base=kb)
        result = tool.execute({"query": "quantum entanglement"})
        assert "No relevant" in result


# ============================================================================
# QueryGenePoolTool Tests
# ============================================================================


class TestQueryGenePoolTool:
    """Tests for gene pool querying tool."""

    def test_no_gene_pool(self):
        tool = QueryGenePoolTool(gene_pool=None)
        result = tool.execute({"query_type": "diversity_stats"})
        assert "not available" in result

    def test_unknown_query_type(self):
        class MockPool:
            pass

        tool = QueryGenePoolTool(gene_pool=MockPool())
        result = tool.execute({"query_type": "unknown_type"})
        assert "Unknown" in result


# ============================================================================
# VariationToolRegistry Tests
# ============================================================================


class TestVariationToolRegistry:
    """Tests for the tool registry."""

    def test_all_tools_registered(self):
        registry = VariationToolRegistry()
        specs = registry.get_tool_specs()
        names = {s.name for s in specs}
        assert "inspect_trace" in names
        assert "compare_sops" in names
        assert "query_gene_pool" in names
        assert "dry_run" in names
        assert "validate_mutation" in names
        assert "query_knowledge_base" in names

    def test_filter_enabled_tools(self):
        registry = VariationToolRegistry(
            enabled_tools=["inspect_trace", "validate_mutation"],
        )
        specs = registry.get_tool_specs()
        assert len(specs) == 2
        names = {s.name for s in specs}
        assert "inspect_trace" in names
        assert "validate_mutation" in names

    def test_execute_dispatches_to_tool(self):
        registry = VariationToolRegistry()
        registry.set_traces({
            "t-1": {"status": "ok", "node_executions": [], "errors": [],
                     "final_outputs": {}},
        })
        result = registry.execute("inspect_trace", {
            "trace_id": "t-1",
            "focus": "full",
        })
        assert "ok" in result

    def test_execute_unknown_tool(self):
        registry = VariationToolRegistry()
        result = registry.execute("nonexistent_tool", {})
        assert "Unknown tool" in result

    def test_get_tool_by_name(self):
        registry = VariationToolRegistry()
        tool = registry.get_tool("validate_mutation")
        assert tool is not None
        assert tool.name == "validate_mutation"

    def test_get_nonexistent_tool(self):
        registry = VariationToolRegistry()
        assert registry.get_tool("nonexistent") is None

    def test_implements_tool_executor_protocol(self):
        from siare.services.agent_session import ToolExecutor

        registry = VariationToolRegistry()
        assert isinstance(registry, ToolExecutor)

    def test_set_traces_updates_inspect_tool(self):
        registry = VariationToolRegistry()
        registry.set_traces({"new-trace": {"status": "test", "node_executions": [],
                                           "errors": [], "final_outputs": {}}})
        result = registry.execute("inspect_trace", {"trace_id": "new-trace"})
        assert "test" in result


# ============================================================================
# DryRunTool Tests
# ============================================================================


class TestDryRunTool:
    """Tests for the dry-run tool."""

    def test_no_engine(self):
        tool = DryRunTool(execution_engine=None)
        result = tool.execute({})
        assert "not available" in result

    def test_budget_exhausted(self):
        budget = InnerLoopBudget(maxDryRuns=0)
        tool = DryRunTool(execution_engine=object(), budget=budget)
        result = tool.execute({})
        assert "exhausted" in result


# ============================================================================
# CompareSOPsTool Tests
# ============================================================================


class TestCompareSOPsTool:
    """Tests for SOP comparison tool."""

    def test_no_gene_pool(self):
        tool = CompareSOPsTool(gene_pool=None)
        result = tool.execute({
            "sop_a_id": "a", "sop_b_id": "b",
        })
        assert "not available" in result
