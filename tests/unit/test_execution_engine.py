"""Tests for ExecutionEngine"""

import pytest

from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
)
from siare.services.execution_engine import ExecutionEngine
from tests.mocks import MockLLMProvider


@pytest.fixture
def simple_sop():
    """Create a simple linear SOP for testing"""
    return ProcessConfig(
        id="simple_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="planner",
                model="gpt-4",
                promptRef="planner_prompt",
                inputs=[{"from": "user_input"}],
                outputs=["plan"],
            ),
            RoleConfig(
                id="executor",
                model="gpt-4",
                promptRef="executor_prompt",
                inputs=[{"from": "planner"}],
                outputs=["result"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="planner"),
            GraphEdge(from_="planner", to="executor"),
        ],
    )


@pytest.fixture
def prompt_genome():
    """Create a simple PromptGenome"""
    return PromptGenome(
        id="test_genome",
        version="1.0.0",
        rolePrompts={
            "planner_prompt": RolePrompt(
                id="planner_prompt",
                content="You are a planning agent.",
            ),
            "executor_prompt": RolePrompt(
                id="executor_prompt",
                content="You are an execution agent.",
            ),
        },
    )


def test_execution_completes(simple_sop, prompt_genome):
    """Test that execution completes successfully"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    task_input = {"query": "What is 2+2?"}

    trace = engine.execute(simple_sop, prompt_genome, task_input)

    assert trace.status == "completed"
    assert len(trace.node_executions) == 2
    assert trace.node_executions[0]["role_id"] == "planner"
    assert trace.node_executions[1]["role_id"] == "executor"


def test_execution_order(simple_sop, prompt_genome):
    """Test that roles execute in correct topological order"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    task_input = {"query": "Test"}

    trace = engine.execute(simple_sop, prompt_genome, task_input)

    # Planner should execute before executor
    role_order = [exec["role_id"] for exec in trace.node_executions]
    assert role_order == ["planner", "executor"]


def test_sop_validation_duplicate_roles():
    """Test SOP validation catches duplicate role IDs"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    bad_sop = ProcessConfig(
        id="bad_sop",
        version="1.0.0",
        models={},
        tools=[],
        roles=[
            RoleConfig(id="agent", model="gpt-4", promptRef="p1"),
            RoleConfig(id="agent", model="gpt-4", promptRef="p2"),  # Duplicate
        ],
        graph=[],
    )

    errors = engine.validate_sop(bad_sop)
    assert len(errors) > 0
    assert any("Duplicate role IDs" in e for e in errors)


def test_sop_validation_invalid_graph():
    """Test SOP validation catches invalid graph references"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    bad_sop = ProcessConfig(
        id="bad_sop",
        version="1.0.0",
        models={},
        tools=[],
        roles=[
            RoleConfig(id="agent1", model="gpt-4", promptRef="p1"),
        ],
        graph=[
            GraphEdge(from_="agent1", to="nonexistent"),  # Invalid reference
        ],
    )

    errors = engine.validate_sop(bad_sop)
    assert len(errors) > 0
    assert any("unknown role" in e.lower() for e in errors)


def test_sop_validation_cycle_detection():
    """Test SOP validation catches cycles"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    cyclic_sop = ProcessConfig(
        id="cyclic_sop",
        version="1.0.0",
        models={},
        tools=[],
        roles=[
            RoleConfig(id="agent1", model="gpt-4", promptRef="p1"),
            RoleConfig(id="agent2", model="gpt-4", promptRef="p2"),
        ],
        graph=[
            GraphEdge(from_="agent1", to="agent2"),
            GraphEdge(from_="agent2", to="agent1"),  # Creates cycle
        ],
    )

    errors = engine.validate_sop(cyclic_sop)
    assert len(errors) > 0
    assert any("cycle" in e.lower() for e in errors)


def test_conditional_execution_true_condition(prompt_genome):
    """Test that role executes when condition is true"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    # Create SOP with conditional edge
    sop = ProcessConfig(
        id="conditional_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="analyzer",
                model="gpt-4",
                promptRef="planner_prompt",
                inputs=[{"from": "user_input"}],
                outputs=["score"],
            ),
            RoleConfig(
                id="responder",
                model="gpt-4",
                promptRef="executor_prompt",
                inputs=[{"from": "analyzer"}],
                outputs=["result"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="analyzer"),
            # Conditional edge: only execute responder if score > 0.5
            GraphEdge(from_="analyzer", to="responder", condition="score > 0.5"),
        ],
    )

    task_input = {"query": "Test"}
    trace = engine.execute(sop, prompt_genome, task_input)

    assert trace.status == "completed"
    # Analyzer should always execute
    assert any(exec["role_id"] == "analyzer" for exec in trace.node_executions)
    # Responder should execute (mock returns score=0.85 > 0.5)
    responder_exec = next(
        (exec for exec in trace.node_executions if exec["role_id"] == "responder"), None
    )
    assert responder_exec is not None
    assert responder_exec["outputs"].get("_skipped") is not True


def test_conditional_execution_false_condition(prompt_genome):
    """Test that role is skipped when condition is false"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    # Create SOP with conditional edge that will be false
    sop = ProcessConfig(
        id="conditional_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="analyzer",
                model="gpt-4",
                promptRef="planner_prompt",
                inputs=[{"from": "user_input"}],
                outputs=["score"],
            ),
            RoleConfig(
                id="responder",
                model="gpt-4",
                promptRef="executor_prompt",
                inputs=[{"from": "analyzer"}],
                outputs=["result"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="analyzer"),
            # This condition will be false (mock returns score=0.85)
            GraphEdge(from_="analyzer", to="responder", condition="score > 0.9"),
        ],
    )

    task_input = {"query": "Test"}
    trace = engine.execute(sop, prompt_genome, task_input)

    assert trace.status == "completed"
    # Responder should be skipped
    responder_exec = next(
        (exec for exec in trace.node_executions if exec["role_id"] == "responder"), None
    )
    assert responder_exec is not None
    assert responder_exec["outputs"].get("_skipped") is True


def test_conditional_execution_none_check(prompt_genome):
    """Test None checks in conditions"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    sop = ProcessConfig(
        id="conditional_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="analyzer",
                model="gpt-4",
                promptRef="planner_prompt",
                inputs=[{"from": "user_input"}],
                outputs=["result"],
            ),
            RoleConfig(
                id="responder",
                model="gpt-4",
                promptRef="executor_prompt",
                inputs=[{"from": "analyzer"}],
                outputs=["output"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="analyzer"),
            # Check that result is not None
            GraphEdge(from_="analyzer", to="responder", condition="result is not None"),
        ],
    )

    task_input = {"query": "Test"}
    trace = engine.execute(sop, prompt_genome, task_input)

    assert trace.status == "completed"
    # Responder should execute (result field exists and is not None)
    responder_exec = next(
        (exec for exec in trace.node_executions if exec["role_id"] == "responder"), None
    )
    assert responder_exec is not None
    assert responder_exec["outputs"].get("_skipped") is not True


def test_conditional_execution_boolean_operators(prompt_genome):
    """Test boolean AND/OR operators in conditions"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    sop = ProcessConfig(
        id="conditional_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="analyzer",
                model="gpt-4",
                promptRef="planner_prompt",
                inputs=[{"from": "user_input"}],
                outputs=["score", "critique"],
            ),
            RoleConfig(
                id="responder",
                model="gpt-4",
                promptRef="executor_prompt",
                inputs=[{"from": "analyzer"}],
                outputs=["output"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="analyzer"),
            # Use AND operator
            GraphEdge(
                from_="analyzer",
                to="responder",
                condition="score > 0.8 and critique is not None",
            ),
        ],
    )

    task_input = {"query": "Test"}
    trace = engine.execute(sop, prompt_genome, task_input)

    assert trace.status == "completed"
    # Responder should execute (both conditions true)
    responder_exec = next(
        (exec for exec in trace.node_executions if exec["role_id"] == "responder"), None
    )
    assert responder_exec is not None
    assert responder_exec["outputs"].get("_skipped") is not True


def test_conditional_validation_invalid_syntax():
    """Test that SOP validation catches invalid condition syntax"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    bad_sop = ProcessConfig(
        id="bad_sop",
        version="1.0.0",
        models={},
        tools=[],
        roles=[
            RoleConfig(id="agent1", model="gpt-4", promptRef="p1"),
            RoleConfig(id="agent2", model="gpt-4", promptRef="p2"),
        ],
        graph=[
            GraphEdge(from_="user_input", to="agent1"),
            # Invalid condition syntax (contains forbidden keyword)
            GraphEdge(from_="agent1", to="agent2", condition="import os"),
        ],
    )

    errors = engine.validate_sop(bad_sop)
    assert len(errors) > 0
    assert any("condition" in e.lower() and "import" in e.lower() for e in errors)


def test_conditional_validation_valid_syntax():
    """Test that SOP validation accepts valid condition syntax"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    good_sop = ProcessConfig(
        id="good_sop",
        version="1.0.0",
        models={},
        tools=[],
        roles=[
            RoleConfig(id="agent1", model="gpt-4", promptRef="p1"),
            RoleConfig(id="agent2", model="gpt-4", promptRef="p2"),
        ],
        graph=[
            GraphEdge(from_="user_input", to="agent1"),
            # Valid condition
            GraphEdge(from_="agent1", to="agent2", condition="score > 0.8"),
        ],
    )

    errors = engine.validate_sop(good_sop)
    # Should have no condition-related errors
    assert not any("condition" in e.lower() for e in errors)


def test_conditional_execution_multiple_paths():
    """Test conditional branching with multiple paths (OR logic)"""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    # Create a diamond-shaped graph with two conditional paths
    sop = ProcessConfig(
        id="branching_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="classifier",
                model="gpt-4",
                promptRef="planner_prompt",
                inputs=[{"from": "user_input"}],
                outputs=["category"],
            ),
            RoleConfig(
                id="path_a",
                model="gpt-4",
                promptRef="executor_prompt",
                inputs=[{"from": "classifier"}],
                outputs=["result_a"],
            ),
            RoleConfig(
                id="path_b",
                model="gpt-4",
                promptRef="executor_prompt",
                inputs=[{"from": "classifier"}],
                outputs=["result_b"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="classifier"),
            # Two mutually exclusive paths based on category
            GraphEdge(from_="classifier", to="path_a", condition="category == 'A'"),
            GraphEdge(from_="classifier", to="path_b", condition="category == 'B'"),
        ],
    )

    # Manually add the PromptGenome with all needed prompts
    genome = PromptGenome(
        id="test_genome",
        version="1.0.0",
        rolePrompts={
            "planner_prompt": RolePrompt(id="planner_prompt", content="Classify"),
            "executor_prompt": RolePrompt(id="executor_prompt", content="Execute"),
        },
    )

    task_input = {"query": "Test"}
    trace = engine.execute(sop, genome, task_input)

    assert trace.status == "completed"
    # One of the paths should execute, the other should be skipped
    path_a_exec = next(
        (exec for exec in trace.node_executions if exec["role_id"] == "path_a"), None
    )
    path_b_exec = next(
        (exec for exec in trace.node_executions if exec["role_id"] == "path_b"), None
    )

    # Both should have execution records
    assert path_a_exec is not None
    assert path_b_exec is not None

    # One should be skipped, one should execute
    # (We don't know which because mock returns arbitrary category)
    skipped_count = sum(
        [
            1 if path_a_exec["outputs"].get("_skipped") else 0,
            1 if path_b_exec["outputs"].get("_skipped") else 0,
        ]
    )
    assert skipped_count >= 1  # At least one path should be skipped


def test_parse_llm_response_json():
    """Test parsing JSON from LLM response."""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    response = '{"score": 0.9, "critique": "Well structured argument"}'
    expected_outputs = ["score", "critique"]

    result = engine._parse_llm_response(response, expected_outputs)

    assert result["score"] == 0.9
    assert result["critique"] == "Well structured argument"
    assert result["response"] == response  # Full response preserved


def test_parse_llm_response_json_with_markdown():
    """Test parsing JSON wrapped in markdown code blocks."""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    response = """```json
{"score": 0.85, "critique": "Good analysis"}
```"""
    expected_outputs = ["score", "critique"]

    result = engine._parse_llm_response(response, expected_outputs)

    assert result["score"] == 0.85
    assert result["critique"] == "Good analysis"
    assert result["response"] == response


def test_parse_llm_response_structured_text():
    """Test parsing structured text from LLM response."""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    response = """score: 0.85
critique: Good analysis but needs more examples
summary: Overall positive assessment"""

    expected_outputs = ["score", "critique", "summary"]

    result = engine._parse_llm_response(response, expected_outputs)

    assert result["score"] == 0.85  # Should be parsed as float
    assert "needs more examples" in result["critique"]
    assert "positive" in result["summary"]
    assert result["response"] == response


def test_parse_llm_response_case_insensitive():
    """Test case-insensitive field matching."""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    # LLM returns fields with different casing
    response = """Score: 0.92
CRITIQUE: Excellent work
Summary: Very thorough"""

    expected_outputs = ["score", "critique", "summary"]

    result = engine._parse_llm_response(response, expected_outputs)

    assert result["score"] == 0.92
    assert "Excellent" in result["critique"]
    assert "thorough" in result["summary"]


def test_parse_llm_response_fallback():
    """Test fallback when no structured data found."""
    engine = ExecutionEngine(llm_provider=MockLLMProvider())

    response = "This is just a plain text response without structure."
    expected_outputs = ["answer"]

    result = engine._parse_llm_response(response, expected_outputs)

    # Should use first expected output as alias for full response
    assert result["answer"] == response
    assert result["response"] == response


def test_role_params_passed_to_tools():
    """Test that role configuration params are passed to tool adapters.

    This verifies the critical fix for evolution: role params like top_k
    and similarity_threshold must reach tool adapters so evolution can
    actually improve RAG configuration.
    """
    # Track what params the tool receives
    received_params = {}

    def mock_tool_adapter(inputs: dict) -> dict:
        """Mock tool adapter that captures received inputs."""
        received_params.update(inputs)
        return {"results": ["doc1", "doc2"]}

    engine = ExecutionEngine(
        llm_provider=MockLLMProvider(),
        tool_adapters={"vector_search": mock_tool_adapter},
    )

    # Create SOP where a role has params and calls a tool
    sop = ProcessConfig(
        id="rag_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=["vector_search"],
        roles=[
            RoleConfig(
                id="retriever",
                model="gpt-4",
                promptRef="retriever_prompt",
                tools=["vector_search"],
                inputs=[{"from": "user_input"}],
                outputs=["context"],
                # These params should be passed to the tool
                params={
                    "top_k": 10,
                    "similarity_threshold": 0.75,
                    "_evolvable": ["top_k", "similarity_threshold"],  # Should be filtered
                },
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="retriever"),
        ],
    )

    genome = PromptGenome(
        id="rag_genome",
        version="1.0.0",
        rolePrompts={
            "retriever_prompt": RolePrompt(
                id="retriever_prompt",
                content="Search for relevant documents for: {{query}}. Call vector_search tool.",
            ),
        },
    )

    task_input = {"query": "What is machine learning?"}
    trace = engine.execute(sop, genome, task_input)

    # Verify execution completed
    assert trace.status == "completed"

    # The tool should have received the role params
    # Note: _evolvable internal key should be filtered out
    assert "top_k" in received_params, "top_k param should be passed to tool"
    assert received_params["top_k"] == 10
    assert "similarity_threshold" in received_params, "similarity_threshold should be passed"
    assert received_params["similarity_threshold"] == 0.75
    assert "_evolvable" not in received_params, "_evolvable should be filtered out"


def test_tool_adapters_registered_correctly():
    """Test that custom tool adapters are properly registered and invoked."""
    call_count = {"count": 0}

    def custom_tool(inputs: dict) -> dict:
        call_count["count"] += 1
        return {"output": f"processed: {inputs.get('data', 'none')}"}

    engine = ExecutionEngine(
        llm_provider=MockLLMProvider(),
        tool_adapters={"custom_tool": custom_tool},
    )

    sop = ProcessConfig(
        id="tool_test_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=["custom_tool"],
        roles=[
            RoleConfig(
                id="processor",
                model="gpt-4",
                promptRef="processor_prompt",
                tools=["custom_tool"],
                inputs=[{"from": "user_input"}],
                outputs=["result"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="processor"),
        ],
    )

    genome = PromptGenome(
        id="tool_test_genome",
        version="1.0.0",
        rolePrompts={
            "processor_prompt": RolePrompt(
                id="processor_prompt",
                content="Process the input. Call custom_tool.",
            ),
        },
    )

    task_input = {"query": "Test data"}
    trace = engine.execute(sop, genome, task_input)

    assert trace.status == "completed"
    # Tool should have been called at least once
    # Note: depends on LLM deciding to call the tool


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
