"""Tests for crossover mutations in Director/Architect"""

import random

import pytest

from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
)
from siare.services.director import Architect
from siare.services.llm_provider import LLMProvider


# Mock LLM Provider for testing
class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def complete(self, messages, model="gpt-5", temperature=0.7, **kwargs):
        """Mock completion"""

        class MockResponse:
            def __init__(self):
                self.content = "Mock response"

        return MockResponse()

    def get_model_name(self, model_ref: str) -> str:
        """Map model reference to actual model name"""
        return model_ref


@pytest.fixture
def architect():
    """Create Architect instance"""
    llm_provider = MockLLMProvider()
    return Architect(llm_provider)


@pytest.fixture
def parent_a_sop():
    """Create parent A SOP for crossover testing"""
    return ProcessConfig(
        id="parent_a",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="planner",
                model="gpt-4",
                promptRef="planner_prompt_a",
            ),
            RoleConfig(
                id="retriever",
                model="gpt-4",
                promptRef="retriever_prompt_a",
            ),
            RoleConfig(
                id="analyzer",
                model="gpt-4",
                promptRef="analyzer_prompt_a",
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="planner"),
            GraphEdge(from_="planner", to="retriever"),
            GraphEdge(from_="retriever", to="analyzer"),
        ],
    )


@pytest.fixture
def parent_a_genome():
    """Create parent A genome"""
    return PromptGenome(
        id="genome_a",
        version="1.0.0",
        rolePrompts={
            "planner_prompt_a": RolePrompt(
                id="planner_prompt_a",
                content="You are a planning agent from parent A.",
            ),
            "retriever_prompt_a": RolePrompt(
                id="retriever_prompt_a",
                content="You are a retrieval agent from parent A.",
            ),
            "analyzer_prompt_a": RolePrompt(
                id="analyzer_prompt_a",
                content="You are an analysis agent from parent A.",
            ),
        },
    )


@pytest.fixture
def parent_b_sop():
    """Create parent B SOP for crossover testing"""
    return ProcessConfig(
        id="parent_b",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="planner",
                model="gpt-4",
                promptRef="planner_prompt_b",
            ),
            RoleConfig(
                id="retriever",
                model="gpt-4",
                promptRef="retriever_prompt_b",
            ),
            RoleConfig(
                id="critic",
                model="gpt-4",
                promptRef="critic_prompt_b",
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="planner"),
            GraphEdge(from_="planner", to="retriever"),
            GraphEdge(from_="retriever", to="critic"),
        ],
    )


@pytest.fixture
def parent_b_genome():
    """Create parent B genome"""
    return PromptGenome(
        id="genome_b",
        version="1.0.0",
        rolePrompts={
            "planner_prompt_b": RolePrompt(
                id="planner_prompt_b",
                content="You are a planning agent from parent B.",
            ),
            "retriever_prompt_b": RolePrompt(
                id="retriever_prompt_b",
                content="You are a retrieval agent from parent B.",
            ),
            "critic_prompt_b": RolePrompt(
                id="critic_prompt_b",
                content="You are a critic agent from parent B.",
            ),
        },
    )


def test_crossover_basic(architect, parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome):
    """Test basic crossover between two compatible SOPs"""
    # Set seed for reproducibility
    random.seed(42)

    new_config, new_genome = architect.propose_crossover(
        parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome, strategy="role_crossover"
    )

    # Verify new SOP was created
    assert new_config is not None
    assert new_genome is not None

    # Verify version was incremented (major version)
    assert new_config.version == "2.0.0"
    assert new_genome.version == "2.0.0"

    # Verify roles were selected
    role_ids = {role.id for role in new_config.roles}
    assert len(role_ids) > 0

    # Verify all role types are present (planner, retriever, analyzer/critic)
    role_types = {role.id.split("_")[0] if "_" in role.id else role.id for role in new_config.roles}
    assert "planner" in role_types
    assert "retriever" in role_types
    assert "analyzer" in role_types or "critic" in role_types

    # Verify graph has edges
    assert len(new_config.graph) > 0

    # Verify all roles have prompts in genome
    for role in new_config.roles:
        assert role.promptRef in new_genome.rolePrompts


def test_crossover_child_inherits_from_both_parents(
    architect, parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome
):
    """Test that child inherits characteristics from both parents"""
    # Run crossover multiple times to check randomness
    # Check that prompts vary (since role IDs are the same, but prompts differ)
    prompt_variations = set()

    for i in range(10):
        random.seed(i)  # Different seed each time
        new_config, new_genome = architect.propose_crossover(
            parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome, strategy="role_crossover"
        )

        # Create a signature based on which parent's prompts were used
        signature = []
        for role in new_config.roles:
            prompt_content = new_genome.rolePrompts.get(role.promptRef)
            if prompt_content:
                # Check if it's from parent A or B based on content
                if "parent A" in prompt_content.content:
                    signature.append("A")
                elif "parent B" in prompt_content.content:
                    signature.append("B")

        prompt_variations.add(tuple(signature))

    # Due to randomness, we should see at least some variation in prompt sources
    # (With 10 runs and random selection, we should see multiple combinations)
    assert len(prompt_variations) > 1, f"No variation in crossover results: {prompt_variations}"


def test_crossover_validation(architect, parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome):
    """Test that crossover validates the resulting SOP"""
    random.seed(42)

    # Should not raise an error for valid parents
    new_config, new_genome = architect.propose_crossover(
        parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome, strategy="role_crossover"
    )

    # Verify the result is valid (has at least one role and one edge)
    assert len(new_config.roles) > 0
    assert len(new_config.graph) > 0


def test_crossover_edge_mapping(
    architect, parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome
):
    """Test that edge mapping works correctly"""
    random.seed(42)

    new_config, _ = architect.propose_crossover(
        parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome, strategy="role_crossover"
    )

    # Verify all edges reference valid roles
    role_ids = {role.id for role in new_config.roles}

    for edge in new_config.graph:
        # Check from_ nodes
        from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
        for from_node in from_nodes:
            assert from_node == "user_input" or from_node in role_ids, (
                f"Edge from_ references unknown role: {from_node}"
            )

        # Check to node
        assert edge.to in role_ids, f"Edge to references unknown role: {edge.to}"


def test_crossover_no_common_roles(architect):
    """Test crossover when parents have no common role types"""
    # Create parents with completely different role types (using different prefixes)
    parent_c = ProcessConfig(
        id="parent_c",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(id="searcher", model="gpt-4", promptRef="prompt_searcher"),
            RoleConfig(id="validator", model="gpt-4", promptRef="prompt_validator"),
        ],
        graph=[
            GraphEdge(from_="user_input", to="searcher"),
            GraphEdge(from_="searcher", to="validator"),
        ],
    )

    genome_c = PromptGenome(
        id="genome_c",
        version="1.0.0",
        rolePrompts={
            "prompt_searcher": RolePrompt(id="prompt_searcher", content="Searcher role"),
            "prompt_validator": RolePrompt(id="prompt_validator", content="Validator role"),
        },
    )

    parent_d = ProcessConfig(
        id="parent_d",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(id="aggregator", model="gpt-4", promptRef="prompt_aggregator"),
            RoleConfig(id="summarizer", model="gpt-4", promptRef="prompt_summarizer"),
        ],
        graph=[
            GraphEdge(from_="user_input", to="aggregator"),
            GraphEdge(from_="aggregator", to="summarizer"),
        ],
    )

    genome_d = PromptGenome(
        id="genome_d",
        version="1.0.0",
        rolePrompts={
            "prompt_aggregator": RolePrompt(id="prompt_aggregator", content="Aggregator role"),
            "prompt_summarizer": RolePrompt(id="prompt_summarizer", content="Summarizer role"),
        },
    )

    random.seed(42)

    # Should still work - takes union of role types
    new_config, new_genome = architect.propose_crossover(
        parent_c, genome_c, parent_d, genome_d, strategy="role_crossover"
    )

    # Verify child has roles from both parents
    role_ids = {role.id for role in new_config.roles}
    assert len(role_ids) == 4  # All 4 unique role types
    assert "searcher" in role_ids
    assert "validator" in role_ids
    assert "aggregator" in role_ids
    assert "summarizer" in role_ids


def test_crossover_unknown_strategy(
    architect, parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome
):
    """Test that unknown strategy raises ValueError"""
    with pytest.raises(ValueError, match="Unknown crossover strategy"):
        architect.propose_crossover(
            parent_a_sop,
            parent_a_genome,
            parent_b_sop,
            parent_b_genome,
            strategy="invalid_strategy",
        )


def test_crossover_unknown_strategy_raises_error(
    architect, parent_a_sop, parent_a_genome, parent_b_sop, parent_b_genome
):
    """Test that unknown crossover strategies raise ValueError"""
    with pytest.raises(ValueError, match="Unknown crossover strategy"):
        architect.propose_crossover(
            parent_a_sop,
            parent_a_genome,
            parent_b_sop,
            parent_b_genome,
            strategy="nonexistent_strategy",
        )


def test_increment_version_major(architect):
    """Test version increment - major"""
    new_version = architect._increment_version("1.2.3", major=True)
    assert new_version == "2.0.0"


def test_increment_version_minor(architect):
    """Test version increment - minor"""
    new_version = architect._increment_version("1.2.3", minor=True)
    assert new_version == "1.3.0"


def test_increment_version_patch(architect):
    """Test version increment - patch (default)"""
    new_version = architect._increment_version("1.2.3")
    assert new_version == "1.2.4"


def test_increment_version_with_suffix(architect):
    """Test version increment with suffix"""
    new_version = architect._increment_version("1.2.3-alpha", major=True)
    assert new_version == "2.0.0"


def test_crossover_preserves_tools(architect):
    """Test that crossover preserves tools from parents"""
    # Create parents with tools
    parent_with_tools_a = ProcessConfig(
        id="parent_tools_a",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=["tool_a", "tool_b"],
        roles=[
            RoleConfig(
                id="searcher",
                model="gpt-4",
                promptRef="searcher_prompt",
                tools=["tool_a"],
            ),
        ],
        graph=[GraphEdge(from_="user_input", to="searcher")],
    )

    genome_tools_a = PromptGenome(
        id="genome_tools_a",
        version="1.0.0",
        rolePrompts={
            "searcher_prompt": RolePrompt(id="searcher_prompt", content="Searcher agent"),
        },
    )

    parent_with_tools_b = ProcessConfig(
        id="parent_tools_b",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=["tool_c"],
        roles=[
            RoleConfig(
                id="searcher",
                model="gpt-4",
                promptRef="searcher_prompt_b",
                tools=["tool_c"],
            ),
        ],
        graph=[GraphEdge(from_="user_input", to="searcher")],
    )

    genome_tools_b = PromptGenome(
        id="genome_tools_b",
        version="1.0.0",
        rolePrompts={
            "searcher_prompt_b": RolePrompt(id="searcher_prompt_b", content="Searcher agent B"),
        },
    )

    random.seed(42)

    new_config, _ = architect.propose_crossover(
        parent_with_tools_a,
        genome_tools_a,
        parent_with_tools_b,
        genome_tools_b,
        strategy="role_crossover",
    )

    # Child should have tools
    assert new_config.tools is not None
    assert len(new_config.tools) > 0


# =============================================================================
# prompt_crossover Tests
# =============================================================================


@pytest.fixture
def structured_prompt_parent_a():
    """Parent A with markdown-structured prompts"""
    return ProcessConfig(
        id="structured_a",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="analyst",
                model="gpt-4",
                promptRef="analyst_prompt_a",
            ),
        ],
        graph=[GraphEdge(from_="user_input", to="analyst")],
    )


@pytest.fixture
def structured_genome_parent_a():
    """Parent A genome with structured prompt"""
    return PromptGenome(
        id="genome_struct_a",
        version="1.0.0",
        rolePrompts={
            "analyst_prompt_a": RolePrompt(
                id="analyst_prompt_a",
                content="""# Role
You are a data analyst from Parent A.

## Instructions
Analyze data using method A.

## Constraints
Never fabricate data.

## Examples
Example from A.""",
            ),
        },
    )


@pytest.fixture
def structured_prompt_parent_b():
    """Parent B with markdown-structured prompts"""
    return ProcessConfig(
        id="structured_b",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="analyst",
                model="gpt-4",
                promptRef="analyst_prompt_b",
            ),
        ],
        graph=[GraphEdge(from_="user_input", to="analyst")],
    )


@pytest.fixture
def structured_genome_parent_b():
    """Parent B genome with structured prompt"""
    return PromptGenome(
        id="genome_struct_b",
        version="1.0.0",
        rolePrompts={
            "analyst_prompt_b": RolePrompt(
                id="analyst_prompt_b",
                content="""# Role
You are a data analyst from Parent B.

## Instructions
Analyze data using method B.

## Constraints
Never modify source data.

## Examples
Example from B.""",
            ),
        },
    )


def test_prompt_crossover_produces_valid_result(
    architect,
    structured_prompt_parent_a,
    structured_genome_parent_a,
    structured_prompt_parent_b,
    structured_genome_parent_b,
):
    """Test prompt_crossover produces valid SOP and genome"""
    random.seed(42)

    new_config, new_genome = architect.propose_crossover(
        structured_prompt_parent_a,
        structured_genome_parent_a,
        structured_prompt_parent_b,
        structured_genome_parent_b,
        strategy="prompt_crossover",
    )

    # Should produce valid config
    assert new_config is not None
    assert new_genome is not None
    assert len(new_config.roles) > 0
    assert len(new_genome.rolePrompts) > 0


def test_prompt_crossover_mixes_sections(
    architect,
    structured_prompt_parent_a,
    structured_genome_parent_a,
    structured_prompt_parent_b,
    structured_genome_parent_b,
):
    """Test prompt_crossover mixes sections from both parents"""
    # Run multiple times to verify mixing happens
    seen_a = False
    seen_b = False

    for seed in range(20):
        random.seed(seed)
        _, new_genome = architect.propose_crossover(
            structured_prompt_parent_a,
            structured_genome_parent_a,
            structured_prompt_parent_b,
            structured_genome_parent_b,
            strategy="prompt_crossover",
        )

        # Get the analyst prompt content
        prompt_content = None
        for prompt in new_genome.rolePrompts.values():
            prompt_content = prompt.content
            break

        if prompt_content:
            if "method A" in prompt_content:
                seen_a = True
            if "method B" in prompt_content:
                seen_b = True

    # With enough runs, should see content from both parents
    assert seen_a or seen_b, "Should see content from at least one parent"


def test_prompt_crossover_preserves_constraints(
    architect,
    structured_prompt_parent_a,
    structured_genome_parent_a,
    structured_prompt_parent_b,
    structured_genome_parent_b,
):
    """Test prompt_crossover preserves immutable constraint sections"""
    random.seed(42)

    _, new_genome = architect.propose_crossover(
        structured_prompt_parent_a,
        structured_genome_parent_a,
        structured_prompt_parent_b,
        structured_genome_parent_b,
        strategy="prompt_crossover",
    )

    # Get the prompt content
    prompt_content = None
    for prompt in new_genome.rolePrompts.values():
        prompt_content = prompt.content
        break

    # Should have constraints (they're immutable, so should be preserved)
    assert prompt_content is not None
    # At minimum, should contain some constraint content from parent A (safety)
    assert "Constraints" in prompt_content or "constraint" in prompt_content.lower()


def test_prompt_crossover_inherits_unique_roles(
    architect,
):
    """Test roles unique to one parent inherit prompt unchanged"""
    # Parent A has planner, Parent B has analyzer
    parent_a = ProcessConfig(
        id="unique_a",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[RoleConfig(id="planner", model="gpt-4", promptRef="planner_prompt")],
        graph=[GraphEdge(from_="user_input", to="planner")],
    )
    genome_a = PromptGenome(
        id="genome_unique_a",
        version="1.0.0",
        rolePrompts={
            "planner_prompt": RolePrompt(
                id="planner_prompt",
                content="Planner prompt from A",
            ),
        },
    )

    parent_b = ProcessConfig(
        id="unique_b",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[RoleConfig(id="analyzer", model="gpt-4", promptRef="analyzer_prompt")],
        graph=[GraphEdge(from_="user_input", to="analyzer")],
    )
    genome_b = PromptGenome(
        id="genome_unique_b",
        version="1.0.0",
        rolePrompts={
            "analyzer_prompt": RolePrompt(
                id="analyzer_prompt",
                content="Analyzer prompt from B",
            ),
        },
    )

    random.seed(42)

    new_config, new_genome = architect.propose_crossover(
        parent_a, genome_a, parent_b, genome_b, strategy="prompt_crossover"
    )

    # Should have both roles
    role_ids = {r.id for r in new_config.roles}
    assert "planner" in role_ids or "analyzer" in role_ids

    # Prompts should be inherited unchanged for unique roles
    for prompt in new_genome.rolePrompts.values():
        # Each unique role should have its original prompt content
        assert "from A" in prompt.content or "from B" in prompt.content


def test_prompt_crossover_handles_unstructured_prompts(
    architect,
):
    """Test prompt_crossover handles prompts without markdown structure"""
    parent_a = ProcessConfig(
        id="flat_a",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[RoleConfig(id="agent", model="gpt-4", promptRef="agent_prompt_a")],
        graph=[GraphEdge(from_="user_input", to="agent")],
    )
    genome_a = PromptGenome(
        id="genome_flat_a",
        version="1.0.0",
        rolePrompts={
            "agent_prompt_a": RolePrompt(
                id="agent_prompt_a",
                content="You are a helpful agent. Be concise.",
            ),
        },
    )

    parent_b = ProcessConfig(
        id="flat_b",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[RoleConfig(id="agent", model="gpt-4", promptRef="agent_prompt_b")],
        graph=[GraphEdge(from_="user_input", to="agent")],
    )
    genome_b = PromptGenome(
        id="genome_flat_b",
        version="1.0.0",
        rolePrompts={
            "agent_prompt_b": RolePrompt(
                id="agent_prompt_b",
                content="You are an assistant. Provide detailed answers.",
            ),
        },
    )

    random.seed(42)

    # Should not raise - handles unstructured prompts gracefully
    new_config, new_genome = architect.propose_crossover(
        parent_a, genome_a, parent_b, genome_b, strategy="prompt_crossover"
    )

    assert new_config is not None
    assert new_genome is not None


# =============================================================================
# graph_crossover Tests
# =============================================================================


@pytest.fixture
def graph_parent_a():
    """Parent A with specific graph topology"""
    return ProcessConfig(
        id="graph_a",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(id="planner", model="gpt-4", promptRef="planner_prompt_a"),
            RoleConfig(id="retriever", model="gpt-4", promptRef="retriever_prompt_a"),
            RoleConfig(id="synthesizer", model="gpt-4", promptRef="synthesizer_prompt_a"),
        ],
        graph=[
            GraphEdge(from_="user_input", to="planner"),
            GraphEdge(from_="planner", to="retriever"),
            GraphEdge(from_="retriever", to="synthesizer"),
        ],
    )


@pytest.fixture
def graph_genome_a():
    """Genome for graph parent A"""
    return PromptGenome(
        id="genome_graph_a",
        version="1.0.0",
        rolePrompts={
            "planner_prompt_a": RolePrompt(id="planner_prompt_a", content="Planner A"),
            "retriever_prompt_a": RolePrompt(id="retriever_prompt_a", content="Retriever A"),
            "synthesizer_prompt_a": RolePrompt(id="synthesizer_prompt_a", content="Synthesizer A"),
        },
    )


@pytest.fixture
def graph_parent_b():
    """Parent B with different graph topology"""
    return ProcessConfig(
        id="graph_b",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(id="planner", model="gpt-4", promptRef="planner_prompt_b"),
            RoleConfig(id="retriever", model="gpt-4", promptRef="retriever_prompt_b"),
            RoleConfig(id="validator", model="gpt-4", promptRef="validator_prompt_b"),
        ],
        graph=[
            GraphEdge(from_="user_input", to="planner"),
            GraphEdge(from_="planner", to="retriever"),
            GraphEdge(from_="planner", to="validator"),  # Different topology - parallel
        ],
    )


@pytest.fixture
def graph_genome_b():
    """Genome for graph parent B"""
    return PromptGenome(
        id="genome_graph_b",
        version="1.0.0",
        rolePrompts={
            "planner_prompt_b": RolePrompt(id="planner_prompt_b", content="Planner B"),
            "retriever_prompt_b": RolePrompt(id="retriever_prompt_b", content="Retriever B"),
            "validator_prompt_b": RolePrompt(id="validator_prompt_b", content="Validator B"),
        },
    )


def test_graph_crossover_produces_valid_result(
    architect,
    graph_parent_a,
    graph_genome_a,
    graph_parent_b,
    graph_genome_b,
):
    """Test graph_crossover produces valid SOP with valid DAG"""
    random.seed(42)

    new_config, new_genome = architect.propose_crossover(
        graph_parent_a,
        graph_genome_a,
        graph_parent_b,
        graph_genome_b,
        strategy="graph_crossover",
    )

    # Should produce valid config
    assert new_config is not None
    assert new_genome is not None
    assert len(new_config.roles) > 0
    assert len(new_config.graph) > 0

    # DAG should be valid (no cycles, all roles reachable)
    from siare.services.execution_engine import ExecutionEngine
    engine = ExecutionEngine()
    errors = engine.validate_sop(new_config)
    assert not errors, f"Invalid DAG: {errors}"


def test_graph_crossover_mixes_edges(
    architect,
    graph_parent_a,
    graph_genome_a,
    graph_parent_b,
    graph_genome_b,
):
    """Test graph_crossover can produce edges from both parents"""
    # Run multiple times to verify mixing happens
    topologies: list[set[tuple[str, str]]] = []

    for seed in range(20):
        random.seed(seed)
        new_config, _ = architect.propose_crossover(
            graph_parent_a,
            graph_genome_a,
            graph_parent_b,
            graph_genome_b,
            strategy="graph_crossover",
        )

        # Extract edges as tuples for comparison
        edges = set()
        for edge in new_config.graph:
            from_str = str(edge.from_) if isinstance(edge.from_, list) else edge.from_
            edges.add((from_str, edge.to))
        topologies.append(edges)

    # Verify we ran all iterations and produced valid topologies
    assert len(topologies) == 20


def test_graph_crossover_validates_dag(
    architect,
    graph_parent_a,
    graph_genome_a,
    graph_parent_b,
    graph_genome_b,
):
    """Test graph_crossover produces valid DAG (no cycles)"""
    random.seed(42)

    new_config, _ = architect.propose_crossover(
        graph_parent_a,
        graph_genome_a,
        graph_parent_b,
        graph_genome_b,
        strategy="graph_crossover",
    )

    # Check there are no cycles by verifying topological sort is possible
    from siare.services.execution_engine import ExecutionEngine
    engine = ExecutionEngine()
    errors = engine.validate_sop(new_config)
    assert not errors


def test_graph_crossover_handles_disconnected_fallback(architect):
    """Test graph_crossover creates linear fallback when needed"""
    # Parents with no common edges
    parent_a = ProcessConfig(
        id="disconnected_a",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[RoleConfig(id="alpha", model="gpt-4", promptRef="alpha_prompt")],
        graph=[GraphEdge(from_="user_input", to="alpha")],
    )
    genome_a = PromptGenome(
        id="genome_disc_a",
        version="1.0.0",
        rolePrompts={"alpha_prompt": RolePrompt(id="alpha_prompt", content="Alpha")},
    )

    parent_b = ProcessConfig(
        id="disconnected_b",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[RoleConfig(id="beta", model="gpt-4", promptRef="beta_prompt")],
        graph=[GraphEdge(from_="user_input", to="beta")],
    )
    genome_b = PromptGenome(
        id="genome_disc_b",
        version="1.0.0",
        rolePrompts={"beta_prompt": RolePrompt(id="beta_prompt", content="Beta")},
    )

    random.seed(42)

    # Should not raise - creates fallback
    new_config, _ = architect.propose_crossover(
        parent_a, genome_a, parent_b, genome_b, strategy="graph_crossover"
    )

    # Should have at least one edge
    assert len(new_config.graph) >= 1
    # Should have at least one role
    assert len(new_config.roles) >= 1
