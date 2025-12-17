"""Tests for topology mutations in Director/Architect"""

import pytest

from siare.core.models import (
    GraphEdge,
    MutationType,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
)
from siare.services.director import Architect
from siare.services.llm_provider import LLMProvider


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing"""

    def __init__(self, mock_response: str = "Mock response"):
        self.mock_response = mock_response

    def complete(self, messages, model, temperature=0.7, max_tokens=None, **kwargs):
        class MockResponse:
            def __init__(self, content):
                self.content = content

        return MockResponse(self.mock_response)

    def get_model_name(self, model_ref: str) -> str:
        """Map model reference to actual model name"""
        return model_ref


@pytest.fixture
def base_sop():
    """Create a base SOP for testing"""
    return ProcessConfig(
        id="test_sop",
        version="1.0.0",
        description="Test SOP",
        models={"default": "gpt-5"},
        tools=["vector_search"],
        roles=[
            RoleConfig(id="planner", model="gpt-5", promptRef="prompt_planner", tools=None),
            RoleConfig(
                id="retriever", model="gpt-5", promptRef="prompt_retriever", tools=["vector_search"]
            ),
            RoleConfig(id="responder", model="gpt-5", promptRef="prompt_responder", tools=None),
        ],
        graph=[
            GraphEdge(**{"from": "user_input", "to": "planner"}),
            GraphEdge(**{"from": "planner", "to": "retriever"}),
            GraphEdge(**{"from": ["planner", "retriever"], "to": "responder"}),
        ],
    )


@pytest.fixture
def base_genome():
    """Create a base PromptGenome for testing"""
    return PromptGenome(
        id="test_genome",
        version="1.0.0",
        rolePrompts={
            "prompt_planner": RolePrompt(id="prompt_planner", content="You are a planner."),
            "prompt_retriever": RolePrompt(id="prompt_retriever", content="You are a retriever."),
            "prompt_responder": RolePrompt(id="prompt_responder", content="You are a responder."),
        },
    )


@pytest.fixture
def architect():
    """Create an Architect instance"""
    return Architect(llm_provider=MockLLMProvider())


class TestAddRoleMutation:
    """Test ADD_ROLE mutation"""

    def test_add_role_basic(self, architect, base_sop, base_genome):
        """Test adding a new role with basic configuration"""
        llm_content = """
ROLE_ID: critic
MODEL: gpt-5
TOOLS: none
PROMPT: You are a critic. Review the response and provide feedback.
EDGES_FROM: responder
EDGES_TO:
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.ADD_ROLE, None, llm_content, base_sop, base_genome, None
        )

        # Check new version
        assert new_sop.version == "2.0.0"  # Major version bump

        # Check role was added
        assert len(new_sop.roles) == 4
        critic_role = next(r for r in new_sop.roles if r.id == "critic")
        assert critic_role is not None
        assert critic_role.model == "gpt-5"
        assert critic_role.promptRef == "prompt_critic"

        # Check prompt was added
        assert "prompt_critic" in new_genome.rolePrompts
        assert "critic" in new_genome.rolePrompts["prompt_critic"].content.lower()

        # Check edge was added
        critic_edges = [e for e in new_sop.graph if e.to == "critic"]
        assert len(critic_edges) == 1
        assert critic_edges[0].from_ == "responder"

    def test_add_role_with_tools(self, architect, base_sop, base_genome):
        """Test adding a role with tools"""
        llm_content = """
ROLE_ID: fact_checker
MODEL: gpt-5
TOOLS: vector_search, web_search
PROMPT: You are a fact checker. Verify claims using available tools.
EDGES_FROM: user_input
EDGES_TO: responder
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.ADD_ROLE, None, llm_content, base_sop, base_genome, None
        )

        # Check role has tools
        fact_checker = next(r for r in new_sop.roles if r.id == "fact_checker")
        assert fact_checker.tools is not None
        assert "vector_search" in fact_checker.tools
        assert "web_search" in fact_checker.tools

        # Check both incoming and outgoing edges
        incoming = [e for e in new_sop.graph if e.to == "fact_checker"]
        outgoing = [e for e in new_sop.graph if e.from_ == "fact_checker"]
        assert len(incoming) == 1
        assert len(outgoing) == 1

    def test_add_role_generates_id_if_missing(self, architect, base_sop, base_genome):
        """Test that role ID is generated if not provided"""
        llm_content = """
MODEL: gpt-5
PROMPT: You are a helper agent.
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.ADD_ROLE, None, llm_content, base_sop, base_genome, None
        )

        # Check that a role was added with generated ID
        assert len(new_sop.roles) == 4
        new_role = new_sop.roles[-1]
        assert new_role.id.startswith("role_")

    def test_add_role_sanitizes_id(self, architect, base_sop, base_genome):
        """Test that role IDs are sanitized"""
        llm_content = """
ROLE_ID: my-special@role!
PROMPT: Test role
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.ADD_ROLE, None, llm_content, base_sop, base_genome, None
        )

        # Check ID was sanitized
        new_role = new_sop.roles[-1]
        assert "@" not in new_role.id
        assert "!" not in new_role.id

    def test_add_role_validates_dag(self, architect, base_sop, base_genome):
        """Test that adding a role that creates a cycle is rejected"""
        # Create a role that would create a cycle
        llm_content = """
ROLE_ID: cyclic_role
PROMPT: Creates a cycle
EDGES_FROM: responder
EDGES_TO: planner
"""

        # This should raise a ValueError due to cycle detection
        with pytest.raises(ValueError, match="Invalid SOP"):
            architect._apply_mutation(MutationType.ADD_ROLE, None, llm_content, base_sop, base_genome, None)


class TestRemoveRoleMutation:
    """Test REMOVE_ROLE mutation"""

    def test_remove_role_basic(self, architect, base_sop, base_genome):
        """Test removing a role"""
        llm_content = """
REMOVE_ROLE: retriever
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.REMOVE_ROLE, None, llm_content, base_sop, base_genome, None
        )

        # Check version
        assert new_sop.version == "2.0.0"  # Major version bump

        # Check role was removed
        assert len(new_sop.roles) == 2
        role_ids = [r.id for r in new_sop.roles]
        assert "retriever" not in role_ids
        assert "planner" in role_ids
        assert "responder" in role_ids

        # Check edges were removed
        for edge in new_sop.graph:
            from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
            assert "retriever" not in from_nodes
            assert edge.to != "retriever"

    def test_remove_role_with_target(self, architect, base_sop, base_genome):
        """Test removing a role using target_role parameter"""
        new_sop, new_genome = architect._apply_mutation(
            MutationType.REMOVE_ROLE, "planner", "", base_sop, base_genome
        )

        # Check role was removed
        role_ids = [r.id for r in new_sop.roles]
        assert "planner" not in role_ids

    def test_remove_role_updates_multi_source_edges(self, architect, base_sop, base_genome):
        """Test that edges with multiple sources are updated correctly"""
        # Remove planner - responder has edge from [planner, retriever]
        llm_content = "REMOVE_ROLE: planner"

        new_sop, new_genome = architect._apply_mutation(
            MutationType.REMOVE_ROLE, None, llm_content, base_sop, base_genome, None
        )

        # The edge from [planner, retriever] -> responder should be removed
        # because it involves planner
        responder_edges = [e for e in new_sop.graph if e.to == "responder"]
        # Should have removed the edge that includes planner
        for edge in responder_edges:
            from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
            assert "planner" not in from_nodes

    def test_remove_role_parse_variations(self, architect, base_sop, base_genome):
        """Test different formats for specifying role to remove"""
        test_cases = [
            "ROLE_TO_REMOVE: retriever",
            "ROLE_ID: retriever",
            "Remove the role retriever from the SOP",
        ]

        for llm_content in test_cases:
            new_sop, new_genome = architect._apply_mutation(
                MutationType.REMOVE_ROLE, None, llm_content, base_sop, base_genome
            )
            role_ids = [r.id for r in new_sop.roles]
            assert "retriever" not in role_ids


class TestRewireGraphMutation:
    """Test REWIRE_GRAPH mutation"""

    def test_rewire_add_edges(self, architect, base_sop, base_genome):
        """Test adding new edges"""
        llm_content = """
ADD_EDGES:
- from: user_input, to: retriever
REMOVE_EDGES:
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.REWIRE_GRAPH, None, llm_content, base_sop, base_genome
        )

        # Check version
        assert new_sop.version == "2.0.0"  # Major version bump

        # Check new edge exists
        new_edge = next(
            (e for e in new_sop.graph if e.from_ == "user_input" and e.to == "retriever"), None
        )
        assert new_edge is not None

        # Original edges should still exist
        assert len(new_sop.graph) == 4  # 3 original + 1 new

    def test_rewire_remove_edges(self, architect, base_sop, base_genome):
        """Test removing edges"""
        llm_content = """
ADD_EDGES:
REMOVE_EDGES:
- from: planner, to: retriever
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.REWIRE_GRAPH, None, llm_content, base_sop, base_genome
        )

        # Check edge was removed
        removed_edge = next(
            (e for e in new_sop.graph if e.from_ == "planner" and e.to == "retriever"), None
        )
        assert removed_edge is None

        # Should have 2 edges left
        assert len(new_sop.graph) == 2

    def test_rewire_add_and_remove(self, architect, base_sop, base_genome):
        """Test both adding and removing edges"""
        llm_content = """
ADD_EDGES:
- from: user_input, to: responder
REMOVE_EDGES:
- from: planner, to: retriever
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.REWIRE_GRAPH, None, llm_content, base_sop, base_genome
        )

        # Check old edge removed
        old_edge = next(
            (e for e in new_sop.graph if e.from_ == "planner" and e.to == "retriever"), None
        )
        assert old_edge is None

        # Check new edge added
        new_edge = next(
            (e for e in new_sop.graph if e.from_ == "user_input" and e.to == "responder"), None
        )
        assert new_edge is not None

    def test_rewire_json_format(self, architect, base_sop, base_genome):
        """Test JSON format for edge specifications"""
        llm_content = """
ADD_EDGES: [{"from": "user_input", "to": "retriever"}]
REMOVE_EDGES: [{"from": "planner", "to": "retriever"}]
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.REWIRE_GRAPH, None, llm_content, base_sop, base_genome
        )

        # Check changes applied
        assert any(e.from_ == "user_input" and e.to == "retriever" for e in new_sop.graph)
        assert not any(e.from_ == "planner" and e.to == "retriever" for e in new_sop.graph)

    def test_rewire_rejects_cycles(self, architect, base_sop, base_genome):
        """Test that rewiring that creates cycles is rejected"""
        llm_content = """
ADD_EDGES:
- from: responder, to: planner
REMOVE_EDGES:
"""

        # This should create a cycle and be rejected
        with pytest.raises(ValueError, match="Invalid SOP"):
            architect._apply_mutation(
                MutationType.REWIRE_GRAPH, None, llm_content, base_sop, base_genome
            )

    def test_rewire_multi_source_edges(self, architect, base_sop, base_genome):
        """Test handling edges with multiple source nodes"""
        llm_content = """
ADD_EDGES:
- from: user_input,planner, to: retriever
REMOVE_EDGES:
"""

        new_sop, new_genome = architect._apply_mutation(
            MutationType.REWIRE_GRAPH, None, llm_content, base_sop, base_genome
        )

        # Find the multi-source edge
        multi_edge = next(
            (
                e
                for e in new_sop.graph
                if isinstance(e.from_, list)
                and "user_input" in e.from_
                and "planner" in e.from_
                and e.to == "retriever"
            ),
            None,
        )
        assert multi_edge is not None


class TestHelperMethods:
    """Test helper parsing methods"""

    def test_parse_new_role_complete(self, architect):
        """Test parsing complete role specification"""
        llm_content = """
ROLE_ID: analyzer
MODEL: gpt-5-mini
TOOLS: vector_search, calculator
PROMPT: You are an analyzer.
Analyze the data carefully.
EDGES_FROM: user_input, planner
EDGES_TO: responder
"""

        result = architect._parse_new_role_from_llm(
            llm_content, ProcessConfig(id="test", version="1.0.0", models={}, tools=[], roles=[], graph=[])
        )

        assert result["role"].id == "analyzer"
        assert result["role"].model == "gpt-5-mini"
        assert result["role"].tools == ["vector_search", "calculator"]
        assert "analyzer" in result["prompt"].content.lower()
        assert len(result["edges"]) == 3  # 2 incoming, 1 outgoing

    def test_parse_role_to_remove_formats(self, architect):
        """Test different formats for role removal"""
        test_cases = [
            ("REMOVE_ROLE: test_role", "test_role"),
            ("ROLE_TO_REMOVE: test_role", "test_role"),
            ("ROLE_ID: test_role", "test_role"),
        ]

        for content, expected in test_cases:
            result = architect._parse_role_to_remove(content)
            assert result == expected

    def test_parse_graph_changes_line_format(self, architect):
        """Test parsing line-based edge format"""
        llm_content = """
ADD_EDGES:
- from: role1, to: role2
- from: role3, to: role4
REMOVE_EDGES:
- from: role5, to: role6
"""

        add, remove = architect._parse_graph_changes(llm_content)

        assert len(add) == 2
        assert len(remove) == 1
        assert add[0].from_ == "role1"
        assert add[0].to == "role2"

    def test_edge_involves_role(self, architect):
        """Test edge role checking"""
        edge1 = {"from": "role1", "to": "role2"}
        edge2 = {"from": ["role1", "role3"], "to": "role4"}

        assert architect._edge_involves_role(edge1, "role1")
        assert architect._edge_involves_role(edge1, "role2")
        assert not architect._edge_involves_role(edge1, "role3")

        assert architect._edge_involves_role(edge2, "role1")
        assert architect._edge_involves_role(edge2, "role3")
        assert architect._edge_involves_role(edge2, "role4")

    def test_edges_match(self, architect):
        """Test edge matching logic"""
        edge1 = {"from": "role1", "to": "role2"}
        edge2 = {"from": "role1", "to": "role2"}
        edge3 = {"from": "role1", "to": "role3"}
        edge4 = {"from": ["role1"], "to": "role2"}

        assert architect._edges_match(edge1, edge2)
        assert architect._edges_match(edge1, edge4)  # Should match with list form
        assert not architect._edges_match(edge1, edge3)


class TestVersioning:
    """Test version number changes"""

    def test_add_role_increments_major(self, architect, base_sop, base_genome):
        """Test that ADD_ROLE increments major version"""
        llm_content = "ROLE_ID: new_role\nPROMPT: Test"

        new_sop, _ = architect._apply_mutation(
            MutationType.ADD_ROLE, None, llm_content, base_sop, base_genome
        )

        assert new_sop.version == "2.0.0"

    def test_remove_role_increments_major(self, architect, base_sop, base_genome):
        """Test that REMOVE_ROLE increments major version"""
        new_sop, _ = architect._apply_mutation(
            MutationType.REMOVE_ROLE, "retriever", "", base_sop, base_genome
        )

        assert new_sop.version == "2.0.0"

    def test_rewire_graph_increments_major(self, architect, base_sop, base_genome):
        """Test that REWIRE_GRAPH increments major version"""
        llm_content = "ADD_EDGES:\n- from: user_input, to: retriever"

        new_sop, _ = architect._apply_mutation(
            MutationType.REWIRE_GRAPH, None, llm_content, base_sop, base_genome
        )

        assert new_sop.version == "2.0.0"

    def test_prompt_change_increments_minor(self, architect, base_sop, base_genome):
        """Test that PROMPT_CHANGE increments minor version"""
        new_sop, _ = architect._apply_mutation(
            MutationType.PROMPT_CHANGE, "planner", "New prompt content", base_sop, base_genome
        )

        assert new_sop.version == "1.1.0"


class TestConstraintValidation:
    """Test constraint validation"""

    def test_add_role_validates_no_cycles(self, architect, base_sop, base_genome):
        """Test that cycles are detected and rejected"""
        # Try to create a cycle
        llm_content = """
ROLE_ID: cyclic
EDGES_FROM: responder
EDGES_TO: planner
PROMPT: Test
"""

        with pytest.raises(ValueError, match="cycle"):
            architect._apply_mutation(MutationType.ADD_ROLE, None, llm_content, base_sop, base_genome, None)

    def test_remove_role_maintains_valid_graph(self, architect, base_genome):
        """Test that removing a role maintains graph validity"""
        # Create SOP where removing a role would leave orphans
        sop = ProcessConfig(
            id="test",
            version="1.0.0",
            models={"default": "gpt-5"},
            tools=[],
            roles=[
                RoleConfig(id="input_handler", model="gpt-5", promptRef="p1"),
                RoleConfig(id="processor", model="gpt-5", promptRef="p2"),
            ],
            graph=[
                GraphEdge(**{"from": "user_input", "to": "input_handler"}),
                GraphEdge(**{"from": "input_handler", "to": "processor"}),
            ],
        )

        # Remove input_handler - processor becomes orphaned
        # But this should still be valid (processor just has no incoming edges)
        new_sop, _ = architect._apply_mutation(
            MutationType.REMOVE_ROLE, "input_handler", "", sop, base_genome
        )

        assert len(new_sop.roles) == 1
        assert new_sop.roles[0].id == "processor"


class TestIntegration:
    """Integration tests for topology mutations"""

    def test_sequential_mutations(self, architect, base_sop, base_genome):
        """Test applying multiple mutations sequentially"""
        # 1. Add a new role
        add_content = """
ROLE_ID: validator
PROMPT: You validate responses.
EDGES_FROM: responder
"""
        sop1, genome1 = architect._apply_mutation(
            MutationType.ADD_ROLE, None, add_content, base_sop, base_genome
        )

        assert len(sop1.roles) == 4
        assert sop1.version == "2.0.0"

        # 2. Rewire the graph
        rewire_content = """
ADD_EDGES:
- from: planner, to: validator
REMOVE_EDGES:
"""
        sop2, genome2 = architect._apply_mutation(
            MutationType.REWIRE_GRAPH, None, rewire_content, sop1, genome1
        )

        assert sop2.version == "3.0.0"
        # Should have added an edge
        assert any(e.from_ == "planner" and e.to == "validator" for e in sop2.graph)

        # 3. Update a prompt
        sop3, genome3 = architect._apply_mutation(
            MutationType.PROMPT_CHANGE, "validator", "Updated prompt", sop2, genome2
        )

        assert sop3.version == "3.1.0"
        assert genome3.rolePrompts["prompt_validator"].content == "Updated prompt"

    def test_complex_topology_change(self, architect, base_sop, base_genome):
        """Test complex topology with multiple roles and edges"""
        # Add multiple roles
        add1 = """
ROLE_ID: preprocessor
PROMPT: Preprocess inputs
EDGES_FROM: user_input
EDGES_TO: planner
"""
        sop1, genome1 = architect._apply_mutation(
            MutationType.ADD_ROLE, None, add1, base_sop, base_genome
        )

        add2 = """
ROLE_ID: postprocessor
PROMPT: Postprocess outputs
EDGES_FROM: responder
"""
        sop2, genome2 = architect._apply_mutation(MutationType.ADD_ROLE, None, add2, sop1, genome1, None)

        # Verify complex graph
        assert len(sop2.roles) == 5
        role_ids = {r.id for r in sop2.roles}
        assert role_ids == {"planner", "retriever", "responder", "preprocessor", "postprocessor"}

        # Verify no cycles
        from siare.services.execution_engine import ExecutionEngine

        engine = ExecutionEngine()
        errors = engine.validate_sop(sop2)
        assert not errors or not any("cycle" in e.lower() for e in errors)
