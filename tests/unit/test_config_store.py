"""Tests for ConfigStore"""

import pytest

from siare.core.models import (
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
    ToolConfig,
    ToolType,
)
from siare.services.config_store import ConfigStore


@pytest.fixture
def config_store():
    """Create a fresh config store for each test"""
    return ConfigStore()


def test_save_and_get_sop(config_store):
    """Test saving and retrieving an SOP"""
    sop = ProcessConfig(
        id="test_sop",
        version="1.0.0",
        models={"gpt-4": "gpt-5"},
        tools=[],
        roles=[
            RoleConfig(
                id="agent",
                model="gpt-4",
                promptRef="agent_prompt",
            )
        ],
        graph=[],
    )

    config_store.save_sop(sop)

    # Retrieve by version
    retrieved = config_store.get_sop("test_sop", "1.0.0")
    assert retrieved is not None
    assert retrieved.id == "test_sop"
    assert retrieved.version == "1.0.0"

    # Retrieve latest
    latest = config_store.get_sop("test_sop")
    assert latest is not None
    assert latest.version == "1.0.0"


def test_multiple_sop_versions(config_store):
    """Test handling multiple versions of same SOP"""
    # Save v1
    sop_v1 = ProcessConfig(
        id="evolving_sop",
        version="1.0.0",
        models={},
        tools=[],
        roles=[],
        graph=[],
    )
    config_store.save_sop(sop_v1)

    # Save v2
    sop_v2 = ProcessConfig(
        id="evolving_sop",
        version="2.0.0",
        models={},
        tools=[],
        roles=[],
        graph=[],
        description="Updated version",
    )
    config_store.save_sop(sop_v2)

    # Get specific versions
    v1 = config_store.get_sop("evolving_sop", "1.0.0")
    v2 = config_store.get_sop("evolving_sop", "2.0.0")

    assert v1.version == "1.0.0"
    assert v2.version == "2.0.0"
    assert v2.description == "Updated version"

    # Get latest
    latest = config_store.get_sop("evolving_sop")
    assert latest.version == "2.0.0"


def test_save_and_get_prompt_genome(config_store):
    """Test saving and retrieving PromptGenome"""
    genome = PromptGenome(
        id="test_genome",
        version="1.0.0",
        rolePrompts={
            "agent_prompt": RolePrompt(
                id="agent_prompt",
                content="You are a helpful agent.",
            )
        },
    )

    config_store.save_prompt_genome(genome)

    retrieved = config_store.get_prompt_genome("test_genome", "1.0.0")
    assert retrieved is not None
    assert "agent_prompt" in retrieved.rolePrompts


def test_save_and_get_tool(config_store):
    """Test saving and retrieving ToolConfig"""
    tool = ToolConfig(
        id="vector_search",
        type=ToolType.VECTOR_SEARCH,
        config={
            "index_name": "my_index",
            "top_k": 5,
        },
    )

    config_store.save_tool(tool)

    retrieved = config_store.get_tool("vector_search")
    assert retrieved is not None
    assert retrieved.type == ToolType.VECTOR_SEARCH
    assert retrieved.config["top_k"] == 5


def test_list_sops(config_store):
    """Test listing SOPs"""
    sop1 = ProcessConfig(id="sop1", version="1.0.0", models={}, tools=[], roles=[], graph=[])
    sop2 = ProcessConfig(id="sop2", version="1.0.0", models={}, tools=[], roles=[], graph=[])

    config_store.save_sop(sop1)
    config_store.save_sop(sop2)

    sop_list = config_store.list_sops()
    assert len(sop_list) == 2
    assert ("sop1", "1.0.0") in sop_list
    assert ("sop2", "1.0.0") in sop_list


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
