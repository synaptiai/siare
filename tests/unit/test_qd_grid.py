"""Tests for QD Grid core functionality and embeddings"""

import numpy as np
import pytest

from siare.core.models import (
    GraphEdge,
    ProcessConfig,
    PromptGenome,
    RoleConfig,
    RolePrompt,
)
from siare.services.qd_grid import (
    _simple_text_embedding,
    calculate_complexity,
    calculate_diversity_embedding,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_sop():
    """Create a simple SOP for testing"""
    roles = [
        RoleConfig(
            id="researcher",
            model="gpt-4",
            tools=None,
            promptRef="researcher_prompt",
        ),
        RoleConfig(
            id="writer",
            model="gpt-4",
            tools=None,
            promptRef="writer_prompt",
        ),
    ]

    graph = [
        GraphEdge(from_="user_input", to="researcher"),
        GraphEdge(from_="researcher", to="writer"),
    ]

    return ProcessConfig(
        id="research_sop",
        version="1.0.0",
        models={"default": "gpt-4"},
        tools=["vector_search"],
        roles=roles,
        graph=graph,
    )


@pytest.fixture
def sample_genome():
    """Create a simple prompt genome"""
    prompts = {
        "researcher_prompt": RolePrompt(
            id="researcher_prompt",
            content="You are a research assistant. Find relevant information about the topic.",
        ),
        "writer_prompt": RolePrompt(
            id="writer_prompt",
            content="You are a writer. Synthesize the research into a coherent document.",
        ),
    }

    return PromptGenome(
        id="research_genome",
        version="1.0.0",
        rolePrompts=prompts,
    )


# ============================================================================
# Embedding Tests
# ============================================================================


def test_embedding_consistency():
    """Test that same text produces same embedding (deterministic)"""
    # This test will fail with hash-based random embeddings
    # because np.random.randn() with same seed should produce same result
    # but we want to ensure REAL embeddings are deterministic

    text = "You are a helpful research assistant."

    # Generate embedding twice
    emb1 = _simple_text_embedding(text, dim=384)
    emb2 = _simple_text_embedding(text, dim=384)

    # Assert embeddings are identical
    np.testing.assert_array_almost_equal(
        emb1, emb2,
        decimal=5,
        err_msg="Same text should produce identical embeddings"
    )


def test_embedding_different_texts():
    """Test that different texts produce different embeddings"""
    text1 = "You are a research assistant."
    text2 = "You are a creative writer."

    emb1 = _simple_text_embedding(text1, dim=384)
    emb2 = _simple_text_embedding(text2, dim=384)

    # Calculate cosine similarity
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Embeddings should be different (not perfectly similar)
    assert cos_sim < 0.99, "Different texts should produce different embeddings"


def test_embedding_dimension():
    """Test that embedding has correct dimension"""
    text = "Test prompt for dimension check"

    emb = _simple_text_embedding(text, dim=384)

    assert emb.shape == (384,), f"Expected shape (384,), got {emb.shape}"


def test_embedding_semantic_similarity():
    """Test that semantically similar texts have higher similarity"""
    # Similar texts
    text1 = "You are a research assistant helping with academic papers."
    text2 = "You are a research helper for scholarly articles."

    # Different texts
    text3 = "You are a creative writer focusing on fiction."

    emb1 = _simple_text_embedding(text1, dim=384)
    emb2 = _simple_text_embedding(text2, dim=384)
    emb3 = _simple_text_embedding(text3, dim=384)

    # Normalize for cosine similarity
    emb1_norm = emb1 / np.linalg.norm(emb1)
    emb2_norm = emb2 / np.linalg.norm(emb2)
    emb3_norm = emb3 / np.linalg.norm(emb3)

    # Similarity between similar texts
    sim_similar = np.dot(emb1_norm, emb2_norm)

    # Similarity between different texts
    sim_diff_1_3 = np.dot(emb1_norm, emb3_norm)
    sim_diff_2_3 = np.dot(emb2_norm, emb3_norm)

    # Similar texts should have higher similarity
    assert sim_similar > sim_diff_1_3, (
        f"Similar texts should have higher similarity: {sim_similar:.3f} vs {sim_diff_1_3:.3f}"
    )
    assert sim_similar > sim_diff_2_3, (
        f"Similar texts should have higher similarity: {sim_similar:.3f} vs {sim_diff_2_3:.3f}"
    )


def test_embedding_stability_across_calls():
    """Test that embeddings are stable across multiple function calls"""
    text = "Stable embedding test"

    embeddings = [_simple_text_embedding(text, dim=384) for _ in range(5)]

    # All embeddings should be identical
    for i in range(1, len(embeddings)):
        np.testing.assert_array_almost_equal(
            embeddings[0], embeddings[i],
            decimal=5,
            err_msg=f"Embedding {i} differs from embedding 0"
        )


# ============================================================================
# Diversity Embedding Tests
# ============================================================================


def test_diversity_embedding_dimension(sample_sop, sample_genome):
    """Test that diversity embedding has correct dimension"""
    emb = calculate_diversity_embedding(sample_sop, sample_genome)

    # Should be normalized to 384 dimensions
    assert emb.shape == (384,), f"Expected shape (384,), got {emb.shape}"


def test_diversity_embedding_normalized(sample_sop, sample_genome):
    """Test that diversity embedding is normalized to unit length"""
    emb = calculate_diversity_embedding(sample_sop, sample_genome)

    norm = np.linalg.norm(emb)

    assert abs(norm - 1.0) < 1e-6, f"Embedding should be unit normalized, got norm={norm}"


def test_diversity_embedding_different_sops(sample_sop, sample_genome):
    """Test that different SOPs produce different embeddings"""
    # First embedding
    emb1 = calculate_diversity_embedding(sample_sop, sample_genome)

    # Create modified SOP with different prompts
    modified_genome = PromptGenome(
        id="modified_genome",
        version="1.0.0",
        rolePrompts={
            "researcher_prompt": RolePrompt(
                id="researcher_prompt",
                content="You are a data analyst. Analyze the dataset thoroughly.",
            ),
            "writer_prompt": RolePrompt(
                id="writer_prompt",
                content="You are a technical writer. Document the findings.",
            ),
        },
    )

    emb2 = calculate_diversity_embedding(sample_sop, modified_genome)

    # Embeddings should be different
    cos_sim = np.dot(emb1, emb2)  # Already normalized

    assert cos_sim < 0.99, "Different prompts should produce different embeddings"


# ============================================================================
# Complexity Tests
# ============================================================================


def test_calculate_complexity_simple(sample_sop, sample_genome):
    """Test complexity calculation on simple SOP"""
    complexity = calculate_complexity(sample_sop, sample_genome)

    # Should be a float in [0, 1]
    assert isinstance(complexity, float), f"Complexity should be float, got {type(complexity)}"
    assert 0.0 <= complexity <= 1.0, f"Complexity should be in [0, 1], got {complexity}"


def test_calculate_complexity_increases_with_roles():
    """Test that complexity increases with more roles"""
    # Simple SOP with 2 roles
    simple_sop = ProcessConfig(
        id="simple",
        version="1.0.0",
        models={"default": "gpt-4"},
        tools=[],
        roles=[
            RoleConfig(id="role1", model="gpt-4", tools=None, promptRef="p1"),
            RoleConfig(id="role2", model="gpt-4", tools=None, promptRef="p2"),
        ],
        graph=[
            GraphEdge(from_="user_input", to="role1"),
            GraphEdge(from_="role1", to="role2"),
        ],
    )

    # Complex SOP with 5 roles
    complex_sop = ProcessConfig(
        id="complex",
        version="1.0.0",
        models={"default": "gpt-4"},
        tools=[],
        roles=[
            RoleConfig(id=f"role{i}", model="gpt-4", tools=None, promptRef=f"p{i}")
            for i in range(5)
        ],
        graph=[
            GraphEdge(from_="user_input", to="role0"),
        ] + [
            GraphEdge(from_=f"role{i}", to=f"role{i+1}")
            for i in range(4)
        ],
    )

    # Genomes with same prompt length
    simple_genome = PromptGenome(
        id="simple_g",
        version="1.0.0",
        rolePrompts={
            "p1": RolePrompt(id="p1", content="Test prompt"),
            "p2": RolePrompt(id="p2", content="Test prompt"),
        },
    )

    complex_genome = PromptGenome(
        id="complex_g",
        version="1.0.0",
        rolePrompts={
            f"p{i}": RolePrompt(id=f"p{i}", content="Test prompt")
            for i in range(5)
        },
    )

    simple_complexity = calculate_complexity(simple_sop, simple_genome)
    complex_complexity = calculate_complexity(complex_sop, complex_genome)

    assert complex_complexity > simple_complexity, (
        f"More roles should increase complexity: {simple_complexity} vs {complex_complexity}"
    )


# ============================================================================
# Edge Cases
# ============================================================================


def test_embedding_empty_text():
    """Test embedding generation with empty text"""
    emb = _simple_text_embedding("", dim=384)

    # Should still produce valid embedding
    assert emb.shape == (384,), "Empty text should produce valid embedding"
    assert not np.all(emb == 0), "Embedding should not be all zeros"


def test_embedding_long_text():
    """Test embedding generation with very long text"""
    long_text = "This is a test. " * 1000  # 16k characters

    emb = _simple_text_embedding(long_text, dim=384)

    # Should handle long text gracefully
    assert emb.shape == (384,), "Long text should produce valid embedding"
    assert np.isfinite(emb).all(), "Embedding should contain finite values"


def test_embedding_special_characters():
    """Test embedding with special characters"""
    text = "Special chars: @#$%^&*()_+{}|:<>?[];',./~`"

    emb = _simple_text_embedding(text, dim=384)

    assert emb.shape == (384,), "Special chars should produce valid embedding"
    assert np.isfinite(emb).all(), "Embedding should contain finite values"
