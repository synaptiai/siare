"""Tests for VectorSearchAdapter with real embeddings"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest

from siare.adapters.vector_search import VectorSearchAdapter


class TestRealEmbeddings:
    """Test real embedding models integration"""

    def test_sentence_transformers_embedding_consistency(self):
        """Sentence-transformers should produce consistent embeddings for same text"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "sentence-transformers",
                "dimension": 384,
            }
        )
        adapter.initialize()

        text = "The quick brown fox jumps over the lazy dog"
        embedding1 = adapter._get_embedding(text)
        embedding2 = adapter._get_embedding(text)

        # Same text should produce identical embeddings
        assert embedding1 == embedding2
        assert len(embedding1) == 384
        assert all(isinstance(x, float) for x in embedding1)

    def test_sentence_transformers_semantic_similarity(self):
        """Similar texts should have higher similarity than dissimilar texts"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "sentence-transformers",
                "dimension": 384,
            }
        )
        adapter.initialize()

        # Similar texts
        text1 = "The cat sat on the mat"
        text2 = "A cat is sitting on a mat"

        # Dissimilar text
        text3 = "Quantum physics explains subatomic particles"

        emb1 = np.array(adapter._get_embedding(text1))
        emb2 = np.array(adapter._get_embedding(text2))
        emb3 = np.array(adapter._get_embedding(text3))

        # Calculate cosine similarities
        def cosine_sim(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

        sim_similar = cosine_sim(emb1, emb2)
        sim_dissimilar = cosine_sim(emb1, emb3)

        # Similar texts should have higher similarity
        assert sim_similar > sim_dissimilar
        assert sim_similar > 0.5  # Should be reasonably similar
        assert sim_dissimilar < 0.5  # Should be less similar

    def test_auto_detection_selects_sentence_transformers(self):
        """Auto-detection should prefer sentence-transformers when available"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "auto",
                "dimension": 384,
            }
        )
        adapter.initialize()

        # Should have selected sentence-transformers
        assert adapter.embedding_model == "sentence-transformers"

        # Should produce valid embeddings
        embedding = adapter._get_embedding("test text")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)

    def test_unsupported_embedding_model_raises_error(self):
        """Unsupported embedding model should raise ValueError"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "unsupported-model",
                "dimension": 128,
            }
        )
        adapter.initialize()

        with pytest.raises(ValueError, match="Unsupported embedding model"):
            adapter._get_embedding("test text")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"})
    def test_openai_embedding_with_api_key(self):
        """OpenAI embedding should work when API key is set"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "openai",
                "dimension": 1536,  # text-embedding-3-small dimension
            }
        )
        adapter.initialize()

        # Mock the OpenAI client
        with patch("siare.adapters.vector_search.OpenAI") as mock_openai_class:
            mock_client = Mock()
            mock_response = Mock()
            mock_response.data = [Mock(embedding=[0.1] * 1536)]
            mock_client.embeddings.create.return_value = mock_response
            mock_openai_class.return_value = mock_client

            embedding = adapter._get_embedding("test text")

            # Should have called OpenAI API
            mock_client.embeddings.create.assert_called_once_with(
                model="text-embedding-3-small", input="test text"
            )
            assert len(embedding) == 1536

    def test_openai_embedding_raises_without_api_key(self):
        """OpenAI embedding should raise error when API key is missing"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "openai",
                "dimension": 1536,
            }
        )
        adapter.initialize()

        # Ensure no API key in environment
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OPENAI_API_KEY environment variable not set"):
                adapter._get_embedding("test text")

    def test_unsupported_embedding_model_raises_error(self):
        """Unsupported embedding model should raise error"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "nonexistent-model",
                "dimension": 128,
            }
        )
        adapter.initialize()

        with pytest.raises(ValueError, match="Unsupported embedding model: nonexistent-model"):
            adapter._get_embedding("test text")


class TestEmbeddingIntegration:
    """Test embedding integration with search functionality"""

    def test_search_with_sentence_transformers(self):
        """End-to-end test: add documents and search with real embeddings"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "sentence-transformers",
                "dimension": 384,
            }
        )
        adapter.initialize()

        # Add documents
        docs = [
            {"id": "1", "text": "Python is a programming language"},
            {"id": "2", "text": "Java is also a programming language"},
            {"id": "3", "text": "The weather is sunny today"},
        ]
        adapter.add_documents(docs)

        # Search for programming-related content
        result = adapter.execute({"query": "programming languages", "top_k": 2})

        # Should find the programming-related documents
        assert result["count"] == 2
        assert result["results"][0]["id"] in ["1", "2"]
        assert result["results"][1]["id"] in ["1", "2"]
        # Weather document should not be in top 2
        assert all(r["id"] != "3" for r in result["results"])


class TestDimensionValidation:
    """Test dimension auto-correction for embedding models"""

    def test_sentence_transformers_dimension_auto_corrected(self):
        """Wrong dimension should be auto-corrected for sentence-transformers"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "sentence-transformers",
                "dimension": 512,  # Wrong dimension - model produces 384
            }
        )

        # Before initialization, dimension is user-provided
        assert adapter.dimension == 512

        # After initialization, dimension should be auto-corrected
        adapter.initialize()
        assert adapter.dimension == 384

        # Embeddings should match corrected dimension
        embedding = adapter._get_embedding("test")
        assert len(embedding) == 384

    def test_openai_dimension_auto_corrected(self):
        """Wrong dimension should be auto-corrected for OpenAI embeddings"""
        adapter = VectorSearchAdapter(
            config={
                "backend": "memory",
                "embedding_model": "openai",
                "dimension": 384,  # Wrong dimension - model produces 1536
            }
        )

        # Before initialization, dimension is user-provided
        assert adapter.dimension == 384

        # After initialization, dimension should be auto-corrected
        adapter.initialize()
        assert adapter.dimension == 1536
