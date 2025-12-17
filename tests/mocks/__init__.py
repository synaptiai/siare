"""Test mocks - FOR TESTING ONLY

This module contains mock implementations that should ONLY be used in tests.
Production code must NEVER import from this module.
"""

from tests.mocks.mock_embeddings import mock_embedding
from tests.mocks.mock_llm_provider import MockLLMProvider


__all__ = ["MockLLMProvider", "mock_embedding"]
