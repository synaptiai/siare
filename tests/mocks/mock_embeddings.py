"""Mock Embeddings - FOR TESTING ONLY

This module provides mock embedding functions for unit tests.
These produce deterministic but meaningless vectors based on text hashes.
They should NEVER be used in production code.
"""

import numpy as np


def mock_embedding(text: str, dimension: int = 384) -> list[float]:
    """
    Generate mock embedding using hash - FOR TESTING ONLY.

    WARNING: This produces RANDOM vectors based on text hash.
    It does NOT capture semantic meaning and should NEVER be used in production.

    Args:
        text: Text to embed
        dimension: Embedding dimension (default: 384)

    Returns:
        List of floats representing the mock embedding
    """
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dimension).tolist()


def mock_embedding_array(text: str, dimension: int = 384) -> np.ndarray:
    """
    Generate mock embedding as numpy array - FOR TESTING ONLY.

    Args:
        text: Text to embed
        dimension: Embedding dimension (default: 384)

    Returns:
        Numpy array representing the mock embedding
    """
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dimension)
