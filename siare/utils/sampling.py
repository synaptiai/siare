"""Sampling utilities for quality-weighted selection"""

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


def quality_weighted_sample(
    items: Sequence[T],
    count: int,
    quality_extractor: Callable[[T], float],
) -> list[T]:
    """
    Sample items using quality-weighted softmax probabilities.

    Applies softmax normalization with numerical stability to avoid overflow
    when computing exponentials. Higher quality items have higher probability
    of being selected.

    Args:
        items: Collection to sample from
        count: Number of items to sample
        quality_extractor: Function to extract quality score from each item

    Returns:
        List of sampled items (without replacement)

    Note:
        Returns empty list if items is empty
        Returns fewer than count if len(items) < count
        All quality scores are normalized via softmax, so negative values are valid

    Example:
        >>> genes = [SOPGene(...), ...]
        >>> sampled = quality_weighted_sample(
        ...     genes,
        ...     count=5,
        ...     quality_extractor=lambda g: g.aggregatedMetrics.get("weighted_aggregate").mean
        ... )
    """
    if not items:
        return []

    # Extract quality scores
    qualities: NDArray[np.floating[Any]] = np.array(
        [quality_extractor(item) for item in items]
    )

    # Softmax normalization with numerical stability
    # Subtracting max prevents overflow in exp()
    exp_q: NDArray[np.floating[Any]] = np.exp(qualities - np.max(qualities))
    probs: NDArray[np.floating[Any]] = exp_q / exp_q.sum()

    # Sample without replacement
    n: int = min(count, len(items))
    indices: NDArray[np.intp] = np.random.choice(len(items), size=n, p=probs, replace=False)

    return [items[i] for i in indices]
