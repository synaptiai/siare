"""FRAMES benchmark dataset loader.

FRAMES (Factuality, Retrieval, And reasoning MEasurement Set) is a
comprehensive benchmark for end-to-end RAG evaluation with multi-hop
reasoning.

Source: https://huggingface.co/datasets/google/frames-benchmark
Paper: https://arxiv.org/abs/2409.12941
"""

import logging
from typing import Any, Optional

from siare.benchmarks.base import BenchmarkDataset, BenchmarkSample


logger = logging.getLogger(__name__)

# Check if datasets library is available
_datasets_available = False
try:
    from datasets import load_dataset  # type: ignore[import-untyped]

    _datasets_available = True
except ImportError:
    logger.info("datasets library not installed. Install with: pip install datasets")


class FRAMESDataset(BenchmarkDataset):
    """FRAMES multi-hop RAG benchmark from Google Research.

    824 challenging multi-hop questions requiring 2-15 Wikipedia articles.
    Each question is labeled with reasoning types and includes gold answers.

    Example:
        >>> dataset = FRAMESDataset(max_samples=100)
        >>> for sample in dataset:
        ...     print(f"Q: {sample.query}")
        ...     print(f"Type: {sample.metadata['reasoning_type']}")
    """

    REASONING_TYPES = [
        "numerical",
        "tabular",
        "multiple_constraints",
        "temporal",
        "post_processing",
    ]

    # Published baselines from Google paper for comparison
    PUBLISHED_BASELINES = {
        "naive_prompting": 0.408,
        "bm25_4docs": 0.474,
        "multi_step_sota": 0.660,
        "oracle_retrieval": 0.729,
    }

    def __init__(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
        reasoning_type: Optional[str] = None,
    ) -> None:
        """Initialize FRAMES dataset.

        Args:
            split: Dataset split (default: test)
            max_samples: Maximum samples to load
            reasoning_type: Filter by reasoning type (optional)
        """
        super().__init__(split=split, max_samples=max_samples)
        self.reasoning_type = reasoning_type

    @property
    def name(self) -> str:
        """Return dataset name."""
        return "frames"

    def load(self) -> None:
        """Load FRAMES samples from HuggingFace."""
        if not _datasets_available:
            logger.warning("datasets library not available, using empty dataset")
            self._samples = []
            self._loaded = True
            return

        try:
            # Load from the Parquet conversion branch to avoid trust_remote_code deprecation
            # See: https://huggingface.co/datasets/google/frames-benchmark/tree/refs%2Fconvert%2Fparquet
            dataset = load_dataset(
                "google/frames-benchmark",
                split=self.split,
            )

            samples: list[BenchmarkSample] = []
            for idx, item in enumerate(dataset):  # type: ignore[var-annotated]
                if self.max_samples and idx >= self.max_samples:
                    break

                # Extract reasoning type from item
                reasoning_types = item.get("reasoning_types", [])
                reasoning_type = (
                    reasoning_types[0] if reasoning_types else "unknown"
                )

                # Filter by reasoning type if specified
                if self.reasoning_type and reasoning_type != self.reasoning_type:
                    continue

                # Extract supporting facts as context
                wiki_links: list[Any] = item.get("wiki_links", [])

                sample = BenchmarkSample(
                    id=str(idx),
                    query=item["Prompt"],
                    ground_truth=item["Answer"],
                    ground_truth_context=wiki_links,
                    metadata={
                        "reasoning_type": reasoning_type,
                        "num_sources": len(wiki_links),
                    },
                )
                samples.append(sample)

            self._samples = samples
            self._loaded = True
            logger.info(f"Loaded {len(samples)} FRAMES samples")

        except Exception:
            logger.exception("Failed to load FRAMES dataset")
            self._samples = []
            self._loaded = True
