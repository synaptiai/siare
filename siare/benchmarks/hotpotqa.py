"""HotpotQA benchmark dataset loader.

HotpotQA is a multi-hop question answering dataset requiring
reasoning over multiple documents.

See: https://hotpotqa.github.io/
"""
import logging

from siare.benchmarks.base import BenchmarkDataset, BenchmarkSample

logger = logging.getLogger(__name__)

# Check if datasets library is available
_datasets_available = False
try:
    from datasets import load_dataset  # type: ignore[import-untyped]

    _datasets_available = True
except ImportError:
    logger.info("datasets library not installed. Install with: pip install datasets")


class HotpotQADataset(BenchmarkDataset):
    """HotpotQA multi-hop question answering benchmark.

    This dataset contains questions that require reasoning over
    multiple Wikipedia paragraphs to answer.

    Example:
        >>> dataset = HotpotQADataset(split="validation", max_samples=100)
        >>> for sample in dataset:
        ...     print(f"Q: {sample.query}")
        ...     print(f"A: {sample.ground_truth}")
    """

    def __init__(
        self,
        split: str = "validation",
        max_samples: int | None = None,
        difficulty: str = "distractor",
    ) -> None:
        """Initialize HotpotQA dataset.

        Args:
            split: Dataset split (train, validation)
            max_samples: Maximum samples to load
            difficulty: "distractor" or "fullwiki" mode
        """
        super().__init__(split=split, max_samples=max_samples)
        self.difficulty = difficulty

    @property
    def name(self) -> str:
        """Return dataset name."""
        return "hotpotqa"

    def load(self) -> None:
        """Load HotpotQA samples from HuggingFace."""
        if not _datasets_available:
            logger.warning("datasets library not available, using empty dataset")
            self._samples = []
            self._loaded = True
            return

        try:
            dataset = load_dataset("hotpot_qa", self.difficulty, split=self.split)  # type: ignore[misc]

            samples: list[BenchmarkSample] = []
            for idx, item in enumerate(dataset):  # type: ignore[var-annotated]
                if self.max_samples and idx >= self.max_samples:
                    break

                # Extract supporting facts as context
                context = []
                for title, sentences in zip(  # type: ignore[arg-type]
                    item["context"]["title"], item["context"]["sentences"], strict=False  # type: ignore[index]
                ):
                    context.append(f"{title}: {' '.join(sentences)}")  # type: ignore[arg-type]

                sample = BenchmarkSample(
                    id=item["id"],  # type: ignore[arg-type]
                    query=item["question"],  # type: ignore[arg-type]
                    ground_truth=item["answer"],  # type: ignore[arg-type]
                    context=context,  # type: ignore[arg-type]
                    metadata={
                        "type": item.get("type", ""),  # type: ignore[dict-item]
                        "level": item.get("level", ""),  # type: ignore[dict-item]
                    },
                )
                samples.append(sample)

            self._samples = samples
            self._loaded = True
            logger.info(f"Loaded {len(samples)} HotpotQA samples")

        except Exception:
            logger.exception("Failed to load HotpotQA")
            self._samples = []
            self._loaded = True
