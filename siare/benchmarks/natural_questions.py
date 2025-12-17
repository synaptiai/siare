"""Natural Questions benchmark dataset loader.

Natural Questions is a large-scale QA dataset from Google,
containing real user questions with answers from Wikipedia.

See: https://ai.google.com/research/NaturalQuestions
"""
import logging

from siare.benchmarks.base import BenchmarkDataset, BenchmarkSample

logger = logging.getLogger(__name__)

_datasets_available = False
try:
    from datasets import load_dataset  # type: ignore[import-untyped]

    _datasets_available = True
except ImportError:
    logger.info("datasets library not installed")


class NaturalQuestionsDataset(BenchmarkDataset):
    """Natural Questions QA benchmark.

    Contains real Google search questions with Wikipedia answers.

    Example:
        >>> dataset = NaturalQuestionsDataset(split="validation", max_samples=100)
        >>> for sample in dataset:
        ...     print(f"Q: {sample.query}")
    """

    def __init__(
        self,
        split: str = "validation",
        max_samples: int | None = None,
    ) -> None:
        """Initialize Natural Questions dataset.

        Args:
            split: Dataset split (train, validation)
            max_samples: Maximum samples to load
        """
        super().__init__(split=split, max_samples=max_samples)

    @property
    def name(self) -> str:
        """Return dataset name."""
        return "natural_questions"

    def load(self) -> None:
        """Load Natural Questions samples from HuggingFace."""
        if not _datasets_available:
            logger.warning("datasets library not available, using empty dataset")
            self._samples = []
            self._loaded = True
            return

        try:
            # Use the simplified version
            dataset = load_dataset(  # type: ignore[misc]
                "google-research-datasets/natural_questions",
                "default",
                split=self.split,
            )

            samples: list[BenchmarkSample] = []
            for idx, item in enumerate(dataset):  # type: ignore[var-annotated]
                if self.max_samples and idx >= self.max_samples:
                    break

                # Extract short answer if available
                short_answers = item.get("annotations", {}).get("short_answers", [])  # type: ignore[union-attr]
                answer = short_answers[0] if short_answers else item.get("answer", "")  # type: ignore[index]

                # Document content as context
                context = [item.get("document", {}).get("html", "")]  # type: ignore[union-attr]

                sample = BenchmarkSample(
                    id=str(item.get("id", idx)),  # type: ignore[arg-type]
                    query=item.get("question", {}).get("text", ""),  # type: ignore[union-attr]
                    ground_truth=str(answer),  # type: ignore[arg-type]
                    context=context,  # type: ignore[arg-type]
                    metadata={
                        "document_title": item.get("document", {}).get("title", ""),  # type: ignore[union-attr]
                    },
                )
                samples.append(sample)

            self._samples = samples
            self._loaded = True
            logger.info(f"Loaded {len(samples)} Natural Questions samples")

        except Exception:
            logger.exception("Failed to load Natural Questions")
            self._samples = []
            self._loaded = True
