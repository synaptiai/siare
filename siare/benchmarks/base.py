"""Base classes for benchmark datasets."""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional


if TYPE_CHECKING:
    from siare.core.models import Task


def _empty_list() -> list[str]:
    """Create empty list for default factory."""
    return []


def _empty_dict() -> dict[str, Any]:
    """Create empty dict for default factory."""
    return {}


@dataclass
class BenchmarkSample:
    """Single benchmark sample for RAG evaluation.

    Attributes:
        id: Unique identifier for the sample
        query: The question or query to answer
        ground_truth: Expected answer
        context: Optional retrieved context documents
        ground_truth_context: Optional gold-standard context
        metadata: Additional sample metadata
    """

    id: str
    query: str
    ground_truth: str
    context: list[str] = field(default_factory=_empty_list)
    ground_truth_context: list[str] = field(default_factory=_empty_list)
    metadata: dict[str, Any] = field(default_factory=_empty_dict)

    def to_task_data(self) -> dict[str, Any]:
        """Convert to SIARE task_data format.

        Returns:
            Dictionary compatible with SIARE execution engine
        """
        return {
            "input": {"query": self.query},
            "groundTruth": {
                "answer": self.ground_truth,
                "context": self.ground_truth_context or self.context,
            },
            "metadata": self.metadata,
        }

    def to_task(self, weight: float = 1.0) -> "Task":
        """Convert to SIARE Task object for evolution loop.

        Args:
            weight: Task weight for evaluation (default 1.0)

        Returns:
            Task object compatible with EvolutionScheduler
        """
        from siare.core.models import Task, TaskMetadata

        return Task(
            id=self.id,
            input={"query": self.query},
            groundTruth={
                "answer": self.ground_truth,
                "context": self.ground_truth_context or self.context,
            },
            metadata=TaskMetadata(
                domain=self.metadata.get("domain", "benchmark"),
                difficulty=self.metadata.get("difficulty", "medium"),
                tags=self.metadata.get("tags", []),
            )
            if self.metadata
            else None,
            weight=weight,
        )


class BenchmarkDataset(ABC):
    """Abstract base class for benchmark datasets.

    Subclasses must implement:
    - name property: Return dataset name
    - load method: Load samples into self._samples

    Example:
        >>> class MyDataset(BenchmarkDataset):
        ...     @property
        ...     def name(self) -> str:
        ...         return "my_dataset"
        ...
        ...     def load(self) -> None:
        ...         self._samples = [...]
        ...         self._loaded = True
        >>>
        >>> dataset = MyDataset(split="test", max_samples=100)
        >>> for sample in dataset:
        ...     print(sample.query)
    """

    def __init__(
        self,
        split: str = "test",
        max_samples: Optional[int] = None,
    ) -> None:
        """Initialize the benchmark dataset.

        Args:
            split: Dataset split to use (train, validation, test)
            max_samples: Maximum number of samples to load
        """
        self.split = split
        self.max_samples = max_samples
        self._samples: list[BenchmarkSample] = []
        self._loaded = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name.

        Returns:
            String identifier for this dataset
        """

    @abstractmethod
    def load(self) -> None:
        """Load dataset samples.

        Implementations should populate self._samples and set
        self._loaded = True.
        """

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        if not self._loaded:
            self.load()
        return len(self._samples)

    def __iter__(self) -> Iterator[BenchmarkSample]:
        """Iterate over samples in the dataset."""
        if not self._loaded:
            self.load()
        return iter(self._samples)

    def __getitem__(self, idx: int) -> BenchmarkSample:
        """Get sample by index."""
        if not self._loaded:
            self.load()
        return self._samples[idx]
