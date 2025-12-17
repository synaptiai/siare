"""BEIR benchmark dataset loader.

BEIR (Benchmarking Information Retrieval) is the industry-standard
benchmark for zero-shot retrieval evaluation across 15 diverse datasets.

Source: https://github.com/beir-cellar/beir
"""

import logging
from pathlib import Path
from typing import Any

from siare.benchmarks.base import BenchmarkDataset, BenchmarkSample

logger = logging.getLogger(__name__)

# Check if beir library is available
_beir_available = False
try:
    from beir import util  # type: ignore[import-not-found]
    from beir.datasets.data_loader import GenericDataLoader  # type: ignore[import-not-found]

    _beir_available = True
except ImportError:
    logger.info("beir library not installed. Install with: pip install beir")


class BEIRDataset(BenchmarkDataset):
    """BEIR benchmark suite - 15 datasets for zero-shot retrieval.

    Supports corpus-level retrieval evaluation with qrels (relevance judgments).

    Example:
        >>> dataset = BEIRDataset(dataset_name="nq", max_samples=100)
        >>> dataset.load()
        >>> print(f"Corpus size: {len(dataset.corpus)}")
        >>> print(f"Queries: {len(dataset)}")
    """

    AVAILABLE_DATASETS = [
        "msmarco",
        "trec-covid",
        "nfcorpus",
        "nq",
        "hotpotqa",
        "fiqa",
        "arguana",
        "webis-touche2020",
        "cqadupstack",
        "quora",
        "dbpedia-entity",
        "scidocs",
        "fever",
        "climate-fever",
        "scifact",
    ]

    # BEIR download URLs (official mirrors)
    _BASE_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets"

    def __init__(
        self,
        dataset_name: str = "nq",
        split: str = "test",
        max_samples: int | None = None,
        data_dir: str | None = None,
    ) -> None:
        """Initialize BEIR dataset.

        Args:
            dataset_name: One of AVAILABLE_DATASETS
            split: Dataset split (test, dev, train)
            max_samples: Maximum queries to load
            data_dir: Directory to store downloaded data

        Raises:
            ValueError: If dataset_name is not valid
        """
        if dataset_name not in self.AVAILABLE_DATASETS:
            raise ValueError(
                f"Unknown BEIR dataset: {dataset_name}. "
                f"Available: {self.AVAILABLE_DATASETS}"
            )

        super().__init__(split=split, max_samples=max_samples)
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir) if data_dir else Path("datasets/beir")
        self._corpus: dict[str, str] = {}
        self._qrels: dict[str, dict[str, int]] = {}

    @property
    def name(self) -> str:
        """Return dataset name including specific BEIR dataset."""
        return f"beir-{self.dataset_name}"

    @property
    def corpus(self) -> dict[str, str]:
        """Full document corpus for retrieval evaluation.

        Returns:
            Dictionary mapping doc_id -> document text
        """
        if not self._loaded:
            self.load()
        return self._corpus

    @property
    def qrels(self) -> dict[str, dict[str, int]]:
        """Query-document relevance judgments (TREC format).

        Returns:
            Dictionary mapping query_id -> {doc_id: relevance_score}
        """
        if not self._loaded:
            self.load()
        return self._qrels

    def load(self) -> None:
        """Load BEIR dataset from official repository."""
        if not _beir_available:
            logger.warning("beir library not available, using empty dataset")
            self._samples = []
            self._loaded = True
            return

        try:
            # Download and extract dataset
            url = f"{self._BASE_URL}/{self.dataset_name}.zip"
            data_path = util.download_and_unzip(url, str(self.data_dir))

            # Load corpus, queries, and qrels
            corpus, queries, qrels = GenericDataLoader(data_path).load(split=self.split)

            # Store corpus (doc_id -> text)
            self._corpus = {}
            for doc_id, doc in corpus.items():
                text = doc.get("text", "")
                title = doc.get("title", "")
                self._corpus[doc_id] = f"{title}\n{text}" if title else text

            # Store qrels
            self._qrels = qrels

            # Convert queries to BenchmarkSamples
            samples: list[BenchmarkSample] = []
            for query_id, query_text in queries.items():
                if self.max_samples and len(samples) >= self.max_samples:
                    break

                # Get relevant documents as ground truth
                relevant_docs = qrels.get(query_id, {})
                ground_truth_docs: list[Any] = [
                    self._corpus.get(doc_id, "")
                    for doc_id, rel in relevant_docs.items()
                    if rel > 0
                ]

                samples.append(
                    BenchmarkSample(
                        id=query_id,
                        query=query_text,
                        ground_truth=ground_truth_docs[0] if ground_truth_docs else "",
                        ground_truth_context=ground_truth_docs,
                        metadata={
                            "dataset": self.dataset_name,
                            "relevant_doc_ids": list(relevant_docs.keys()),
                            "relevance_scores": relevant_docs,
                        },
                    )
                )

            self._samples = samples
            self._loaded = True
            logger.info(
                f"Loaded BEIR-{self.dataset_name}: "
                f"{len(samples)} queries, {len(self._corpus)} documents"
            )

        except Exception:
            logger.exception(f"Failed to load BEIR-{self.dataset_name}")
            self._samples = []
            self._loaded = True
