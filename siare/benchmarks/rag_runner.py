"""RAG-aware benchmark runner with retrieval metrics.

Extends the base BenchmarkRunner to:
1. Load corpus data before running benchmarks
2. Track retrieval quality (NDCG, Recall, MRR) separately from generation quality
3. Support qrels-based evaluation for BEIR-style benchmarks
"""

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from siare.benchmarks.metrics.integration import (
    RetrievalEvaluator,
    extract_retrieved_docs_from_trace,
)
from siare.benchmarks.runner import BenchmarkResults, BenchmarkRunner, SampleResult


if TYPE_CHECKING:
    from siare.benchmarks.base import BenchmarkDataset, BenchmarkSample
    from siare.benchmarks.corpus.loader import CorpusLoader
    from siare.core.models import ProcessConfig, PromptGenome
    from siare.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class RAGSampleResult(SampleResult):
    """Extended sample result with retrieval information."""

    retrieved_doc_ids: list[str] = field(default_factory=list)
    retrieval_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class RAGBenchmarkResults(BenchmarkResults):
    """Extended results with separate retrieval and generation metrics."""

    retrieval_metrics: dict[str, float] = field(default_factory=dict)
    generation_metrics: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return summary including retrieval metrics."""
        base = super().summary()
        base["retrieval_metrics"] = self.retrieval_metrics
        base["generation_metrics"] = self.generation_metrics
        return base


class RAGBenchmarkRunner(BenchmarkRunner):
    """Benchmark runner with RAG-specific features.

    Features:
    - Corpus loading and validation
    - Retrieval metrics computation (NDCG, Recall, MRR)
    - Separate tracking of retrieval vs generation quality
    - qrels integration for BEIR-style evaluation

    Example:
        >>> runner = RAGBenchmarkRunner(sop=rag_sop, genome=rag_genome)
        >>> runner.ensure_corpus_loaded("beir_nfcorpus", corpus_loader)
        >>> results = runner.run_with_retrieval_metrics(dataset, qrels)
    """

    def __init__(
        self,
        sop: Optional["ProcessConfig"] = None,
        genome: Optional["PromptGenome"] = None,
        llm_provider: Optional["LLMProvider"] = None,
        model_fallback_cascade: Optional[list[str]] = None,
        retrieval_k_values: Optional[list[int]] = None,
        tool_adapters: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize RAG benchmark runner.

        Args:
            sop: RAG SOP configuration
            genome: Prompt genome for the SOP
            llm_provider: LLM provider for execution
            model_fallback_cascade: Model fallback list
            retrieval_k_values: K values for retrieval metrics
            tool_adapters: Dict of {tool_name: callable} for tool execution
        """
        super().__init__(
            sop=sop,
            genome=genome,
            llm_provider=llm_provider,
            model_fallback_cascade=model_fallback_cascade,
        )
        self.retrieval_evaluator = RetrievalEvaluator(
            k_values=retrieval_k_values or [1, 3, 5, 10]
        )
        self._corpus_loaded: dict[str, bool] = {}
        self._tool_adapters = tool_adapters or {}

    def ensure_corpus_loaded(
        self,
        index_name: str,
        corpus_loader: "CorpusLoader",
    ) -> bool:
        """Check if corpus is loaded and accessible.

        Args:
            index_name: Name of the vector index
            corpus_loader: CorpusLoader instance

        Returns:
            True if corpus is ready, False otherwise
        """
        if index_name in self._corpus_loaded:
            return self._corpus_loaded[index_name]

        count = corpus_loader.get_document_count(index_name)
        if count > 0:
            logger.info(f"Corpus '{index_name}' ready with {count:,} documents")
            self._corpus_loaded[index_name] = True
            return True
        logger.warning(f"Corpus '{index_name}' is empty or not found")
        self._corpus_loaded[index_name] = False
        return False

    def run_with_retrieval_metrics(
        self,
        dataset: "BenchmarkDataset",
        qrels: Optional[dict[str, dict[str, int]]] = None,
        max_samples: Optional[int] = None,
    ) -> RAGBenchmarkResults:
        """Run benchmark with retrieval metrics tracking.

        Args:
            dataset: Benchmark dataset to evaluate
            qrels: Query relevance judgments {query_id: {doc_id: score}}
            max_samples: Maximum samples to run

        Returns:
            RAGBenchmarkResults with retrieval and generation metrics
        """
        start_time = time.time()
        sample_results: list[RAGSampleResult] = []
        retrieval_results: dict[str, list[str]] = {}
        completed = 0
        failed = 0

        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]

        for sample in samples:
            try:
                result = self._run_rag_sample(sample)
                sample_results.append(result)

                if result.error:
                    failed += 1
                else:
                    completed += 1
                    # Track retrieval results for aggregation
                    if result.retrieved_doc_ids:
                        retrieval_results[sample.id] = result.retrieved_doc_ids

            except Exception as e:
                logger.exception(f"Error running sample {sample.id}")
                sample_results.append(
                    RAGSampleResult(
                        sample_id=sample.id,
                        query=sample.query,
                        generated_answer="",
                        ground_truth=sample.ground_truth,
                        error=str(e),
                    )
                )
                failed += 1

        # Compute retrieval metrics if qrels provided
        retrieval_metrics: dict[str, float] = {}
        if qrels and retrieval_results:
            retrieval_metrics = self.retrieval_evaluator.evaluate(
                retrieval_results, qrels
            )
            logger.info(f"Retrieval metrics: {retrieval_metrics}")

        # Compute generation metrics (from sample results)
        generation_metrics = self._compute_generation_metrics(sample_results)

        # Combined aggregate metrics
        aggregate_metrics = {**retrieval_metrics, **generation_metrics}

        return RAGBenchmarkResults(
            dataset_name=dataset.name,
            total_samples=len(samples),
            completed_samples=completed,
            failed_samples=failed,
            retrieval_metrics=retrieval_metrics,
            generation_metrics=generation_metrics,
            aggregate_metrics=aggregate_metrics,
            sample_results=sample_results,  # type: ignore
            total_time_seconds=time.time() - start_time,
        )

    def _run_rag_sample(self, sample: "BenchmarkSample") -> RAGSampleResult:
        """Run a single RAG benchmark sample.

        Args:
            sample: The benchmark sample to evaluate

        Returns:
            RAGSampleResult with retrieval information
        """
        from siare.services.execution_engine import ExecutionEngine

        start = time.time()

        if self._sop is None or self._genome is None:
            return RAGSampleResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer="",
                ground_truth=sample.ground_truth,
                error="No SOP or genome configured",
            )

        try:
            engine = ExecutionEngine(
                llm_provider=self._llm_provider,
                model_fallback_cascade=self._model_fallback_cascade,
                tool_adapters=self._tool_adapters,
            )
            task_input = sample.to_task_data()["input"]

            trace = engine.execute(
                sop=self._sop,
                prompt_genome=self._genome,
                task_input=task_input,
            )

            # Extract answer
            generated_answer = ""
            for role_outputs in trace.final_outputs.values():
                if isinstance(role_outputs, dict):
                    generated_answer = role_outputs.get(
                        "answer", role_outputs.get("response", "")
                    )
                    if generated_answer:
                        break

            if not generated_answer:
                generated_answer = trace.final_outputs.get(
                    "answer", trace.final_outputs.get("response", "")
                )

            # Extract retrieved doc IDs
            retrieved_doc_ids = extract_retrieved_docs_from_trace(trace)

            latency = (time.time() - start) * 1000

            return RAGSampleResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer=generated_answer,
                ground_truth=sample.ground_truth,
                latency_ms=latency,
                retrieved_doc_ids=retrieved_doc_ids,
                execution_trace=trace,
            )

        except Exception as e:
            logger.exception(f"RAG execution failed for sample {sample.id}")
            return RAGSampleResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer="",
                ground_truth=sample.ground_truth,
                error=str(e),
            )

    def _compute_generation_metrics(
        self,
        results: list[RAGSampleResult],
    ) -> dict[str, float]:
        """Compute generation quality metrics.

        Args:
            results: List of sample results

        Returns:
            Dict of generation metrics
        """
        if not results:
            return {}

        # Simple exact match accuracy
        correct = 0
        total = 0

        for result in results:
            if result.error:
                continue

            total += 1
            # Normalize for comparison
            predicted = result.generated_answer.strip().lower()
            expected = result.ground_truth.strip().lower()

            if predicted == expected or expected in predicted:
                correct += 1

        accuracy = correct / total if total > 0 else 0.0

        return {
            "accuracy": accuracy,
            "exact_match": accuracy,
            "total_evaluated": float(total),
        }
