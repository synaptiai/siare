"""Benchmark runner for executing RAG evaluations."""
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from siare.benchmarks.base import BenchmarkDataset, BenchmarkSample
    from siare.core.models import EvaluationVector, ProcessConfig, PromptGenome
    from siare.services.evaluation_service import EvaluationService
    from siare.services.execution_engine import ExecutionTrace
    from siare.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


def _empty_float_dict() -> dict[str, float]:
    """Factory for empty metrics dict."""
    return {}


def _empty_sample_list() -> list["SampleResult"]:
    """Factory for empty sample results list."""
    return []


@dataclass
class SampleResult:
    """Result of running a single benchmark sample."""

    sample_id: str
    query: str
    generated_answer: str
    ground_truth: str
    metrics: dict[str, float] = field(default_factory=_empty_float_dict)
    latency_ms: float = 0.0
    error: str | None = None
    # Evolution integration fields
    evaluation_vector: Optional["EvaluationVector"] = None
    execution_trace: Optional["ExecutionTrace"] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for checkpointing.

        Note: evaluation_vector and execution_trace are not serialized
        as they contain complex objects. They will be None on restore.
        """
        return {
            "sample_id": self.sample_id,
            "query": self.query,
            "generated_answer": self.generated_answer,
            "ground_truth": self.ground_truth,
            "metrics": self.metrics,
            "latency_ms": self.latency_ms,
            "error": self.error,
            # Complex objects not serialized - will be None on restore
            "evaluation_vector": None,
            "execution_trace": None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SampleResult":
        """Deserialize from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            query=data["query"],
            generated_answer=data["generated_answer"],
            ground_truth=data["ground_truth"],
            metrics=data.get("metrics", {}),
            latency_ms=data.get("latency_ms", 0.0),
            error=data.get("error"),
            evaluation_vector=None,  # Cannot restore complex objects
            execution_trace=None,
        )


@dataclass
class BenchmarkResults:
    """Aggregated results from a benchmark run."""

    dataset_name: str
    total_samples: int
    completed_samples: int
    failed_samples: int
    aggregate_metrics: dict[str, float] = field(default_factory=_empty_float_dict)
    sample_results: list[SampleResult] = field(default_factory=_empty_sample_list)
    total_time_seconds: float = 0.0

    def summary(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "dataset": self.dataset_name,
            "total_samples": self.total_samples,
            "completed": self.completed_samples,
            "failed": self.failed_samples,
            "success_rate": self.completed_samples / self.total_samples if self.total_samples else 0,
            "metrics": self.aggregate_metrics,
            "total_time_seconds": self.total_time_seconds,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for checkpointing."""
        return {
            "dataset_name": self.dataset_name,
            "total_samples": self.total_samples,
            "completed_samples": self.completed_samples,
            "failed_samples": self.failed_samples,
            "aggregate_metrics": self.aggregate_metrics,
            "sample_results": [sr.to_dict() for sr in self.sample_results],
            "total_time_seconds": self.total_time_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResults":
        """Deserialize from dictionary."""
        sample_results = [
            SampleResult.from_dict(sr) for sr in data.get("sample_results", [])
        ]
        return cls(
            dataset_name=data["dataset_name"],
            total_samples=data["total_samples"],
            completed_samples=data["completed_samples"],
            failed_samples=data["failed_samples"],
            aggregate_metrics=data.get("aggregate_metrics", {}),
            sample_results=sample_results,
            total_time_seconds=data.get("total_time_seconds", 0.0),
        )


class BenchmarkRunner:
    """Runs benchmark datasets through RAG pipelines.

    Example:
        >>> runner = BenchmarkRunner(sop=my_sop, genome=my_genome)
        >>> results = runner.run(HotpotQADataset(max_samples=100))
        >>> print(results.summary())
    """

    def __init__(
        self,
        sop: Optional["ProcessConfig"] = None,
        genome: Optional["PromptGenome"] = None,
        llm_provider: Optional["LLMProvider"] = None,
        model_fallback_cascade: list[str] | None = None,
        tool_adapters: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the benchmark runner.

        Args:
            sop: Process configuration for execution
            genome: Prompt genome for execution
            llm_provider: LLM provider for evaluation metrics
            model_fallback_cascade: List of models to try on failure (prevents
                cross-provider fallback when using local models like Ollama)
            tool_adapters: Dictionary mapping tool names to adapter functions
                (e.g., {"vector_search": adapter.execute})
        """
        self._sop = sop
        self._genome = genome
        self._llm_provider = llm_provider
        self._model_fallback_cascade = model_fallback_cascade
        self._tool_adapters = tool_adapters or {}

    def run(
        self,
        dataset: "BenchmarkDataset",
        max_samples: int | None = None,
        metrics: list[str] | None = None,
    ) -> BenchmarkResults:
        """Run benchmark on dataset.

        Args:
            dataset: Benchmark dataset to evaluate
            max_samples: Maximum samples to run (overrides dataset limit)
            metrics: List of metric IDs to evaluate

        Returns:
            BenchmarkResults with aggregate and per-sample metrics
        """
        start_time = time.time()
        sample_results: list[SampleResult] = []
        completed = 0
        failed = 0

        # Apply max_samples limit
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]

        for sample in samples:
            try:
                result = self._run_sample(sample, metrics or [])
                sample_results.append(result)
                if result.error:
                    failed += 1
                else:
                    completed += 1
            except Exception as e:
                logger.exception(f"Error running sample {sample.id}")
                sample_results.append(
                    SampleResult(
                        sample_id=sample.id,
                        query=sample.query,
                        generated_answer="",
                        ground_truth=sample.ground_truth,
                        error=str(e),
                    )
                )
                failed += 1

        # Aggregate metrics
        aggregate = self._aggregate_metrics(sample_results)

        return BenchmarkResults(
            dataset_name=dataset.name,
            total_samples=len(samples),
            completed_samples=completed,
            failed_samples=failed,
            aggregate_metrics=aggregate,
            sample_results=sample_results,
            total_time_seconds=time.time() - start_time,
        )

    def run_parallel(
        self,
        dataset: "BenchmarkDataset",
        max_samples: int | None = None,
        metrics: list[str] | None = None,
        max_workers: int = 1,
    ) -> BenchmarkResults:
        """Run benchmark on dataset with parallel sample processing.

        Args:
            dataset: Benchmark dataset to evaluate
            max_samples: Maximum samples to run (overrides dataset limit)
            metrics: List of metric IDs to evaluate
            max_workers: Number of parallel workers (default: 1 = sequential)

        Returns:
            BenchmarkResults with aggregate and per-sample metrics
        """
        # If max_workers is 1, fall back to sequential processing
        if max_workers <= 1:
            return self.run(dataset=dataset, max_samples=max_samples, metrics=metrics)

        start_time = time.time()
        sample_results: list[SampleResult] = []
        completed = 0
        failed = 0

        # Apply max_samples limit
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]

        logger.info(f"Running {len(samples)} samples with {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(self._run_sample, sample, metrics or []): sample
                for sample in samples
            }

            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    sample_results.append(result)
                    if result.error:
                        failed += 1
                    else:
                        completed += 1
                except Exception as e:
                    logger.exception(f"Error running sample {sample.id}")
                    sample_results.append(
                        SampleResult(
                            sample_id=sample.id,
                            query=sample.query,
                            generated_answer="",
                            ground_truth=sample.ground_truth,
                            error=str(e),
                        )
                    )
                    failed += 1

        # Sort results back to original order
        sample_id_order = {s.id: i for i, s in enumerate(samples)}
        sample_results.sort(key=lambda r: sample_id_order.get(r.sample_id, 999999))

        # Aggregate metrics
        aggregate = self._aggregate_metrics(sample_results)

        return BenchmarkResults(
            dataset_name=dataset.name,
            total_samples=len(samples),
            completed_samples=completed,
            failed_samples=failed,
            aggregate_metrics=aggregate,
            sample_results=sample_results,
            total_time_seconds=time.time() - start_time,
        )

    def _run_sample(
        self,
        sample: "BenchmarkSample",
        metrics: list[str],
    ) -> SampleResult:
        """Run a single benchmark sample.

        Args:
            sample: The benchmark sample to evaluate
            metrics: List of metric IDs to evaluate (future use)

        Returns:
            SampleResult with generated answer and metrics
        """
        from siare.services.execution_engine import ExecutionEngine

        start = time.time()

        # Execute using ExecutionEngine if SOP and genome are provided
        if self._sop is not None and self._genome is not None:
            engine = ExecutionEngine(
                llm_provider=self._llm_provider,
                model_fallback_cascade=self._model_fallback_cascade,
                tool_adapters=self._tool_adapters,
            )
            task_input = sample.to_task_data()["input"]

            try:
                trace = engine.execute(
                    sop=self._sop,
                    prompt_genome=self._genome,
                    task_input=task_input,
                )
                # Extract answer from trace final outputs
                # final_outputs structure: {role_id: {output_key: value, ...}}
                generated_answer = ""
                for role_outputs in trace.final_outputs.values():
                    if isinstance(role_outputs, dict):
                        # Look for common answer keys in role outputs
                        generated_answer = role_outputs.get(
                            "answer", role_outputs.get("response", "")
                        )
                        if generated_answer:
                            break
                # Fallback: try top-level (legacy format)
                if not generated_answer:
                    generated_answer = trace.final_outputs.get(
                        "answer", trace.final_outputs.get("response", "")
                    )
                error = None
            except Exception as e:
                logger.exception(f"Execution failed for sample {sample.id}")
                generated_answer = ""
                error = str(e)
        else:
            # No SOP/genome provided - cannot execute
            generated_answer = ""
            error = "No SOP or genome configured for benchmark runner"

        latency = (time.time() - start) * 1000

        return SampleResult(
            sample_id=sample.id,
            query=sample.query,
            generated_answer=generated_answer,
            ground_truth=sample.ground_truth,
            metrics={},
            latency_ms=latency,
            error=error,
        )

    def _aggregate_metrics(
        self,
        results: list[SampleResult],
    ) -> dict[str, float]:
        """Aggregate metrics across all samples."""
        if not results:
            return {}

        # Collect all metric values
        metric_values: dict[str, list[float]] = {}
        for result in results:
            for metric_id, value in result.metrics.items():
                if metric_id not in metric_values:
                    metric_values[metric_id] = []
                metric_values[metric_id].append(value)

        # Calculate means
        return {
            metric_id: sum(values) / len(values)
            for metric_id, values in metric_values.items()
        }

    def run_with_evaluation(
        self,
        dataset: "BenchmarkDataset",
        evaluation_service: "EvaluationService",
        metric_configs: list[Any],
        max_samples: int | None = None,
    ) -> BenchmarkResults:
        """Run benchmark with full EvaluationService integration.

        This method integrates with SIARE's evolution loop by:
        1. Running each sample through ExecutionEngine
        2. Evaluating with EvaluationService for each sample
        3. Storing execution traces and evaluation vectors

        Args:
            dataset: Benchmark dataset to evaluate
            evaluation_service: EvaluationService instance for metrics
            metric_configs: List of MetricConfig for evaluation
            max_samples: Maximum samples to run (overrides dataset limit)

        Returns:
            BenchmarkResults with evaluation vectors attached to each sample
        """
        from siare.services.execution_engine import ExecutionEngine

        start_time = time.time()
        sample_results: list[SampleResult] = []
        completed = 0
        failed = 0

        # Apply max_samples limit
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]

        for sample in samples:
            sample_start = time.time()

            # Execute sample
            if self._sop is None or self._genome is None:
                sample_results.append(
                    SampleResult(
                        sample_id=sample.id,
                        query=sample.query,
                        generated_answer="",
                        ground_truth=sample.ground_truth,
                        error="No SOP or genome configured",
                    )
                )
                failed += 1
                continue

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

                # Extract answer from trace
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

                # Run evaluation with EvaluationService
                task_data = sample.to_task_data()
                eval_vector = evaluation_service.evaluate(
                    trace=trace,
                    metrics=metric_configs,
                    task_data=task_data,
                )

                # Extract metrics as dict from EvaluationVector
                # MetricResult has metricId and score (or rawValue) attributes
                metrics_dict = {
                    m.metricId: m.score if m.score is not None else (m.rawValue or 0.0)
                    for m in eval_vector.metrics
                }

                latency = (time.time() - sample_start) * 1000

                sample_results.append(
                    SampleResult(
                        sample_id=sample.id,
                        query=sample.query,
                        generated_answer=generated_answer,
                        ground_truth=sample.ground_truth,
                        metrics=metrics_dict,
                        latency_ms=latency,
                        evaluation_vector=eval_vector,
                        execution_trace=trace,
                    )
                )
                completed += 1

            except Exception as e:
                logger.exception(f"Error running sample {sample.id} with evaluation")
                sample_results.append(
                    SampleResult(
                        sample_id=sample.id,
                        query=sample.query,
                        generated_answer="",
                        ground_truth=sample.ground_truth,
                        error=str(e),
                    )
                )
                failed += 1

        # Aggregate metrics
        aggregate = self._aggregate_metrics(sample_results)

        return BenchmarkResults(
            dataset_name=dataset.name,
            total_samples=len(samples),
            completed_samples=completed,
            failed_samples=failed,
            aggregate_metrics=aggregate,
            sample_results=sample_results,
            total_time_seconds=time.time() - start_time,
        )

    def _evaluate_single_sample(
        self,
        sample: "BenchmarkSample",
        evaluation_service: "EvaluationService",
        metric_configs: list[Any],
    ) -> SampleResult:
        """Evaluate a single sample with full error handling.

        Args:
            sample: The benchmark sample to evaluate
            evaluation_service: EvaluationService instance for metrics
            metric_configs: List of MetricConfig for evaluation

        Returns:
            SampleResult with evaluation results or error
        """
        from siare.services.execution_engine import ExecutionEngine

        sample_start = time.time()

        if self._sop is None or self._genome is None:
            return SampleResult(
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

            # Extract answer from trace
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

            # Run evaluation with EvaluationService
            task_data = sample.to_task_data()
            eval_vector = evaluation_service.evaluate(
                trace=trace,
                metrics=metric_configs,
                task_data=task_data,
            )

            # Extract metrics as dict from EvaluationVector
            metrics_dict = {
                m.metricId: m.score if m.score is not None else (m.rawValue or 0.0)
                for m in eval_vector.metrics
            }

            latency = (time.time() - sample_start) * 1000

            return SampleResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer=generated_answer,
                ground_truth=sample.ground_truth,
                metrics=metrics_dict,
                latency_ms=latency,
                evaluation_vector=eval_vector,
                execution_trace=trace,
            )

        except Exception as e:
            logger.exception(f"Error running sample {sample.id} with evaluation")
            return SampleResult(
                sample_id=sample.id,
                query=sample.query,
                generated_answer="",
                ground_truth=sample.ground_truth,
                error=str(e),
            )

    def run_with_evaluation_parallel(
        self,
        dataset: "BenchmarkDataset",
        evaluation_service: "EvaluationService",
        metric_configs: list[Any],
        max_samples: int | None = None,
        max_workers: int = 1,
    ) -> BenchmarkResults:
        """Run benchmark with parallel sample processing.

        This method uses ThreadPoolExecutor to evaluate samples concurrently,
        providing significant speedups for I/O-bound LLM evaluations.

        Args:
            dataset: Benchmark dataset to evaluate
            evaluation_service: EvaluationService instance for metrics
            metric_configs: List of MetricConfig for evaluation
            max_samples: Maximum samples to run (overrides dataset limit)
            max_workers: Number of parallel workers (default: 1 = sequential)

        Returns:
            BenchmarkResults with evaluation vectors attached to each sample
        """
        start_time = time.time()

        # Apply max_samples limit
        samples = list(dataset)
        if max_samples:
            samples = samples[:max_samples]

        # If max_workers is 1, fall back to sequential processing
        if max_workers <= 1:
            return self.run_with_evaluation(
                dataset=dataset,
                evaluation_service=evaluation_service,
                metric_configs=metric_configs,
                max_samples=max_samples,
            )

        sample_results: list[SampleResult] = []
        completed = 0
        failed = 0

        logger.info(f"Running {len(samples)} samples with {max_workers} parallel workers")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all samples for parallel processing
            future_to_sample = {
                executor.submit(
                    self._evaluate_single_sample,
                    sample,
                    evaluation_service,
                    metric_configs,
                ): sample
                for sample in samples
            }

            # Collect results as they complete
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    sample_results.append(result)
                    if result.error:
                        failed += 1
                    else:
                        completed += 1
                except Exception as e:
                    logger.exception(f"Unexpected error for sample {sample.id}")
                    sample_results.append(
                        SampleResult(
                            sample_id=sample.id,
                            query=sample.query,
                            generated_answer="",
                            ground_truth=sample.ground_truth,
                            error=str(e),
                        )
                    )
                    failed += 1

        # Sort results back to original order
        sample_id_order = {s.id: i for i, s in enumerate(samples)}
        sample_results.sort(key=lambda r: sample_id_order.get(r.sample_id, 999999))

        # Aggregate metrics
        aggregate = self._aggregate_metrics(sample_results)

        return BenchmarkResults(
            dataset_name=dataset.name,
            total_samples=len(samples),
            completed_samples=completed,
            failed_samples=failed,
            aggregate_metrics=aggregate,
            sample_results=sample_results,
            total_time_seconds=time.time() - start_time,
        )
