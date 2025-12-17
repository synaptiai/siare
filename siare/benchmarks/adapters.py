"""Adapters for converting benchmark datasets to evolution formats."""
from datetime import datetime, timezone

from siare.benchmarks.base import BenchmarkDataset
from siare.core.models import TaskSet


def benchmark_to_taskset(
    dataset: BenchmarkDataset,
    version: str = "1.0.0",
    domain: str | None = None,
    description: str | None = None,
) -> TaskSet:
    """Convert a BenchmarkDataset to a TaskSet for evolution.

    Args:
        dataset: Benchmark dataset to convert
        version: TaskSet version string (default "1.0.0")
        domain: Domain name override (defaults to dataset name)
        description: TaskSet description override

    Returns:
        TaskSet containing all benchmark samples as Tasks
    """
    # Ensure dataset is loaded
    if not dataset._loaded:
        dataset.load()

    tasks = [sample.to_task() for sample in dataset]

    return TaskSet(
        id=f"benchmark_{dataset.name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
        domain=domain or dataset.name,
        description=description or f"Benchmark TaskSet from {dataset.name} dataset",
        tasks=tasks,
        version=version,
    )


def taskset_to_benchmark_samples(task_set: TaskSet) -> list[dict]:
    """Convert a TaskSet back to benchmark sample format.

    Args:
        task_set: TaskSet to convert

    Returns:
        List of dictionaries in benchmark sample format
    """
    samples = []
    for task in task_set.tasks:
        sample = {
            "id": task.id,
            "query": task.input.get("query", ""),
            "ground_truth": task.groundTruth.get("answer", "") if task.groundTruth else "",
            "context": task.groundTruth.get("context", []) if task.groundTruth else [],
            "metadata": {},
        }
        if task.metadata:
            sample["metadata"] = {
                "domain": task.metadata.domain,
                "difficulty": task.metadata.difficulty,
                "tags": task.metadata.tags,
            }
        samples.append(sample)
    return samples
