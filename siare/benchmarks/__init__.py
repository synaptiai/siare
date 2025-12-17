"""SIARE Benchmarks - Enterprise Benchmark Suite for RAG evaluation.

Standard Benchmarks:
    - FRAMES: 824 multi-hop questions from Google Research
    - BEIR: 15 datasets for zero-shot retrieval evaluation

Retrieval Metrics (BEIR-compatible):
    - NDCG@K, MAP@K, Recall@K, Precision@K, MRR
    - Standard k-values: [1, 3, 5, 10, 100, 1000]

RAG SOPs:
    - simple_qa: Minimal Q&A (no retrieval) for baseline testing
    - rag_retriever: Vector search retrieval + generation
    - multihop_rag: Multi-step retrieval for complex reasoning

RAG Benchmarking:
    - RAGBenchmarkRunner: Tracks retrieval + generation metrics separately
    - CorpusLoader: Loads BEIR corpora into vector stores
    - RetrievalEvaluator: Computes NDCG, Recall, MRR across queries

Example Usage:
    >>> from siare.benchmarks import FRAMESDataset, BenchmarkRunner
    >>> dataset = FRAMESDataset(max_samples=100)
    >>> runner = BenchmarkRunner()
    >>> results = runner.run(dataset)
    >>> print(results.summary())

RAG Benchmark Usage:
    >>> from siare.benchmarks import (
    ...     RAGBenchmarkRunner, create_rag_sop, create_rag_genome
    ... )
    >>> sop = create_rag_sop(model="llama3.2:1b", top_k=10)
    >>> genome = create_rag_genome(top_k=10)
    >>> runner = RAGBenchmarkRunner(sop=sop, genome=genome)
    >>> results = runner.run_with_retrieval_metrics(dataset, qrels)
"""

from siare.benchmarks.adapters import benchmark_to_taskset, taskset_to_benchmark_samples
from siare.benchmarks.base import BenchmarkDataset, BenchmarkSample
from siare.benchmarks.corpus import CorpusLoader
from siare.benchmarks.datasets import BEIRDataset, FRAMESDataset
from siare.benchmarks.evolution_runner import (
    EvolutionBenchmarkConfig,
    EvolutionBenchmarkResult,
    EvolutionBenchmarkRunner,
    GenerationHistoryEntry,
    StatisticalComparison,
)
from siare.benchmarks.self_improvement_benchmark import (
    GenerationSnapshot,
    SelfImprovementBenchmark,
    SelfImprovementConfig,
    SelfImprovementResult,
)
from siare.benchmarks.tracking import (
    ChangeSummary,
    ConvergenceInfo,
    LearningCurvePoint,
    LearningCurveTracker,
    PromptDiff,
    PromptDiffTracker,
    PromptSnapshot,
)
from siare.benchmarks.hotpotqa import HotpotQADataset
from siare.benchmarks.metrics import (
    STANDARD_K_VALUES,
    EvolutionMetrics,
    RetrievalEvaluator,
    RetrievalMetrics,
    average_precision,
    benchmark_accuracy,
    benchmark_f1,
    benchmark_partial_match,
    evaluate_retrieval,
    extract_retrieved_docs_from_trace,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    register_benchmark_metrics,
)
from siare.benchmarks.natural_questions import NaturalQuestionsDataset
from siare.benchmarks.publication_suite import (
    AblationResult,
    LearningCurveData,
    PowerAnalysisResult,
    PublicationBenchmark,
    PublicationBenchmarkResult,
    PublicationMetricStats,
    PublicationStatisticalTest,
)
from siare.benchmarks.quality_gate_suite import (
    BaselineComparison,
    MetricStatistics,
    QualityGateBenchmark,
    QualityGateResult,
    StatisticalTest,
)
from siare.benchmarks.rag_runner import (
    RAGBenchmarkResults,
    RAGBenchmarkRunner,
    RAGSampleResult,
)
from siare.benchmarks.reporter import BenchmarkReporter
from siare.benchmarks.reproducibility import EnvironmentSnapshot, ReproducibilityTracker
from siare.benchmarks.runner import BenchmarkResults, BenchmarkRunner, SampleResult
from siare.benchmarks.sops import (
    create_benchmark_genome,
    create_benchmark_sop,
    create_multihop_genome,
    create_multihop_sop,
    create_rag_genome,
    create_rag_sop,
)


__all__ = [
    # Constants
    "STANDARD_K_VALUES",
    # Datasets
    "BEIRDataset",
    "BenchmarkDataset",
    "BenchmarkSample",
    "CorpusLoader",
    "FRAMESDataset",
    "HotpotQADataset",
    "NaturalQuestionsDataset",
    # Runners
    "BenchmarkReporter",
    "BenchmarkResults",
    "BenchmarkRunner",
    "EvolutionBenchmarkConfig",
    "EvolutionBenchmarkResult",
    "EvolutionBenchmarkRunner",
    "GenerationHistoryEntry",
    "RAGBenchmarkResults",
    "RAGBenchmarkRunner",
    "RAGSampleResult",
    "SampleResult",
    "StatisticalComparison",
    # Self-improvement benchmark
    "GenerationSnapshot",
    "SelfImprovementBenchmark",
    "SelfImprovementConfig",
    "SelfImprovementResult",
    # Tracking
    "ChangeSummary",
    "ConvergenceInfo",
    "LearningCurvePoint",
    "LearningCurveTracker",
    "PromptDiff",
    "PromptDiffTracker",
    "PromptSnapshot",
    # Publication suite
    "AblationResult",
    "BaselineComparison",
    "EnvironmentSnapshot",
    "EvolutionMetrics",
    "LearningCurveData",
    "MetricStatistics",
    "PowerAnalysisResult",
    "PublicationBenchmark",
    "PublicationBenchmarkResult",
    "PublicationMetricStats",
    "PublicationStatisticalTest",
    "QualityGateBenchmark",
    "QualityGateResult",
    "ReproducibilityTracker",
    "RetrievalEvaluator",
    "RetrievalMetrics",
    "StatisticalTest",
    # Metrics
    "average_precision",
    "benchmark_accuracy",
    "benchmark_f1",
    "benchmark_partial_match",
    "evaluate_retrieval",
    "extract_retrieved_docs_from_trace",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "register_benchmark_metrics",
    # Adapters
    "benchmark_to_taskset",
    "taskset_to_benchmark_samples",
    # SOP factories
    "create_benchmark_genome",
    "create_benchmark_sop",
    "create_multihop_genome",
    "create_multihop_sop",
    "create_rag_genome",
    "create_rag_sop",
]
