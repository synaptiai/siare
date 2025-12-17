"""Run RAG benchmarks with retrieval metrics.

Usage:
    # Run BEIR benchmark with vector search
    python -m siare.benchmarks.scripts.run_rag_benchmark \
        --dataset beir-nfcorpus \
        --sop rag \
        --model llama3.2:1b \
        --html

    # Run FRAMES with multi-hop RAG and HTML report
    python -m siare.benchmarks.scripts.run_rag_benchmark \
        --dataset frames \
        --sop multihop \
        --max-samples 50 \
        --html --markdown

Outputs:
    - JSON results file with retrieval and generation metrics
    - HTML report with interactive Plotly charts (if --html)
    - Markdown report (if --markdown)
"""

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path


logger = logging.getLogger(__name__)


def load_qrels(dataset_name: str, data_dir: Path) -> dict[str, dict[str, int]]:
    """Load relevance judgments for a dataset.

    Args:
        dataset_name: Name of the dataset
        data_dir: Directory containing qrels files

    Returns:
        Dict of {query_id: {doc_id: relevance}}
    """
    # For BEIR datasets
    if dataset_name.startswith("beir-"):
        beir_name = dataset_name[5:]  # Remove "beir-" prefix
        qrels_path = data_dir / f"{beir_name}_qrels.json"
        if qrels_path.exists():
            with qrels_path.open() as f:
                return json.load(f)

    # FRAMES doesn't have doc-level qrels
    return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG benchmarks with retrieval metrics"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset: 'beir-<name>', 'frames', 'hotpotqa'",
    )
    parser.add_argument(
        "--sop",
        type=str,
        choices=["rag", "multihop", "simple"],
        default="rag",
        help="SOP type to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:1b",
        help="Model identifier",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to run",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/beir",
        help="Directory for benchmark data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report with interactive charts",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Generate markdown report",
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")

    if args.dataset.startswith("beir-"):
        from siare.benchmarks.datasets.beir import BEIRDataset

        dataset_name = args.dataset[5:]
        dataset = BEIRDataset(
            dataset_name=dataset_name,
            max_samples=args.max_samples,
        )
    elif args.dataset == "frames":
        from siare.benchmarks.datasets.frames import FRAMESDataset

        dataset = FRAMESDataset(max_samples=args.max_samples)
    elif args.dataset == "hotpotqa":
        from siare.benchmarks.hotpotqa import HotpotQADataset

        dataset = HotpotQADataset(max_samples=args.max_samples)
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return 1

    # Create SOP
    logger.info(f"Creating {args.sop} SOP with model {args.model}")

    if args.sop == "rag":
        from siare.benchmarks.sops.rag_retriever import (
            create_rag_genome,
            create_rag_sop,
        )

        sop = create_rag_sop(model=args.model, top_k=args.top_k)
        genome = create_rag_genome(top_k=args.top_k)
    elif args.sop == "multihop":
        from siare.benchmarks.sops.multihop_rag import (
            create_multihop_genome,
            create_multihop_sop,
        )

        sop = create_multihop_sop(model=args.model)
        genome = create_multihop_genome()
    else:
        from siare.benchmarks.sops.simple_qa import (
            create_benchmark_genome,
            create_benchmark_sop,
        )

        sop = create_benchmark_sop(model=args.model)
        genome = create_benchmark_genome()

    # Load qrels if available
    qrels = load_qrels(args.dataset, data_dir)
    if qrels:
        logger.info(f"Loaded qrels for {len(qrels)} queries")
    else:
        logger.info("No qrels available, skipping retrieval metrics")

    # Create LLM provider for Ollama
    from siare.providers.ollama_provider import OllamaProvider

    llm_provider = OllamaProvider(
        model=args.model,
        timeout=120.0,
    )
    logger.info(f"Initialized Ollama provider for model {args.model}")

    # Create tool adapters for RAG
    tool_adapters = {}

    # Add web search adapter for multihop RAG
    if args.sop == "multihop":
        # Use Wikipedia API instead of DuckDuckGo (more reliable, no rate limits)
        from siare.adapters.wikipedia_search import WikipediaSearchAdapter

        wiki_adapter = WikipediaSearchAdapter(
            config={
                "max_results": 5,
                "timeout": 30,
                "extract_chars": 1500,  # More context per article
            }
        )
        wiki_adapter.initialize()
        tool_adapters["web_search"] = wiki_adapter.execute  # Register as web_search
        logger.info("Initialized web_search adapter with Wikipedia provider")

    # Create runner
    from siare.benchmarks.rag_runner import RAGBenchmarkRunner

    runner = RAGBenchmarkRunner(
        sop=sop,
        genome=genome,
        llm_provider=llm_provider,
        model_fallback_cascade=[args.model],
        tool_adapters=tool_adapters,
    )

    # Run benchmark
    dataset_samples = list(dataset)
    logger.info(f"Running benchmark on {len(dataset_samples)} samples...")

    # Always use run_with_retrieval_metrics to get generation metrics
    # Retrieval metrics will only be computed if qrels are available
    results = runner.run_with_retrieval_metrics(
        dataset,
        qrels=qrels if qrels else None,
        max_samples=args.max_samples,
    )

    # Save results
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
    result_file = output_dir / f"{args.dataset}_{args.sop}_{timestamp}.json"

    with result_file.open("w") as f:
        json.dump(results.summary(), f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"RAG Benchmark Results: {args.dataset}")
    print("=" * 60)
    print(f"SOP: {args.sop}")
    print(f"Model: {args.model}")
    print(f"Samples: {results.total_samples}")
    print(f"Completed: {results.completed_samples}")
    print(f"Failed: {results.failed_samples}")
    print(f"Runtime: {results.total_time_seconds:.2f}s")

    if hasattr(results, "retrieval_metrics") and results.retrieval_metrics:
        print("\nRetrieval Metrics:")
        for metric, value in sorted(results.retrieval_metrics.items()):
            print(f"  {metric}: {value:.4f}")

    if hasattr(results, "generation_metrics") and results.generation_metrics:
        print("\nGeneration Metrics:")
        for metric, value in sorted(results.generation_metrics.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    print(f"\nResults saved to: {result_file}")

    # Generate additional reports if requested
    if args.html or args.markdown:
        from siare.benchmarks.reporter import BenchmarkReporter

        reporter = BenchmarkReporter()
        base_name = f"{args.dataset}_{args.sop}_{timestamp}"

        if args.html:
            html_file = output_dir / f"{base_name}.html"
            reporter.save_report(results, str(html_file), "html")
            print(f"HTML report saved to: {html_file}")

        if args.markdown:
            md_file = output_dir / f"{base_name}.md"
            reporter.save_report(results, str(md_file), "markdown")
            print(f"Markdown report saved to: {md_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
