#!/usr/bin/env python3
"""CLI script for running evolution-aware benchmarks.

Usage:
    python -m siare.benchmarks.scripts.run_evolution_benchmark \
        --dataset frames \
        --max-samples 50 \
        --max-generations 10 \
        --model llama3.2:1b

For quick validation:
    python -m siare.benchmarks.scripts.run_evolution_benchmark \
        --dataset frames \
        --max-samples 10 \
        --quick
"""
import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_ollama_health(model: str) -> bool:
    """Check if Ollama is running and has the specified model.

    Args:
        model: Model name to check

    Returns:
        True if Ollama is healthy and model is available
    """
    import httpx

    try:
        response = httpx.get("http://localhost:11434/api/tags", timeout=5.0)
        if response.status_code != 200:
            return False

        data = response.json()
        models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        model_base = model.split(":")[0]
        return model_base in models or model in [
            m.get("name", "") for m in data.get("models", [])
        ]
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        return False


def load_dataset(name: str, max_samples: int | None = None):
    """Load benchmark dataset by name.

    Args:
        name: Dataset name (frames, beir, hotpotqa)
        max_samples: Maximum samples to load

    Returns:
        BenchmarkDataset instance
    """
    name_lower = name.lower()

    if name_lower == "frames":
        from siare.benchmarks.datasets.frames import FRAMESDataset

        return FRAMESDataset(max_samples=max_samples)
    if name_lower.startswith("beir"):
        from siare.benchmarks.datasets.beir import BEIRDataset

        subset = name_lower.replace("beir_", "").replace("beir-", "")
        if subset == "beir":
            subset = "msmarco"
        return BEIRDataset(subset=subset, max_samples=max_samples)
    raise ValueError(f"Unknown dataset: {name}. Supported: frames, beir_<subset>")


def create_benchmark_sop(model: str, use_retrieval: bool = True):
    """Create a benchmark SOP for evolution.

    Args:
        model: LLM model to use
        use_retrieval: If True, use RAG SOP with web_search tool (default: True)

    Returns:
        Tuple of (ProcessConfig, PromptGenome)
    """
    if use_retrieval:
        # Use multihop RAG SOP with web_search tool (WikipediaSearchAdapter)
        # This allows evolution to improve retrieval-augmented generation
        from siare.benchmarks.sops.multihop_rag import (
            create_multihop_genome,
            create_multihop_sop,
        )

        return create_multihop_sop(model=model), create_multihop_genome()
    else:
        # Fallback to simple QA without retrieval (baseline comparison)
        from siare.benchmarks.sops.simple_qa import (
            create_benchmark_genome,
            create_benchmark_sop,
        )

        return create_benchmark_sop(model), create_benchmark_genome()


def save_results(
    result,
    output_dir: Path,
    timestamp: str,
) -> dict[str, Path]:
    """Save evolution benchmark results to files.

    Args:
        result: EvolutionBenchmarkResult
        output_dir: Output directory
        timestamp: Timestamp string for filenames

    Returns:
        Dictionary of output file paths
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"evolution_{timestamp}"

    paths = {}

    # Save JSON results
    json_path = output_dir / f"{prefix}.json"
    with json_path.open("w") as f:
        json.dump(result.summary(), f, indent=2, default=str)
    paths["json"] = json_path

    # Save markdown report
    md_path = output_dir / f"{prefix}.md"
    with md_path.open("w") as f:
        f.write(generate_markdown_report(result))
    paths["markdown"] = md_path

    return paths


def generate_markdown_report(result) -> str:
    """Generate markdown report from evolution benchmark result.

    Args:
        result: EvolutionBenchmarkResult

    Returns:
        Markdown string
    """
    lines = [
        "# Evolution Benchmark Report",
        "",
        f"**Dataset:** {result.dataset_name}",
        f"**Date:** {datetime.now(timezone.utc).isoformat()}",
        f"**Total Time:** {result.total_time_seconds:.2f}s",
        f"**Generations Run:** {result.generations_run}",
        "",
        "## Summary",
        "",
        "| Phase | Samples | Completed | Failed |",
        "|-------|---------|-----------|--------|",
        f"| Baseline | {result.baseline_results.total_samples} | {result.baseline_results.completed_samples} | {result.baseline_results.failed_samples} |",
        f"| Evolved | {result.evolved_results.total_samples} | {result.evolved_results.completed_samples} | {result.evolved_results.failed_samples} |",
        "",
        "## Improvement Analysis",
        "",
        "| Metric | Baseline | Evolved | Improvement | % Change |",
        "|--------|----------|---------|-------------|----------|",
    ]

    for comp in result.comparisons:
        sign = "+" if comp.improvement >= 0 else ""
        lines.append(
            f"| {comp.metric_name} | {comp.baseline_mean:.4f} | {comp.evolved_mean:.4f} | {sign}{comp.improvement:.4f} | {sign}{comp.improvement_pct:.1f}% |"
        )

    lines.extend([
        "",
        "## Statistical Significance",
        "",
        "| Metric | Baseline CI (95%) | Evolved CI (95%) | p-value |",
        "|--------|-------------------|------------------|---------|",
    ])

    for comp in result.comparisons:
        p_str = f"{comp.p_value:.4f}" if comp.p_value is not None else "N/A"
        sig = "**" if comp.p_value is not None and comp.p_value < 0.05 else ""
        lines.append(
            f"| {comp.metric_name} | [{comp.baseline_ci[0]:.4f}, {comp.baseline_ci[1]:.4f}] | [{comp.evolved_ci[0]:.4f}, {comp.evolved_ci[1]:.4f}] | {sig}{p_str}{sig} |"
        )

    lines.extend([
        "",
        "## Configuration",
        "",
        f"- **Model:** {result.config.model}",
        f"- **Max Generations:** {result.config.max_generations}",
        f"- **Quick Mode:** {result.config.quick_mode}",
        f"- **Metrics:** {', '.join(result.config.metrics_to_optimize)}",
        "",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Run evolution-aware benchmarks on RAG datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="frames",
        help="Dataset name: frames, beir_msmarco, beir_nfcorpus, etc.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use from dataset",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        default=5,
        help="Maximum evolution generations",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=3,
        help="Population size per generation",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:1b",
        help="LLM model to use (Ollama model name)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (2-3 generations)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip Ollama health check",
    )

    args = parser.parse_args()

    # Check Ollama health
    if not args.skip_health_check:
        logger.info(f"Checking Ollama health for model {args.model}...")
        if not check_ollama_health(args.model):
            logger.error(
                f"Ollama health check failed. Ensure Ollama is running and model {args.model} is available."
            )
            logger.error("Run: ollama pull llama3.2:1b")
            sys.exit(1)
        logger.info("Ollama health check passed")

    # Create LLM provider
    from siare.providers.ollama_provider import OllamaProvider

    provider = OllamaProvider(
        model=args.model,
        timeout=120.0,
    )

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, args.max_samples)
    logger.info(f"Dataset loaded: {len(dataset)} samples")

    # Create base SOP and genome (with retrieval capability)
    logger.info("Creating baseline SOP with retrieval capability")
    base_sop, base_genome = create_benchmark_sop(args.model, use_retrieval=True)

    # Initialize tool adapters for web_search (Wikipedia)
    # This enables the RAG pipeline to retrieve external knowledge
    from siare.adapters.wikipedia_search import WikipediaSearchAdapter

    logger.info("Initializing WikipediaSearchAdapter for web_search tool")
    wiki_adapter = WikipediaSearchAdapter(
        config={
            "max_results": 5,
            "timeout": 30,
            "extract_chars": 1500,
        }
    )
    wiki_adapter.initialize()
    tool_adapters = {"web_search": wiki_adapter.execute}

    # Create config
    from siare.benchmarks.evolution_runner import (
        EvolutionBenchmarkConfig,
        EvolutionBenchmarkRunner,
    )

    config = EvolutionBenchmarkConfig(
        max_generations=args.max_generations,
        population_size=args.population_size,
        model=args.model,
        quick_mode=args.quick,
        max_samples=args.max_samples,
    )

    # Create runner with tool adapters
    runner = EvolutionBenchmarkRunner(
        config=config,
        llm_provider=provider,
        base_sop=base_sop,
        base_genome=base_genome,
        tool_adapters=tool_adapters,
    )

    # Run benchmark
    logger.info("Starting evolution benchmark...")
    if args.quick:
        result = runner.run_quick_validation(dataset)
    else:
        result = runner.run(dataset)

    # Print summary
    summary = result.summary()
    print("\n" + "=" * 60)
    print("EVOLUTION BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Dataset: {summary['dataset']}")
    print(f"Generations: {summary['generations']}")
    print(f"Total Time: {summary['total_time_seconds']:.2f}s")
    print()
    print("Baseline Results:")
    print(f"  - Completed: {summary['baseline']['completed']}")
    print(f"  - Metrics: {summary['baseline']['metrics']}")
    print()
    print("Evolved Results:")
    print(f"  - Completed: {summary['evolved']['completed']}")
    print(f"  - Metrics: {summary['evolved']['metrics']}")
    print()
    print("Improvements:")
    for metric, data in summary["improvements"].items():
        print(f"  - {metric}: {data['baseline']:.4f} -> {data['evolved']:.4f} ({data['improvement_pct']})")
    print("=" * 60)

    # Save results
    output_dir = Path(args.output_dir) if args.output_dir else Path("benchmarks/results")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    paths = save_results(result, output_dir, timestamp)

    print("\nResults saved to:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
