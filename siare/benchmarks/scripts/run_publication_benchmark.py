#!/usr/bin/env python3
"""CLI script for running publication-grade benchmarks (Tier 2/Tier 3).

Tier 2 (QualityGate): Pre-production validation with Bonferroni correction
    python -m siare.benchmarks.scripts.run_publication_benchmark \
        --tier 2 \
        --dataset frames \
        --max-samples 50 \
        --n-runs 30 \
        --model llama3.1:8b

Tier 3 (Publication): Publication-grade with FDR correction, ablations, learning curves
    python -m siare.benchmarks.scripts.run_publication_benchmark \
        --tier 3 \
        --dataset frames \
        --max-samples 100 \
        --n-runs 30 \
        --confidence 0.99 \
        --model llama3.1:8b \
        --html --markdown
"""
import argparse
import logging
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from siare.benchmarks.base import BenchmarkDataset
    from siare.benchmarks.publication_suite import PublicationBenchmarkResult
    from siare.benchmarks.quality_gate_suite import QualityGateResult
    from siare.core.models import ProcessConfig, PromptGenome, SOPGene
    from siare.services.llm_provider import LLMProvider


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default confidence levels by tier
DEFAULT_CONFIDENCE_TIER_2 = 0.95
DEFAULT_CONFIDENCE_TIER_3 = 0.99

# HTTP status codes
HTTP_OK = 200

# Tier constants for comparison
TIER_QUALITY_GATE = 2
TIER_PUBLICATION = 3


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
        if response.status_code != HTTP_OK:
            return False

        data = response.json()
        models = [m.get("name", "").split(":")[0] for m in data.get("models", [])]
        model_base = model.split(":")[0]
        return model_base in models or model in [
            m.get("name", "") for m in data.get("models", [])
        ]
    except (httpx.RequestError, httpx.HTTPStatusError, KeyError, ValueError):
        logger.exception("Failed to connect to Ollama")
        return False


def load_dataset(name: str, max_samples: int | None = None) -> "BenchmarkDataset":
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

        dataset_name = name_lower.replace("beir_", "").replace("beir-", "")
        if dataset_name == "beir":
            dataset_name = "msmarco"
        return BEIRDataset(dataset_name=dataset_name, max_samples=max_samples)
    raise ValueError(f"Unknown dataset: {name}. Supported: frames, beir_<subset>")


def create_benchmark_configs(
    model: str,
) -> tuple[
    "ProcessConfig",
    "PromptGenome",
    dict[str, tuple["ProcessConfig", "PromptGenome"]],
    dict[str, tuple["ProcessConfig", "PromptGenome"]],
]:
    """Create SOP and genome configurations for benchmarking.

    Args:
        model: LLM model name

    Returns:
        Tuple of (evolved_sop, evolved_genome, baselines_dict, ablations_dict)
    """
    from siare.benchmarks.sops.multihop_rag import (
        create_multihop_genome,
        create_multihop_sop,
    )
    from siare.benchmarks.sops.simple_qa import (
        create_benchmark_genome as create_simple_genome,
    )
    from siare.benchmarks.sops.simple_qa import (
        create_benchmark_sop as create_simple_sop,
    )

    # Main evolved SOP (multihop RAG)
    evolved_sop = create_multihop_sop(model=model)
    evolved_genome = create_multihop_genome()

    # Baselines for comparison
    baselines = {
        "simple_qa": (create_simple_sop(model), create_simple_genome()),
    }

    # Ablations (for Tier 3 only)
    # These are variants with specific components disabled
    ablations = {
        # Example: Could add variants with different tools disabled, etc.
        # For now, we use the simple_qa as an ablation of the retrieval component
        "no_retrieval": (create_simple_sop(model), create_simple_genome()),
    }

    return evolved_sop, evolved_genome, baselines, ablations


def run_tier2_benchmark(
    dataset: "BenchmarkDataset",
    provider: "LLMProvider",
    tool_adapters: dict[str, Callable[..., Any]],
    evolved_sop: "ProcessConfig",
    evolved_genome: "PromptGenome",
    baselines: dict[str, tuple["ProcessConfig", "PromptGenome"]],
    n_runs: int,
    confidence_level: float,
    max_samples: int | None,
    random_seed: int,
) -> "QualityGateResult":
    """Run Tier 2 quality gate benchmark.

    Args:
        dataset: Benchmark dataset
        provider: LLM provider
        tool_adapters: Tool adapters for execution
        evolved_sop: Evolved SOP configuration
        evolved_genome: Evolved prompt genome
        baselines: Dictionary of baseline configs
        n_runs: Number of runs per sample
        confidence_level: Statistical confidence level
        max_samples: Maximum samples to use
        random_seed: Random seed for reproducibility

    Returns:
        QualityGateResult
    """
    from siare.benchmarks.quality_gate_suite import QualityGateBenchmark

    logger.info("Initializing Tier 2 (QualityGate) benchmark...")
    benchmark = QualityGateBenchmark(
        dataset=dataset,
        llm_provider=provider,
        metrics=None,  # Use defaults
        tool_adapters=tool_adapters,
    )

    logger.info(f"Running benchmark with n_runs={n_runs}, confidence={confidence_level}")
    return benchmark.run(
        sop=evolved_sop,
        genome=evolved_genome,
        baselines=baselines,
        n_runs=n_runs,
        confidence_level=confidence_level,
        max_samples=max_samples,
        random_seed=random_seed,
    )


def run_tier3_benchmark(
    dataset: "BenchmarkDataset",
    provider: "LLMProvider",
    tool_adapters: dict[str, Callable[..., Any]],
    evolved_sop: "ProcessConfig",
    evolved_genome: "PromptGenome",
    baselines: dict[str, tuple["ProcessConfig", "PromptGenome"]],
    ablations: dict[str, tuple["ProcessConfig", "PromptGenome"]],
    n_runs: int,
    confidence_level: float,
    max_samples: int | None,
    random_seed: int,
    evolution_history: list["SOPGene"],
) -> "PublicationBenchmarkResult":
    """Run Tier 3 publication benchmark.

    Args:
        dataset: Benchmark dataset (golden dataset)
        provider: LLM provider
        tool_adapters: Tool adapters for execution
        evolved_sop: Evolved SOP configuration
        evolved_genome: Evolved prompt genome
        baselines: Dictionary of baseline configs
        ablations: Dictionary of ablation configs
        n_runs: Number of runs per sample
        confidence_level: Statistical confidence level
        max_samples: Maximum samples to use
        random_seed: Random seed for reproducibility
        evolution_history: List of SOPGene from evolution history

    Returns:
        PublicationBenchmarkResult
    """
    from siare.benchmarks.publication_suite import PublicationBenchmark

    logger.info("Initializing Tier 3 (Publication) benchmark...")
    benchmark = PublicationBenchmark(
        golden_dataset=dataset,
        llm_provider=provider,
        public_dataset=None,  # Could be different dataset for public comparison
        metrics=None,  # Use defaults
        tool_adapters=tool_adapters,
    )

    logger.info(
        f"Running full publication suite with n_runs={n_runs}, confidence={confidence_level}"
    )
    return benchmark.run_full_suite(
        evolved_sop=evolved_sop,
        evolved_genome=evolved_genome,
        evolution_history=evolution_history,
        baselines=baselines,
        ablations=ablations,
        n_runs=n_runs,
        confidence_level=confidence_level,
        max_samples=max_samples,
        random_seed=random_seed,
    )


def save_tier2_results(
    result: "QualityGateResult",
    output_dir: Path,
    timestamp: str,
    html: bool,
    markdown: bool,
) -> dict[str, Path]:
    """Save Tier 2 benchmark results.

    Args:
        result: QualityGateResult
        output_dir: Output directory
        timestamp: Timestamp string
        html: Generate HTML report
        markdown: Generate markdown report

    Returns:
        Dictionary of output file paths
    """
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"quality_gate_{timestamp}"
    paths = {}

    # Save JSON
    json_path = output_dir / f"{prefix}.json"
    with json_path.open("w") as f:
        json.dump(result.summary(), f, indent=2, default=str)
    paths["json"] = json_path

    # Summary report
    summary = result.summary()
    gate_status = "PASSED" if result.passes_gate() else "FAILED"

    if markdown:
        md_path = output_dir / f"{prefix}.md"
        lines = [
            "# Quality Gate Benchmark Report",
            "",
            f"**Status:** {gate_status}",
            f"**Date:** {datetime.now(timezone.utc).isoformat()}",
            f"**Samples:** {summary['n_queries']}",
            f"**Runs:** {summary['n_runs']}",
            f"**Confidence Level:** {summary['confidence_level']:.0%}",
            "",
            "## Evolved SOP Metrics",
            "",
            "| Metric | Mean | 95% CI | Std |",
            "|--------|------|--------|-----|",
        ]

        for metric_name, metric_data in summary["sop_metrics"].items():
            lines.append(
                f"| {metric_name} | {metric_data['mean']:.4f} | "
                f"[{metric_data['ci_lower']:.4f}, {metric_data['ci_upper']:.4f}] | "
                f"{metric_data['std']:.4f} |"
            )

        lines.extend(["", "## Baseline Comparisons", ""])

        for baseline_name, baseline_data in summary["baseline_comparisons"].items():
            lines.append(f"### vs. {baseline_name}")
            lines.append("")
            lines.append("| Metric | p-value | Significant | Effect Size |")
            lines.append("|--------|---------|-------------|-------------|")

            for metric_name, test_data in baseline_data["statistical_tests"].items():
                sig = "Yes" if test_data["significant"] else "No"
                lines.append(
                    f"| {metric_name} | {test_data['p_value']:.4f} | {sig} | "
                    f"{test_data['effect_size']:.3f} |"
                )
            lines.append("")

        with md_path.open("w") as f:
            f.write("\n".join(lines))
        paths["markdown"] = md_path

    if html:
        html_path = output_dir / f"{prefix}.html"
        # Simple HTML report for Tier 2
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Quality Gate Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 1000px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: {'#27ae60' if gate_status == 'PASSED' else '#e74c3c'}; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background: #1f77b4; color: white; }}
    </style>
</head>
<body>
    <h1>Quality Gate: {gate_status}</h1>
    <p>Generated: {datetime.now(timezone.utc).isoformat()}</p>
    <p>Samples: {summary['n_queries']} | Runs: {summary['n_runs']} | Confidence: {summary['confidence_level']:.0%}</p>

    <h2>Results</h2>
    <table>
        <tr><th>Metric</th><th>Mean</th><th>95% CI</th></tr>
"""
        for metric_name, metric_data in summary["sop_metrics"].items():
            html_content += f"""        <tr>
            <td>{metric_name}</td>
            <td>{metric_data['mean']:.4f}</td>
            <td>[{metric_data['ci_lower']:.4f}, {metric_data['ci_upper']:.4f}]</td>
        </tr>
"""
        html_content += """    </table>
</body>
</html>"""

        with html_path.open("w") as f:
            f.write(html_content)
        paths["html"] = html_path

    return paths


def save_tier3_results(
    result: "PublicationBenchmarkResult",
    output_dir: Path,
    timestamp: str,
    html: bool,
    markdown: bool,
) -> dict[str, Path]:
    """Save Tier 3 publication benchmark results.

    Args:
        result: PublicationBenchmarkResult
        output_dir: Output directory
        timestamp: Timestamp string
        html: Generate HTML report
        markdown: Generate markdown report

    Returns:
        Dictionary of output file paths
    """
    from siare.benchmarks.reporter import BenchmarkReporter

    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"publication_{timestamp}"
    paths = {}

    # Save JSON
    json_path = output_dir / f"{prefix}.json"
    with json_path.open("w") as f:
        import json

        json.dump(result.to_dict(), f, indent=2, default=str)
    paths["json"] = json_path

    # Use reporter for formatted outputs
    reporter = BenchmarkReporter()

    if markdown:
        md_path = output_dir / f"{prefix}.md"
        reporter.save_publication_report(result, str(md_path), "markdown")
        paths["markdown"] = md_path

    if html:
        html_path = output_dir / f"{prefix}.html"
        reporter.save_publication_report(result, str(html_path), "html")
        paths["html"] = html_path

    return paths


def main():
    parser = argparse.ArgumentParser(
        description="Run publication-grade benchmarks (Tier 2 or Tier 3)"
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=[2, 3],
        required=True,
        help="Benchmark tier: 2 (QualityGate) or 3 (Publication)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="frames",
        help="Dataset name: frames, beir_msmarco, etc.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to use from dataset",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=30,
        help="Number of runs per sample (default: 30)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Confidence level (default: 0.95 for Tier 2, 0.99 for Tier 3)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="LLM model to use (Ollama model name)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="Generate markdown report",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip Ollama health check",
    )
    parser.add_argument(
        "--skip-ablations",
        action="store_true",
        help="Skip ablation studies (Tier 3 only)",
    )

    args = parser.parse_args()

    # Set default confidence level based on tier
    if args.confidence is None:
        args.confidence = (
            DEFAULT_CONFIDENCE_TIER_2
            if args.tier == TIER_QUALITY_GATE
            else DEFAULT_CONFIDENCE_TIER_3
        )

    # Check Ollama health
    if not args.skip_health_check:
        logger.info(f"Checking Ollama health for model {args.model}...")
        if not check_ollama_health(args.model):
            logger.error(
                f"Ollama health check failed. Ensure Ollama is running and model {args.model} is available."
            )
            logger.error(f"Run: ollama pull {args.model}")
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

    # Create benchmark configs
    logger.info("Creating benchmark configurations...")
    evolved_sop, evolved_genome, baselines, ablations = create_benchmark_configs(args.model)

    # Initialize tool adapters
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

    # Run appropriate benchmark tier
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    if args.tier == TIER_QUALITY_GATE:
        logger.info("=" * 60)
        logger.info("RUNNING TIER 2 (QUALITY GATE) BENCHMARK")
        logger.info("=" * 60)

        result = run_tier2_benchmark(
            dataset=dataset,
            provider=provider,
            tool_adapters=tool_adapters,
            evolved_sop=evolved_sop,
            evolved_genome=evolved_genome,
            baselines=baselines,
            n_runs=args.n_runs,
            confidence_level=args.confidence,
            max_samples=args.max_samples,
            random_seed=args.random_seed,
        )

        # Print summary
        summary = result.summary()
        gate_status = "PASSED" if result.passes_gate() else "FAILED"

        print("\n" + "=" * 60)
        print(f"QUALITY GATE RESULT: {gate_status}")
        print("=" * 60)
        print(f"Dataset Hash: {summary['dataset_hash'][:16]}...")
        print(f"Samples: {summary['n_queries']}")
        print(f"Runs: {summary['n_runs']}")
        print(f"Confidence: {summary['confidence_level']:.0%}")
        print()
        print("Evolved SOP Metrics:")
        for metric, data in summary["sop_metrics"].items():
            print(f"  - {metric}: {data['mean']:.4f} (CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}])")
        print("=" * 60)

        # Save results
        output_dir = (
            Path(args.output_dir) if args.output_dir else Path("benchmarks/results/quality_gate")
        )
        paths = save_tier2_results(result, output_dir, timestamp, args.html, args.markdown)

    else:  # Tier 3
        logger.info("=" * 60)
        logger.info("RUNNING TIER 3 (PUBLICATION) BENCHMARK")
        logger.info("=" * 60)

        # For Tier 3, we need evolution history for learning curves
        # In a real scenario, this would come from an actual evolution run
        # For now, we create a minimal placeholder
        evolution_history: list[SOPGene] = []  # Would be populated from actual evolution

        # Skip ablations if requested
        ablations_to_use = {} if args.skip_ablations else ablations

        result = run_tier3_benchmark(
            dataset=dataset,
            provider=provider,
            tool_adapters=tool_adapters,
            evolved_sop=evolved_sop,
            evolved_genome=evolved_genome,
            baselines=baselines,
            ablations=ablations_to_use,
            n_runs=args.n_runs,
            confidence_level=args.confidence,
            max_samples=args.max_samples,
            random_seed=args.random_seed,
            evolution_history=evolution_history,
        )

        # Print summary
        summary = result.to_dict()
        evolved_metrics = summary.get("evolved_sop_results", {}).get("metrics", {})

        print("\n" + "=" * 60)
        print("PUBLICATION BENCHMARK RESULTS")
        print("=" * 60)
        print(f"Dataset: {summary['metadata'].get('dataset_name', 'Unknown')}")
        print(f"Samples: {summary['metadata'].get('n_queries', 0)}")
        print(f"Runs: {summary['metadata'].get('n_runs', 0)}")
        print(f"Confidence: {summary['metadata'].get('confidence_level', 0):.0%}")
        print()
        print("Evolved SOP Metrics:")
        for metric, data in evolved_metrics.items():
            print(
                f"  - {metric}: {data['mean']:.4f} "
                f"(CI: [{data['ci_lower']:.4f}, {data['ci_upper']:.4f}])"
            )
        print()

        # Power analysis
        if result.power_analysis:
            pa = result.power_analysis
            print("Power Analysis:")
            print(f"  - Effect Size: {pa.effect_size:.3f}")
            print(f"  - Achieved Power: {pa.power:.3f}")
            print(f"  - Sufficient: {'Yes' if pa.sufficient else 'No'}")
        print("=" * 60)

        # Save results
        output_dir = (
            Path(args.output_dir) if args.output_dir else Path("benchmarks/results/publication")
        )
        paths = save_tier3_results(result, output_dir, timestamp, args.html, args.markdown)

    print("\nResults saved to:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")


if __name__ == "__main__":
    main()
