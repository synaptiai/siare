#!/usr/bin/env python
"""Verify benchmark suite with Ollama.

Runs a small FRAMES benchmark to verify the full pipeline works:
- Dataset loading
- LLM execution via Ollama
- Metrics collection
- Report generation

Usage:
    python -m siare.benchmarks.scripts.verify_benchmark
    python -m siare.benchmarks.scripts.verify_benchmark --samples 10 --model llama3.2:1b
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

from siare.benchmarks.datasets.frames import FRAMESDataset
from siare.benchmarks.reporter import BenchmarkReporter
from siare.benchmarks.runner import BenchmarkRunner
from siare.benchmarks.sops import create_benchmark_genome, create_benchmark_sop
from siare.providers.ollama_provider import OllamaProvider


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_ollama_health(provider: OllamaProvider, model: str) -> bool:
    """Check if Ollama is running and model is available."""
    try:
        if not provider.health_check():
            logger.error("Ollama server not responding")
            return False

        if not provider.is_model_available(model):
            logger.error(f"Model {model} not available")
            logger.info("Available models:")
            for m in provider.list_models():
                logger.info(f"  - {m.get('name', 'unknown')}")
            return False

        return True
    except Exception as e:
        logger.error(f"Ollama health check failed: {e}")
        return False


def run_verification(
    samples: int = 50,
    model: str = "llama3.2:1b",
    output_dir: str = "benchmarks/results",
) -> int:
    """Run benchmark verification.

    Args:
        samples: Number of FRAMES samples to run
        model: Ollama model to use
        output_dir: Directory for results

    Returns:
        0 on success, 1 on failure
    """
    print("=" * 60)
    print("SIARE Benchmark Verification")
    print("=" * 60)

    # Step 1: Setup Ollama provider
    print(f"\n[1/5] Setting up Ollama provider with model: {model}")
    provider = OllamaProvider(model=model, timeout=120.0)

    if not check_ollama_health(provider, model):
        return 1

    print(f"  ✓ Ollama connected, model {model} ready")

    # Step 2: Create SOP and genome
    print("\n[2/5] Creating benchmark SOP and genome")
    sop = create_benchmark_sop(model=model)
    genome = create_benchmark_genome()
    print(f"  ✓ SOP: {sop.id} v{sop.version}")
    print(f"  ✓ Genome: {genome.id} v{genome.version}")

    # Step 3: Load FRAMES dataset
    print(f"\n[3/5] Loading FRAMES dataset ({samples} samples)")
    dataset = FRAMESDataset(max_samples=samples)
    dataset.load()
    actual_samples = len(dataset)
    print(f"  ✓ Loaded {actual_samples} samples")

    if actual_samples == 0:
        logger.error("No samples loaded. Check if 'datasets' library is installed.")
        return 1

    # Step 4: Run benchmark
    print("\n[4/5] Running benchmark...")
    print("-" * 40)

    runner = BenchmarkRunner(
        sop=sop,
        genome=genome,
        llm_provider=provider,
    )

    results = runner.run(dataset)

    print("-" * 40)
    print(f"  ✓ Completed: {results.completed_samples}/{results.total_samples}")
    print(f"  ✗ Failed: {results.failed_samples}")
    print(f"  ⏱ Runtime: {results.total_time_seconds:.2f}s")

    # Step 5: Generate reports
    print("\n[5/5] Generating reports")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reporter = BenchmarkReporter()

    # Save reports
    json_path = output_path / f"verify_{timestamp}.json"
    md_path = output_path / f"verify_{timestamp}.md"
    html_path = output_path / f"verify_{timestamp}.html"

    reporter.save_report(results, str(json_path), "json")
    reporter.save_report(results, str(md_path), "markdown")
    reporter.save_report(results, str(html_path), "html")

    print(f"  ✓ JSON: {json_path}")
    print(f"  ✓ Markdown: {md_path}")
    print(f"  ✓ HTML: {html_path}")

    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    summary = results.summary()
    print(f"Dataset: {summary['dataset']}")
    print(f"Samples: {summary['total_samples']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Runtime: {summary['total_time_seconds']:.2f}s")

    if summary["total_samples"] > 0:
        avg_latency = summary["total_time_seconds"] / summary["total_samples"]
        print(f"Avg Latency: {avg_latency:.2f}s per sample")

    # Compare to published baselines
    print("\nFRAMES Published Baselines:")
    for name, baseline in FRAMESDataset.PUBLISHED_BASELINES.items():
        print(f"  {name}: {baseline:.1%}")

    # Show sample outputs
    if results.sample_results:
        print("\n" + "-" * 40)
        print("Sample Outputs (first 3):")
        for i, sample in enumerate(results.sample_results[:3]):
            print(f"\n[{i+1}] Query: {sample.query[:80]}...")
            if sample.error:
                print(f"    Error: {sample.error}")
            else:
                answer_preview = sample.generated_answer[:100] if sample.generated_answer else "(empty)"
                print(f"    Answer: {answer_preview}...")
                print(f"    Latency: {sample.latency_ms:.0f}ms")

    print("\n" + "=" * 60)

    success = results.completed_samples > 0
    if success:
        print("✓ VERIFICATION PASSED")
    else:
        print("✗ VERIFICATION FAILED - No samples completed")

    return 0 if success else 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Verify SIARE benchmark suite")
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of samples to run (default: 50)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.2:1b",
        help="Ollama model to use (default: llama3.2:1b)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    return run_verification(
        samples=args.samples,
        model=args.model,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    sys.exit(main())
