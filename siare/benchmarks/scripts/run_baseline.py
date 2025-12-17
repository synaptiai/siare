"""Run baseline benchmarks on SIARE default SOP.

Usage:
    python -m siare.benchmarks.scripts.run_baseline --dataset frames --max-samples 100
    python -m siare.benchmarks.scripts.run_baseline --dataset beir-nq --max-samples 50
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from siare.benchmarks.reporter import BenchmarkReporter
from siare.benchmarks.runner import BenchmarkRunner


logger = logging.getLogger(__name__)


def run_frames_baseline(
    max_samples: Optional[int] = None,
    output_dir: Path = Path("benchmarks/results"),
) -> dict[str, Any]:
    """Run FRAMES benchmark baseline.

    Args:
        max_samples: Maximum samples to run
        output_dir: Directory to save results

    Returns:
        Dictionary with results summary
    """
    from siare.benchmarks.datasets.frames import FRAMESDataset

    logger.info("Loading FRAMES dataset...")
    dataset = FRAMESDataset(max_samples=max_samples)

    logger.info(f"Running benchmark on {len(dataset)} samples...")
    runner = BenchmarkRunner()
    results = runner.run(dataset)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reporter = BenchmarkReporter()

    # Save multiple formats
    reporter.save_report(
        results,
        str(output_dir / f"frames_{timestamp}.json"),
        "json",
    )
    reporter.save_report(
        results,
        str(output_dir / f"frames_{timestamp}.md"),
        "markdown",
    )
    reporter.save_report(
        results,
        str(output_dir / f"frames_{timestamp}_marketing.md"),
        "marketing",
    )

    logger.info(f"Results saved to {output_dir}")

    # Print summary
    summary = results.summary()
    print("\n" + "=" * 50)
    print("FRAMES Baseline Results")
    print("=" * 50)
    print(f"Samples: {summary['total_samples']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Runtime: {summary['total_time_seconds']:.2f}s")

    # Compare to published baselines
    from siare.benchmarks.datasets.frames import FRAMESDataset

    print("\nComparison to Published Baselines:")
    for name, baseline in FRAMESDataset.PUBLISHED_BASELINES.items():
        print(f"  {name}: {baseline:.1%}")

    return summary


def run_beir_baseline(
    dataset_name: str = "nq",
    max_samples: Optional[int] = None,
    output_dir: Path = Path("benchmarks/results"),
) -> dict[str, Any]:
    """Run BEIR benchmark baseline.

    Args:
        dataset_name: BEIR dataset to run
        max_samples: Maximum samples to run
        output_dir: Directory to save results

    Returns:
        Dictionary with results summary
    """
    from siare.benchmarks.datasets.beir import BEIRDataset

    logger.info(f"Loading BEIR-{dataset_name} dataset...")
    dataset = BEIRDataset(dataset_name=dataset_name, max_samples=max_samples)

    logger.info(f"Running benchmark on {len(dataset)} queries...")
    runner = BenchmarkRunner()
    results = runner.run(dataset)

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    reporter = BenchmarkReporter()
    reporter.save_report(
        results,
        str(output_dir / f"beir_{dataset_name}_{timestamp}.json"),
        "json",
    )
    reporter.save_report(
        results,
        str(output_dir / f"beir_{dataset_name}_{timestamp}.md"),
        "markdown",
    )

    logger.info(f"Results saved to {output_dir}")

    summary = results.summary()
    print("\n" + "=" * 50)
    print(f"BEIR-{dataset_name} Baseline Results")
    print("=" * 50)
    print(f"Queries: {summary['total_samples']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Runtime: {summary['total_time_seconds']:.2f}s")

    return summary


def main() -> int:
    """Main entry point for baseline runner."""
    parser = argparse.ArgumentParser(description="Run SIARE benchmark baselines")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to run: 'frames' or 'beir-<name>' (e.g., 'beir-nq')",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to run (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output_dir)

    if args.dataset == "frames":
        run_frames_baseline(args.max_samples, output_dir)
    elif args.dataset.startswith("beir-"):
        dataset_name = args.dataset[5:]  # Remove "beir-" prefix
        run_beir_baseline(dataset_name, args.max_samples, output_dir)
    else:
        print(f"Unknown dataset: {args.dataset}")
        print("Use 'frames' or 'beir-<name>' (e.g., 'beir-nq', 'beir-scifact')")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
