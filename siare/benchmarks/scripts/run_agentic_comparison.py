#!/usr/bin/env python3
"""Run agentic variation mode comparison benchmark.

Runs the self-improvement benchmark in multiple variation modes
(single_turn, agentic, adaptive) and generates a comparison report.

Usage:
    python -m siare.benchmarks.scripts.run_agentic_comparison \
        --provider ollama --model llama3.1:8b \
        --reasoning-model deepseek-r1:7b \
        --dataset-tier 1 --quick

    python -m siare.benchmarks.scripts.run_agentic_comparison \
        --provider openai --model gpt-4o-mini \
        --reasoning-model gpt-4o \
        --dataset-tier 1 --generations 10 --samples 50
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def setup_logging(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def check_ollama_model(model: str) -> bool:
    """Check if Ollama model is available."""
    try:
        import subprocess

        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        return model in result.stdout
    except Exception:
        return False


def print_comparison_table(
    results: dict[str, dict[str, Any]],
    logger: logging.Logger,
) -> None:
    """Print a comparison summary table across all completed modes.

    Args:
        results: Mapping of mode name to its summary dict.
        logger: Logger instance for output.
    """
    modes = list(results.keys())
    if not modes:
        logger.warning("No results to compare.")
        return

    # Collect all metric names across modes
    all_metrics: set[str] = set()
    for summary in results.values():
        all_metrics.update(summary.get("improvements", {}).keys())

    logger.info("")
    logger.info("=" * 72)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 72)

    # Header
    header = f"{'Metric':<25}"
    for mode in modes:
        header += f" | {mode:>14}"
    logger.info(header)
    logger.info("-" * len(header))

    for metric in sorted(all_metrics):
        row = f"{metric:<25}"
        for mode in modes:
            data = results[mode].get("improvements", {}).get(metric)
            if data:
                pct = data["improvement_pct"]
                sig = "*" if data["significant"] else ""
                row += f" | {pct:>12}{sig:>2}"
            else:
                row += f" | {'N/A':>14}"
        logger.info(row)

    logger.info("-" * len(header))
    logger.info("(* = statistically significant)")

    # Timing row
    time_row = f"{'Time (s)':<25}"
    for mode in modes:
        t = results[mode].get("total_time_seconds", 0.0)
        time_row += f" | {t:>13.1f} "
    logger.info(time_row)

    # Generations row
    gen_row = f"{'Generations':<25}"
    for mode in modes:
        g = results[mode].get("generations", "?")
        gen_row += f" | {g!s:>14}"
    logger.info(gen_row)

    logger.info("=" * 72)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run agentic variation mode comparison benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick comparison across all modes
  python -m siare.benchmarks.scripts.run_agentic_comparison \\
      --provider ollama --model llama3.1:8b \\
      --reasoning-model deepseek-r1:7b --quick

  # Full comparison with specific modes
  python -m siare.benchmarks.scripts.run_agentic_comparison \\
      --provider openai --model gpt-4o-mini \\
      --reasoning-model gpt-4o \\
      --modes single_turn,agentic --generations 10
        """,
    )

    # Model options
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai", "anthropic"],
        required=True,
        help="LLM provider type",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="LLM model for execution",
    )
    parser.add_argument(
        "--reasoning-model",
        type=str,
        required=True,
        help="Reasoning model for Director and AgenticDirector",
    )

    # Dataset options
    parser.add_argument(
        "--dataset-tier",
        type=int,
        choices=[1, 2, 3],
        default=1,
        help="Dataset difficulty: 1=BEIR/NQ, 2=HotpotQA, 3=FRAMES (default: 1)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="Number of benchmark samples (default: 50)",
    )

    # Evolution options
    parser.add_argument(
        "--generations",
        type=int,
        default=10,
        help="Maximum evolution generations (default: 10)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=3,
        help="Population size per generation (default: 3)",
    )

    # Mode selection
    parser.add_argument(
        "--modes",
        type=str,
        default="single_turn,agentic,adaptive",
        help=(
            "Comma-separated variation modes to compare "
            "(default: single_turn,agentic,adaptive)"
        ),
    )

    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 3 generations, 20 samples",
    )

    # Output options
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmarks/results/agentic_comparison",
        help="Output directory for reports (default: benchmarks/results/agentic_comparison)",
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Parse modes
    valid_modes = {"single_turn", "agentic", "adaptive"}
    requested_modes = [m.strip() for m in args.modes.split(",")]
    for mode in requested_modes:
        if mode not in valid_modes:
            logger.error(
                f"Invalid mode '{mode}'. Must be one of: {', '.join(sorted(valid_modes))}"
            )
            return 1

    # Check model/API availability based on provider
    if args.provider == "ollama":
        logger.info(f"Checking model availability: {args.model}")
        if not check_ollama_model(args.model):
            logger.error(f"Model {args.model} not found. Run: ollama pull {args.model}")
            return 1
        if args.reasoning_model != args.model:
            logger.info(f"Checking reasoning model: {args.reasoning_model}")
            if not check_ollama_model(args.reasoning_model):
                logger.warning(
                    f"Reasoning model {args.reasoning_model} not found. "
                    f"Falling back to {args.model}"
                )
                args.reasoning_model = args.model
    elif args.provider == "openai":
        import os
        if not os.getenv("OPENAI_API_KEY"):
            logger.error("OPENAI_API_KEY environment variable required for OpenAI provider")
            return 1
    elif args.provider == "anthropic":
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.error(
                "ANTHROPIC_API_KEY environment variable required for Anthropic provider"
            )
            return 1

    # Import benchmark components
    try:
        from siare.adapters.wikipedia_search import WikipediaSearchAdapter
        from siare.benchmarks.self_improvement_benchmark import (
            SelfImprovementBenchmark,
            SelfImprovementConfig,
        )
        from siare.benchmarks.sops import create_multihop_genome, create_multihop_sop
        from siare.core.models import AgenticVariationConfig
        from siare.services.llm_provider import LLMProviderFactory
    except ImportError:
        logger.exception("Failed to import required modules")
        return 1

    # Create LLM provider (shared across all mode runs)
    logger.info(f"Initializing {args.provider} LLM provider...")
    if args.provider == "ollama":
        try:
            from siare.providers.ollama_provider import OllamaProvider
            llm_provider = OllamaProvider(model=args.model)
        except ImportError:
            llm_provider = LLMProviderFactory.create(
                "ollama", model=args.model,
            )
    elif args.provider == "openai":
        llm_provider = LLMProviderFactory.create("openai")
    elif args.provider == "anthropic":
        llm_provider = LLMProviderFactory.create("anthropic")
    else:
        logger.error(f"Unknown provider: {args.provider}")
        return 1

    # Create base SOP and genome (same starting point for all modes)
    base_sop = create_multihop_sop(model=args.model)
    base_genome = create_multihop_genome()

    # Create tool adapters
    wikipedia_adapter = WikipediaSearchAdapter(config={
        "max_results": 5,
        "timeout": 30,
        "extract_chars": 1000,
    })
    wikipedia_adapter.initialize()
    tool_adapters: dict[str, Any] = {
        "web_search": wikipedia_adapter.execute,
    }

    # Resolve generation / sample counts
    generations = 3 if args.quick else args.generations
    samples = 20 if args.quick else args.samples
    population = 2 if args.quick else args.population

    # Banner
    logger.info("=" * 72)
    logger.info("SIARE Agentic Variation Mode Comparison")
    logger.info("=" * 72)
    logger.info(f"Provider:        {args.provider}")
    logger.info(f"Model:           {args.model}")
    logger.info(f"Reasoning Model: {args.reasoning_model}")
    logger.info(f"Dataset Tier:    {args.dataset_tier}")
    logger.info(f"Generations:     {generations}")
    logger.info(f"Samples:         {samples}")
    logger.info(f"Population:      {population}")
    logger.info(f"Modes:           {', '.join(requested_modes)}")
    if args.random_seed is not None:
        logger.info(f"Random Seed:     {args.random_seed}")
    logger.info("=" * 72)

    # Run each mode independently
    mode_results: dict[str, Any] = {}
    mode_summaries: dict[str, dict[str, Any]] = {}

    for i, mode in enumerate(requested_modes, start=1):
        logger.info("")
        logger.info("#" * 72)
        logger.info(f"# MODE {i}/{len(requested_modes)}: {mode}")
        logger.info("#" * 72)

        agentic_config = AgenticVariationConfig(
            mode=mode,
            agentModel=args.reasoning_model,
        )

        config = SelfImprovementConfig(
            max_generations=generations,
            population_size=population,
            model=args.model,
            reasoning_model=args.reasoning_model,
            dataset_tier=args.dataset_tier,
            max_samples=samples,
            quick_mode=args.quick,
            output_dir=str(
                Path(args.output_dir) / mode
            ),
            agentic_config=agentic_config,
            random_seed=args.random_seed,
        )

        benchmark = SelfImprovementBenchmark(
            config=config,
            llm_provider=llm_provider,
            base_sop=base_sop,
            base_genome=base_genome,
            tool_adapters=tool_adapters,
        )

        try:
            if args.quick:
                result = benchmark.run_quick()
            else:
                result = benchmark.run()

            mode_results[mode] = result
            summary = result.summary()
            mode_summaries[mode] = summary

            logger.info(f"Mode '{mode}' completed in {summary['total_time_seconds']:.1f}s")
            logger.info(f"  Generations: {summary['generations']}")
            logger.info(f"  Converged:   {summary['converged']}")
            for metric, data in summary["improvements"].items():
                sig = "Y" if data["significant"] else "N"
                logger.info(
                    f"  {metric}: {data['initial']:.4f} -> {data['evolved']:.4f} "
                    f"({data['improvement_pct']}) [significant={sig}]"
                )
        except Exception:
            logger.exception(f"Mode '{mode}' FAILED -- continuing with remaining modes")
            continue

    # Print comparison table
    if mode_summaries:
        print_comparison_table(mode_summaries, logger)
    else:
        logger.error("All modes failed. No results to compare.")
        return 1

    # Generate comparison report
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    try:
        from siare.benchmarks.reports.agentic_comparison_report import (
            AgenticComparisonReport,
        )

        mode_configs_dict = {
            mode: {"mode": mode} for mode in mode_summaries
        }
        report = AgenticComparisonReport(mode_summaries, mode_configs_dict)
        report.save(str(output_dir))
        logger.info(f"Comparison report saved to: {output_dir}")
    except ImportError:
        logger.warning(
            "AgenticComparisonReport not available; skipping markdown report."
        )

    # Always save raw JSON regardless of report availability
    json_path = output_dir / f"agentic_comparison_{timestamp}.json"
    json_data: dict[str, Any] = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "provider": args.provider,
            "model": args.model,
            "reasoning_model": args.reasoning_model,
            "dataset_tier": args.dataset_tier,
            "generations": generations,
            "samples": samples,
            "population": population,
            "random_seed": args.random_seed,
            "modes": requested_modes,
        },
        "summaries": {
            mode: summary for mode, summary in mode_summaries.items()
        },
    }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    logger.info(f"JSON data: {json_path}")

    logger.info("")
    logger.info("=" * 72)
    logger.info("Agentic comparison benchmark complete!")
    logger.info(f"  Modes run:    {len(mode_summaries)}/{len(requested_modes)}")
    logger.info(f"  Output dir:   {output_dir}")
    logger.info("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
