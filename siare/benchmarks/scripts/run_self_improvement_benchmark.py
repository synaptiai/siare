#!/usr/bin/env python3
"""CLI script for running SIARE self-improvement benchmark.

This benchmark demonstrates SIARE's core value proposition:
**SOP evolution measurably improves RAG performance** while keeping
the model constant.

Two modes are supported:
1. **Prompt-only evolution** (default): Optimizes prompts while keeping topology fixed
2. **Topology evolution** (--enable-topology-evolution): Full SOP evolution including
   ADD_ROLE, REMOVE_ROLE, REWIRE_GRAPH mutations

Usage:
    # Prompt-only evolution (default)
    python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
        --provider openai --model gpt-5-nano \
        --dataset-tier 1 --generations 10

    # Full SOP topology evolution
    python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
        --provider openai --model gpt-5-nano \
        --dataset-tier 3 --generations 10 \
        --enable-topology-evolution \
        --output-dir results/topology_evolution

    # Quick test (3 generations, 20 samples)
    python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
        --provider openai --model gpt-5-nano \
        --quick --enable-topology-evolution

    # Publication-ready run (tier 3 = FRAMES)
    python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
        --provider openai --model gpt-5-nano --reasoning-model gpt-5-mini \
        --dataset-tier 3 --generations 15 --samples 100 \
        --enable-topology-evolution \
        --output-dir results/publication
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path


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
            timeout=10, check=False,
        )
        return model in result.stdout
    except Exception:
        return False


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run SIARE self-improvement benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test
  python -m siare.benchmarks.scripts.run_self_improvement_benchmark --quick

  # Full benchmark
  python -m siare.benchmarks.scripts.run_self_improvement_benchmark \\
      --model llama3.1:8b --generations 10 --samples 100

  # Publication-ready (FRAMES dataset)
  python -m siare.benchmarks.scripts.run_self_improvement_benchmark \\
      --dataset-tier 3 --generations 15 --samples 100
        """,
    )

    # Model options
    parser.add_argument(
        "--provider",
        type=str,
        choices=["ollama", "openai", "anthropic"],
        default="ollama",
        help="LLM provider type (default: ollama)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama3.1:8b",
        help="LLM model for execution (default: llama3.1:8b for ollama, gpt-5-nano for openai)",
    )
    parser.add_argument(
        "--reasoning-model",
        type=str,
        default="deepseek-r1:7b",
        help="Reasoning model for Director (default: deepseek-r1:7b for ollama, gpt-5-mini for openai)",
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
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["adaptive", "textgrad", "evoprompt", "metaprompt"],
        default="adaptive",
        help="Prompt optimization strategy (default: adaptive)",
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
        default="benchmarks/results/self_improvement",
        help="Output directory for reports (default: benchmarks/results/self_improvement)",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=True,
        help="Generate Markdown report (default: True)",
    )
    parser.add_argument(
        "--html",
        action="store_true",
        default=True,
        help="Generate HTML report (default: True)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Generate JSON data file",
    )

    # Convergence control
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable convergence detection, run all generations",
    )
    parser.add_argument(
        "--convergence-threshold",
        type=float,
        default=0.01,
        help="Minimum improvement to continue evolution (default: 0.01 = 1%%)",
    )
    parser.add_argument(
        "--convergence-window",
        type=int,
        default=20,
        help="Number of generations to check for convergence (default: 20)",
    )

    # Parallelization options
    parser.add_argument(
        "--parallel-samples",
        type=int,
        default=8,
        help="Number of samples to evaluate concurrently (default: 8)",
    )
    parser.add_argument(
        "--parallel-offspring",
        action="store_true",
        help="Evaluate offspring in parallel (default: sequential)",
    )

    # Crash recovery options
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint if available",
    )

    # Topology evolution options
    parser.add_argument(
        "--enable-topology-evolution",
        action="store_true",
        help="Enable SOP topology mutations (ADD_ROLE, REMOVE_ROLE, REWIRE_GRAPH)",
    )
    parser.add_argument(
        "--max-roles",
        type=int,
        default=8,
        help="Maximum roles when topology evolution is enabled (default: 8)",
    )
    parser.add_argument(
        "--mandatory-roles",
        type=str,
        nargs="+",
        default=["query_decomposer", "synthesizer"],
        help="Roles that cannot be removed (default: query_decomposer synthesizer)",
    )

    # Other options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Skip Ollama model availability check",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Set default models based on provider if using defaults
    if args.provider == "openai":
        if args.model == "llama3.1:8b":
            args.model = "gpt-5-nano"
        if args.reasoning_model == "deepseek-r1:7b":
            args.reasoning_model = "gpt-5-mini"
    elif args.provider == "anthropic":
        if args.model == "llama3.1:8b":
            args.model = "claude-3-haiku-20240307"
        if args.reasoning_model == "deepseek-r1:7b":
            args.reasoning_model = "claude-3-5-sonnet-20241022"

    # Check model/API availability based on provider
    if args.provider == "ollama":
        if not args.skip_model_check:
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
        logger.info(f"Using OpenAI provider with model: {args.model}, reasoning: {args.reasoning_model}")
    elif args.provider == "anthropic":
        import os
        if not os.getenv("ANTHROPIC_API_KEY"):
            logger.error("ANTHROPIC_API_KEY environment variable required for Anthropic provider")
            return 1
        logger.info(f"Using Anthropic provider with model: {args.model}, reasoning: {args.reasoning_model}")

    # Import benchmark components
    try:
        from siare.adapters.wikipedia_search import WikipediaSearchAdapter
        from siare.benchmarks.reports.self_improvement_report import SelfImprovementReport
        from siare.benchmarks.self_improvement_benchmark import (
            SelfImprovementBenchmark,
            SelfImprovementConfig,
        )
        from siare.benchmarks.sops import create_multihop_genome, create_multihop_sop
        from siare.providers.ollama_provider import OllamaProvider
        from siare.services.llm_provider import LLMProviderFactory
    except ImportError:
        logger.exception("Failed to import required modules")
        return 1

    # Configure
    if args.quick:
        config = SelfImprovementConfig(
            max_generations=3,
            population_size=2,
            model=args.model,
            reasoning_model=args.reasoning_model,
            dataset_tier=args.dataset_tier,
            max_samples=20,
            prompt_strategy=args.strategy,
            quick_mode=True,
            # Convergence control
            no_early_stop=args.no_early_stop,
            convergence_threshold=args.convergence_threshold,
            convergence_window=args.convergence_window,
            # Parallelization
            parallel_samples=args.parallel_samples,
            parallel_offspring=args.parallel_offspring,
            # Crash recovery
            resume=args.resume,
            output_dir=args.output_dir,
            # Topology evolution
            enable_topology_evolution=args.enable_topology_evolution,
            max_roles=args.max_roles,
            mandatory_roles=args.mandatory_roles,
        )
    else:
        config = SelfImprovementConfig(
            max_generations=args.generations,
            population_size=args.population,
            model=args.model,
            reasoning_model=args.reasoning_model,
            dataset_tier=args.dataset_tier,
            max_samples=args.samples,
            prompt_strategy=args.strategy,
            # Convergence control
            no_early_stop=args.no_early_stop,
            convergence_threshold=args.convergence_threshold,
            convergence_window=args.convergence_window,
            # Parallelization
            parallel_samples=args.parallel_samples,
            parallel_offspring=args.parallel_offspring,
            # Crash recovery
            resume=args.resume,
            output_dir=args.output_dir,
            # Topology evolution
            enable_topology_evolution=args.enable_topology_evolution,
            max_roles=args.max_roles,
            mandatory_roles=args.mandatory_roles,
        )

    # Create provider and SOP
    logger.info(f"Initializing {args.provider} LLM provider and SOP...")
    if args.provider == "ollama":
        llm_provider = OllamaProvider(model=config.model)
    elif args.provider == "openai":
        llm_provider = LLMProviderFactory.create("openai")
    elif args.provider == "anthropic":
        llm_provider = LLMProviderFactory.create("anthropic")
    else:
        logger.error(f"Unknown provider: {args.provider}")
        return 1

    base_sop = create_multihop_sop(model=config.model)
    base_genome = create_multihop_genome()

    # Create tool adapters
    wikipedia_adapter = WikipediaSearchAdapter(config={
        "max_results": 5,
        "timeout": 30,
        "extract_chars": 1000,
    })
    wikipedia_adapter.initialize()
    tool_adapters = {
        "web_search": wikipedia_adapter.execute,
    }

    # Create benchmark
    benchmark = SelfImprovementBenchmark(
        config=config,
        llm_provider=llm_provider,
        base_sop=base_sop,
        base_genome=base_genome,
        tool_adapters=tool_adapters,
    )

    # Run benchmark
    logger.info("=" * 60)
    logger.info("SIARE Self-Improvement Benchmark")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model}")
    logger.info(f"Reasoning Model: {config.reasoning_model}")
    logger.info(f"Dataset Tier: {config.dataset_tier}")
    logger.info(f"Max Generations: {config.max_generations}")
    logger.info(f"Max Samples: {config.max_samples}")
    if config.enable_topology_evolution:
        logger.info("Topology Evolution: ENABLED")
        logger.info(f"  Max Roles: {config.max_roles}")
        logger.info(f"  Mandatory Roles: {config.mandatory_roles}")
    else:
        logger.info("Topology Evolution: disabled (prompt-only)")
    logger.info("=" * 60)

    try:
        if args.quick:
            result = benchmark.run_quick()
        else:
            result = benchmark.run()
    except Exception:
        logger.exception("Benchmark failed")
        return 1

    # Print summary
    summary = result.summary()
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Generations evolved: {summary['generations']}")
    logger.info(f"Converged: {summary['converged']}")
    logger.info(f"Total time: {summary['total_time_seconds']:.1f}s")
    logger.info("")
    logger.info("Improvements:")
    for metric, data in summary["improvements"].items():
        sig = "✓" if data["significant"] else "✗"
        logger.info(
            f"  {metric}: {data['initial']:.4f} → {data['evolved']:.4f} "
            f"({data['improvement_pct']}) [{sig}]"
        )

    # Generate reports
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report = SelfImprovementReport(result)

    if args.markdown:
        md_path = output_dir / f"self_improvement_{timestamp}.md"
        report.save_markdown(md_path)
        logger.info(f"Markdown report: {md_path}")

    if args.html:
        html_path = output_dir / f"self_improvement_{timestamp}.html"
        report.save_html(html_path)
        logger.info(f"HTML report: {html_path}")

    if args.json:
        json_path = output_dir / f"self_improvement_{timestamp}.json"
        report.save_json(json_path)
        logger.info(f"JSON data: {json_path}")

    logger.info("=" * 60)
    logger.info("Benchmark complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
