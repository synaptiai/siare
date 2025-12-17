"""
SIARE Command Line Interface

Interactive CLI for building and evolving RAG pipelines.

Commands:
    siare init     - Initialize a new SIARE project
    siare evolve   - Run evolution on a pipeline
    siare run      - Execute a pipeline against queries
    siare version  - Show version information
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from siare.services.llm_provider import LLMProvider

try:
    import questionary
except ImportError:
    questionary = None  # type: ignore[assignment]

try:
    import yaml

    yaml_available = True
except ImportError:
    yaml = None  # type: ignore[assignment]
    yaml_available = False

from siare import __version__

console = Console()
logger = logging.getLogger(__name__)


def require_questionary() -> None:
    """Ensure questionary is available for interactive mode."""
    if questionary is None:
        console.print(
            "[red]Error:[/red] questionary is required for interactive mode.\n"
            "Install it with: [cyan]pip install siare[full][/cyan]"
        )
        sys.exit(1)


@click.group()
@click.version_option(version=__version__, prog_name="siare")
def main() -> None:
    """SIARE - Self-Improving Agentic RAG Engine

    Build self-evolving multi-agent RAG systems using Quality-Diversity
    optimization and evolutionary algorithms.

    Get started:
        siare init
    """


@main.command()
@click.option(
    "--non-interactive",
    is_flag=True,
    help="Skip prompts and use defaults",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    default="siare.yaml",
    help="Output config file path",
)
def init(non_interactive: bool, output: str) -> None:
    """Initialize a new SIARE project.

    Creates a siare.yaml configuration file and example documents.
    """
    console.print(
        Panel.fit(
            "[bold cyan]SIARE Project Initialization[/bold cyan]\n"
            "Let's set up your self-evolving RAG pipeline.",
            border_style="cyan",
        )
    )

    if non_interactive:
        config = _get_default_config()
    else:
        require_questionary()
        config = _interactive_init()

    # Write config file
    config_path = Path(output)
    _write_config(config_path, config)

    # Create example docs directory
    docs_dir = Path("example_docs")
    docs_dir.mkdir(exist_ok=True)
    _create_sample_docs(docs_dir)

    console.print()
    console.print("[green]✓[/green] Created [cyan]siare.yaml[/cyan]")
    console.print("[green]✓[/green] Created [cyan]example_docs/[/cyan] with sample documents")
    console.print()
    console.print("[bold]Next steps:[/bold]")
    console.print("  1. Add your documents to [cyan]example_docs/[/cyan]")
    console.print("  2. Run: [cyan]siare evolve[/cyan]")


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="siare.yaml",
    help="Config file path",
)
@click.option(
    "--generations",
    "-g",
    type=int,
    default=None,
    help="Number of generations to run",
)
@click.option(
    "--metric",
    "-m",
    type=str,
    default=None,
    help="Primary metric to optimize",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output",
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(),
    default=".siare",
    help="Output directory for results",
)
def evolve(
    config: str,
    generations: int | None,
    metric: str | None,
    verbose: bool,
    output_dir: str,
) -> None:
    """Run evolution to optimize your RAG pipeline.

    Evolves the pipeline configuration over multiple generations,
    using Quality-Diversity optimization to find high-performing
    and diverse solutions.
    """
    config_path = Path(config)
    if not config_path.exists():
        console.print(
            f"[red]Error:[/red] Config file not found: {config}\n"
            "Run [cyan]siare init[/cyan] first to create a project."
        )
        sys.exit(1)

    # Load YAML config
    project_config = _load_yaml_config(config_path)
    if project_config is None:
        sys.exit(1)

    # Interactive prompts if options not provided
    if generations is None or metric is None:
        require_questionary()
        assert questionary is not None  # For type narrowing after require_questionary()

    if generations is None:
        assert questionary is not None  # Type narrowing for this block

        def _validate_generations(text: str) -> bool:
            return text.isdigit() and int(text) > 0

        generations_str: str = (
            questionary.text(
                "Generations to run:",
                default=str(project_config.get("evolution", {}).get("generations", 10)),
                validate=_validate_generations,
            ).ask()
            or "10"
        )
        generations = int(generations_str)

    if metric is None:
        assert questionary is not None  # Type narrowing for this block
        default_metric: str = project_config.get("evolution", {}).get("metric", "accuracy")
        metric_result: str | None = questionary.select(
            "Primary metric to optimize:",
            choices=[
                "accuracy",
                "relevance",
                "faithfulness",
                "completeness",
                "latency",
                "cost",
            ],
            default=default_metric,
        ).ask()
        metric = metric_result if metric_result is not None else "accuracy"

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    console.print()
    console.print(
        Panel.fit(
            f"[bold cyan]Starting Evolution[/bold cyan]\n"
            f"Config: {config}\n"
            f"Generations: {generations}\n"
            f"Metric: {metric}",
            border_style="cyan",
        )
    )
    console.print()

    # Run evolution
    try:
        result = _run_evolution(
            project_config=project_config,
            generations=generations,
            metric=metric,
            output_path=output_path,
            verbose=verbose,
        )

        # Display results
        _display_evolution_results(result, output_path)

    except KeyboardInterrupt:
        console.print("\n[yellow]Evolution interrupted by user[/yellow]")
        sys.exit(130)
    except (ValueError, RuntimeError, OSError) as e:
        console.print(f"\n[red]Error during evolution:[/red] {e}")
        if verbose:
            console.print_exception()
        sys.exit(1)


@main.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default="siare.yaml",
    help="Config file path",
)
@click.option(
    "--sop",
    "-s",
    type=click.Path(exists=True),
    default=None,
    help="Path to evolved SOP (default: .siare/best_sop.json)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose output with execution trace",
)
@click.option(
    "--json-output",
    "-j",
    is_flag=True,
    help="Output result as JSON",
)
@click.argument("query", required=False)
def run(
    config: str,
    sop: str | None,
    verbose: bool,
    json_output: bool,
    query: str | None,
) -> None:
    """Execute a pipeline against a query.

    Runs the evolved (or default) SOP against a question.
    """
    config_path = Path(config)
    if not config_path.exists():
        console.print(
            f"[red]Error:[/red] Config file not found: {config}\n"
            "Run [cyan]siare init[/cyan] first to create a project."
        )
        sys.exit(1)

    # Load project config
    project_config = _load_yaml_config(config_path)
    if project_config is None:
        sys.exit(1)

    # Interactive query if not provided
    if query is None:
        require_questionary()
        query = questionary.text("Your question:").ask()  # type: ignore[union-attr]

    if not query:
        console.print("[red]Error:[/red] No query provided.")
        sys.exit(1)

    # Determine SOP path
    sop_path = Path(sop) if sop else Path(".siare/best_sop.json")

    if not json_output:
        console.print()
        console.print(f"[bold]Query:[/bold] {query}")
        console.print()

    # Run execution
    try:
        result = _run_query(
            project_config=project_config,
            sop_path=sop_path if sop_path.exists() else None,
            query=query,
            verbose=verbose,
        )

        # Display results
        if json_output:
            console.print(json.dumps(result, indent=2))
        else:
            _display_run_results(result, verbose)

    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as e:
        if json_output:
            console.print(json.dumps({"error": str(e)}, indent=2))
        else:
            console.print(f"\n[red]Error during execution:[/red] {e}")
            if verbose:
                console.print_exception()
        sys.exit(1)


@main.command()
def version() -> None:
    """Show version information."""
    console.print(f"[cyan]SIARE[/cyan] version [bold]{__version__}[/bold]")
    console.print("Self-Improving Agentic RAG Engine")
    console.print()
    console.print("GitHub: [link]https://github.com/synaptiai/siare[/link]")
    console.print("Docs:   [link]https://siare.dev[/link]")


# =============================================================================
# Helper Functions
# =============================================================================


def _get_default_config() -> dict[str, Any]:
    """Get default configuration."""
    return {
        "name": "my-rag-pipeline",
        "version": "1.0.0",
        "llm": {
            "provider": "openai",
            "model": "gpt-4o-mini",
        },
        "vector_store": {
            "type": "memory",
        },
        "evolution": {
            "generations": 10,
            "population_size": 5,
            "metric": "answer_accuracy",
        },
    }


def _interactive_init() -> dict[str, Any]:
    """Run interactive initialization prompts."""
    # This function is only called after require_questionary() check
    assert questionary is not None

    # Project type
    project_type: str = questionary.select(
        "What type of RAG system?",
        choices=[
            questionary.Choice("Customer Support (answer questions from docs)", value="support"),
            questionary.Choice("Research Assistant (multi-hop Q&A)", value="research"),
            questionary.Choice("Custom (define your own agents)", value="custom"),
        ],
    ).ask() or "support"

    # LLM provider
    llm_provider: str = questionary.select(
        "LLM Provider?",
        choices=[
            questionary.Choice("OpenAI (requires OPENAI_API_KEY)", value="openai"),
            questionary.Choice("Ollama (local, free)", value="ollama"),
            questionary.Choice("Anthropic (requires ANTHROPIC_API_KEY)", value="anthropic"),
        ],
    ).ask() or "openai"

    # Model selection based on provider
    model: str
    if llm_provider == "openai":
        model = questionary.select(
            "OpenAI Model?",
            choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        ).ask() or "gpt-4o-mini"
    elif llm_provider == "ollama":
        model = questionary.text(
            "Ollama model name:",
            default="llama3.2",
        ).ask() or "llama3.2"
    else:  # anthropic
        model = questionary.select(
            "Anthropic Model?",
            choices=["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
        ).ask() or "claude-3-haiku-20240307"

    # Vector store
    vector_store: str = questionary.select(
        "Vector store?",
        choices=[
            questionary.Choice("In-memory (simple, no setup)", value="memory"),
            questionary.Choice("ChromaDB (persistent)", value="chromadb"),
        ],
    ).ask() or "memory"

    return {
        "name": f"{project_type}-pipeline",
        "version": "1.0.0",
        "type": project_type,
        "llm": {
            "provider": llm_provider,
            "model": model,
        },
        "vector_store": {
            "type": vector_store,
        },
        "evolution": {
            "generations": 10,
            "population_size": 5,
            "metric": "answer_accuracy",
        },
    }


def _write_config(path: Path, config: dict[str, Any]) -> None:
    """Write configuration to YAML file."""
    import yaml

    with path.open("w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)


def _create_sample_docs(docs_dir: Path) -> None:
    """Create sample documents for the example."""
    # Sample FAQ document
    faq_path = docs_dir / "faq.md"
    if not faq_path.exists():
        faq_path.write_text(
            """# Frequently Asked Questions

## How do I reset my password?

To reset your password:
1. Go to Settings > Security
2. Click "Reset Password"
3. Enter your email address
4. Check your inbox for a reset link

## What are the supported file formats?

We support the following file formats:
- PDF documents
- Microsoft Word (.docx)
- Plain text (.txt)
- Markdown (.md)

## How do I contact support?

You can reach our support team:
- Email: support@example.com
- Chat: Available 9am-5pm EST
- Phone: 1-800-EXAMPLE
"""
        )

    # Sample user guide
    guide_path = docs_dir / "user-guide.md"
    if not guide_path.exists():
        guide_path.write_text(
            """# User Guide

## Getting Started

Welcome to our platform! This guide will help you get up and running quickly.

### Installation

1. Download the installer from our website
2. Run the installer and follow the prompts
3. Launch the application

### Creating Your First Project

1. Click "New Project" in the dashboard
2. Choose a template or start from scratch
3. Configure your project settings
4. Start adding content

## Advanced Features

### Automation

You can automate repetitive tasks using our workflow builder:
- Drag and drop actions
- Set triggers and conditions
- Test before deploying

### Integrations

Connect with your favorite tools:
- Slack
- Microsoft Teams
- Google Workspace
- Zapier
"""
        )


def _load_yaml_config(config_path: Path) -> dict[str, Any] | None:
    """Load and validate YAML configuration."""
    if not yaml_available or yaml is None:
        console.print(
            "[red]Error:[/red] PyYAML is required for config files.\n"
            "Install it with: [cyan]pip install pyyaml[/cyan]"
        )
        return None

    try:
        with config_path.open() as f:
            raw_config: Any = yaml.safe_load(f)
    except yaml.YAMLError as e:
        console.print(f"[red]Error parsing YAML:[/red] {e}")
        return None

    if not isinstance(raw_config, dict):
        console.print("[red]Error:[/red] Config file must be a YAML dictionary")
        return None

    # Cast to proper type - yaml.safe_load returns dict[Any, Any] for dicts
    config: dict[str, Any] = cast("dict[str, Any]", raw_config)
    return config


def _create_llm_provider(config: dict[str, Any]) -> LLMProvider:
    """Create LLM provider from config."""
    from siare.services.llm_provider import LLMProviderFactory

    llm_config = config.get("llm", {})
    provider_type = llm_config.get("provider", "openai")

    # Check for API key
    if provider_type == "openai" and not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Set it with: export OPENAI_API_KEY=your-key"
        )
    if provider_type == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError(
            "ANTHROPIC_API_KEY environment variable not set.\n"
            "Set it with: export ANTHROPIC_API_KEY=your-key"
        )

    return LLMProviderFactory.create(
        provider_type=provider_type,
        model=llm_config.get("model"),
    )


def _create_default_sop(config: dict[str, Any]) -> tuple[Any, Any]:
    """Create a default SOP and PromptGenome for a basic RAG pipeline."""
    from siare.core.models import (
        GraphEdge,
        ProcessConfig,
        PromptGenome,
        RoleConfig,
        RoleInput,
        RolePrompt,
    )

    project_type = config.get("type", "support")
    llm_config = config.get("llm", {})
    model = llm_config.get("model", "gpt-4o-mini")

    # Create a simple RAG pipeline based on project type
    if project_type == "support":
        roles = [
            RoleConfig(
                id="retriever",
                model=model,
                promptRef="retriever_prompt",
                inputs=[RoleInput(from_="user_input", fields=["query"])],
                outputs=["context"],
            ),
            RoleConfig(
                id="answerer",
                model=model,
                promptRef="answerer_prompt",
                inputs=[
                    RoleInput(from_="user_input", fields=["query"]),
                    RoleInput(from_="retriever", fields=["context"]),
                ],
                outputs=["answer"],
            ),
        ]
        graph = [
            GraphEdge(from_="user_input", to="retriever"),
            GraphEdge(from_="retriever", to="answerer"),
        ]
        models_dict = {"retriever": model, "answerer": model}
    else:
        # Research or custom - single agent
        roles = [
            RoleConfig(
                id="agent",
                model=model,
                promptRef="agent_prompt",
                inputs=[RoleInput(from_="user_input", fields=["query"])],
                outputs=["answer"],
            ),
        ]
        graph = [
            GraphEdge(from_="user_input", to="agent"),
        ]
        models_dict = {"agent": model}

    sop = ProcessConfig(
        id=f"sop-{config.get('name', 'default')}",
        version="1.0.0",
        description=config.get("name", "Default RAG Pipeline"),
        models=models_dict,
        tools=[],
        roles=roles,
        graph=graph,
    )

    # Create prompts
    if project_type == "support":
        role_prompts = {
            "retriever_prompt": RolePrompt(
                id="retriever_prompt",
                content=(
                    "You are a document retriever. Given a query, identify and return "
                    "the most relevant information from the available documents.\n\n"
                    "Query: {query}\n\n"
                    "Return the relevant context that would help answer this query."
                ),
            ),
            "answerer_prompt": RolePrompt(
                id="answerer_prompt",
                content=(
                    "You are a helpful customer support agent. Based on the provided "
                    "context, answer the user's question accurately and helpfully.\n\n"
                    "Query: {query}\n"
                    "Context: {context}\n\n"
                    "Provide a clear, helpful answer based on the context."
                ),
            ),
        }
    else:
        role_prompts = {
            "agent_prompt": RolePrompt(
                id="agent_prompt",
                content=(
                    "You are a helpful AI assistant. Answer the user's question "
                    "accurately and thoroughly.\n\n"
                    "Query: {query}\n\n"
                    "Provide a comprehensive answer."
                ),
            ),
        }

    genome = PromptGenome(
        id=f"genome-{config.get('name', 'default')}",
        version="1.0.0",
        rolePrompts=role_prompts,
    )

    return sop, genome


def _run_evolution(
    project_config: dict[str, Any],
    generations: int,
    metric: str,
    output_path: Path,
    verbose: bool,
) -> dict[str, Any]:
    """Run the evolution loop with real services."""
    from siare.core.models import (
        BudgetUsage,
        EvolutionConstraints,
        EvolutionJob,
        EvolutionJobStatus,
        EvolutionPhase,
        MetricConfig,
        MetricType,
        MutationType,
        SelectionStrategy,
        TaskSet,
    )
    from siare.services.config_store import ConfigStore
    from siare.services.director import DirectorService
    from siare.services.evaluation_service import EvaluationService
    from siare.services.execution_engine import ExecutionEngine
    from siare.services.gene_pool import GenePool
    from siare.services.qd_grid import QDGridManager
    from siare.services.scheduler import EvolutionScheduler

    # Initialize LLM provider
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        init_task = progress.add_task("Initializing services...", total=None)

        # Create LLM provider
        llm_provider = _create_llm_provider(project_config)

        # Initialize services
        config_store = ConfigStore(storage_path=str(output_path / "config_store"))
        gene_pool = GenePool()
        qd_grid = QDGridManager()
        execution_engine = ExecutionEngine(llm_provider=llm_provider)
        evaluation_service = EvaluationService(llm_provider=llm_provider)
        director_service = DirectorService(llm_provider=llm_provider)

        # Create default SOP and genome
        progress.update(init_task, description="Creating pipeline configuration...")
        sop, genome = _create_default_sop(project_config)

        # Save to config store
        config_store.save_sop(sop)
        config_store.save_prompt_genome(genome)

        # Create metric configuration
        metric_config = MetricConfig(
            id=metric,
            type=MetricType.LLM_JUDGE,
            inputs=["query", "answer", "groundTruth"],
        )
        config_store.save_metric(metric_config)

        # Create task set from example docs if available
        progress.update(init_task, description="Loading evaluation tasks...")
        tasks = _create_evaluation_tasks(project_config)
        task_set = TaskSet(
            id="default-tasks",
            domain=project_config.get("type", "support"),
            version="1.0.0",
            tasks=tasks,
        )

        progress.update(init_task, description="Services initialized!")

    # Create evolution job
    job = EvolutionJob(
        id=f"job-{uuid.uuid4().hex[:8]}",
        domain=project_config.get("type", "support"),
        baseSops=[
            {
                "sopId": sop.id,
                "sopVersion": sop.version,
                "promptGenomeId": genome.id,
                "promptGenomeVersion": genome.version,
            }
        ],
        taskSet=task_set,
        metricsToOptimize=[metric],
        qualityScoreWeights={metric: 1.0},
        constraints=EvolutionConstraints(),
        phases=[
            EvolutionPhase(
                name="Main Evolution",
                maxGenerations=generations,
                selectionStrategy=SelectionStrategy.QD_QUALITY_WEIGHTED,
                parentsPerGeneration=1,
                allowedMutationTypes=[
                    MutationType.PROMPT_CHANGE,
                    MutationType.PARAM_TWEAK,
                ],
            ),
        ],
        status=EvolutionJobStatus.PENDING,
        budgetUsed=BudgetUsage(),
    )

    # Create scheduler
    scheduler = EvolutionScheduler(
        config_store=config_store,
        gene_pool=gene_pool,
        qd_grid=qd_grid,
        execution_engine=execution_engine,
        evaluation_service=evaluation_service,
        director_service=director_service,
        checkpoint_dir=str(output_path / "checkpoints"),
    )

    # Run evolution with progress bar
    generation_stats: list[dict[str, Any]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        evo_task = progress.add_task(
            f"Generation 0/{generations}",
            total=generations,
        )

        def on_generation(gen: int, stats: dict[str, Any]) -> None:
            generation_stats.append(stats)
            progress.update(
                evo_task,
                completed=gen,
                description=(
                    f"Gen {gen}/{generations} | "
                    f"Best: {stats.get('best_quality', 0):.3f} | "
                    f"Offspring: {stats.get('offspring_count', 0)}"
                ),
            )
            if verbose:
                console.print(f"  [dim]Phase: {stats.get('phase', 'unknown')}[/dim]")

        # Run to completion
        completed_job = scheduler.run_to_completion(
            job,
            verbose=False,
            on_generation_complete=on_generation,
        )

    # Save results
    _save_evolution_results(completed_job, scheduler, output_path, config_store)

    return {
        "job_id": completed_job.id,
        "status": completed_job.status.value,
        "generations": completed_job.currentGeneration,
        "best_sop": completed_job.bestSopSoFar,
        "budget_used": completed_job.budgetUsed.model_dump(),
        "generation_history": generation_stats,
        "output_dir": str(output_path),
    }


def _create_evaluation_tasks(_config: dict[str, Any]) -> list[Any]:
    """Create evaluation tasks from example docs.

    Args:
        _config: Project configuration (reserved for future use).

    Returns:
        List of evaluation tasks.
    """
    from siare.core.models import Task

    # Default evaluation tasks (future: read from config.tasks or docs directory)
    return [
        Task(
            id="task-1",
            input={"query": "How do I reset my password?"},
            groundTruth={
                "answer": "To reset your password: 1) Go to Settings > Security, "
                "2) Click 'Reset Password', 3) Enter your email, 4) Check your inbox."
            },
        ),
        Task(
            id="task-2",
            input={"query": "What file formats are supported?"},
            groundTruth={
                "answer": "Supported formats include PDF, Microsoft Word (.docx), "
                "plain text (.txt), and Markdown (.md)."
            },
        ),
        Task(
            id="task-3",
            input={"query": "How can I contact support?"},
            groundTruth={
                "answer": "Contact support via email at support@example.com, "
                "chat (9am-5pm EST), or phone at 1-800-EXAMPLE."
            },
        ),
    ]


def _save_evolution_results(
    job: Any,
    scheduler: Any,
    output_path: Path,
    config_store: Any,
) -> None:
    """Save evolution results to files."""
    # Save best SOP
    if job.bestSopSoFar:
        best_sop_id = job.bestSopSoFar.get("sopId")
        best_sop_version = job.bestSopSoFar.get("version")
        best_sop = config_store.get_sop(best_sop_id, best_sop_version)

        if best_sop:
            best_sop_path = output_path / "best_sop.json"
            with best_sop_path.open("w") as f:
                json.dump(best_sop.model_dump(mode="json"), f, indent=2)

        # Save best genome
        best_genome_id = job.bestSopSoFar.get("promptGenomeId")
        best_genome_version = job.bestSopSoFar.get("promptGenomeVersion")
        best_genome = config_store.get_prompt_genome(best_genome_id, best_genome_version)

        if best_genome:
            best_genome_path = output_path / "best_genome.json"
            with best_genome_path.open("w") as f:
                json.dump(best_genome.model_dump(mode="json"), f, indent=2)

    # Save job summary
    summary_path = output_path / "evolution_summary.json"
    summary: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status.value,
        "generations": job.currentGeneration,
        "best_quality": job.bestSopSoFar.get("quality") if job.bestSopSoFar else None,
        "best_metrics": job.bestSopSoFar.get("metrics") if job.bestSopSoFar else {},
        "budget_used": job.budgetUsed.model_dump(),
        "statistics": scheduler.get_statistics(),
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)


def _display_evolution_results(result: dict[str, Any], output_path: Path) -> None:
    """Display evolution results in a nice format."""
    console.print()
    console.print(
        Panel.fit(
            "[bold green]Evolution Complete![/bold green]",
            border_style="green",
        )
    )
    console.print()

    # Summary table
    table = Table(title="Evolution Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Job ID", result.get("job_id", "N/A"))
    table.add_row("Status", result.get("status", "N/A"))
    table.add_row("Generations", str(result.get("generations", 0)))

    best = result.get("best_sop", {})
    if best:
        table.add_row("Best Quality", f"{best.get('quality', 0):.4f}")
        metrics = best.get("metrics", {})
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                table.add_row(f"  {metric_name}", f"{metric_value:.4f}")

    budget = result.get("budget_used", {})
    table.add_row("Evaluations", str(budget.get("evaluations", 0)))
    table.add_row("LLM Calls", str(budget.get("llmCalls", 0)))
    table.add_row("Cost", f"${budget.get('cost', 0):.4f}")

    console.print(table)
    console.print()

    # Output files
    console.print("[bold]Output files:[/bold]")
    console.print(f"  Best SOP: [cyan]{output_path}/best_sop.json[/cyan]")
    console.print(f"  Best Genome: [cyan]{output_path}/best_genome.json[/cyan]")
    console.print(f"  Summary: [cyan]{output_path}/evolution_summary.json[/cyan]")
    console.print()

    console.print("[bold]Next steps:[/bold]")
    console.print('  Run: [cyan]siare run "your question"[/cyan]')


def _run_query(
    project_config: dict[str, Any],
    sop_path: Path | None,
    query: str,
    verbose: bool,
) -> dict[str, Any]:
    """Execute a query against the pipeline."""
    from siare.core.models import ProcessConfig, PromptGenome
    from siare.services.execution_engine import ExecutionEngine

    # Initialize LLM provider
    llm_provider = _create_llm_provider(project_config)

    # Load or create SOP
    if sop_path and sop_path.exists():
        with sop_path.open() as f:
            sop_data = json.load(f)
        sop = ProcessConfig(**sop_data)

        # Load genome
        genome_path = sop_path.parent / "best_genome.json"
        if genome_path.exists():
            with genome_path.open() as f:
                genome_data = json.load(f)
            genome = PromptGenome(**genome_data)
        else:
            # Create default genome
            _, genome = _create_default_sop(project_config)
    else:
        # Use default SOP
        sop, genome = _create_default_sop(project_config)

    # Create execution engine
    execution_engine = ExecutionEngine(llm_provider=llm_provider)

    # Execute
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Executing pipeline...", total=None)

        trace = execution_engine.execute(
            sop=sop,
            prompt_genome=genome,
            task_input={"query": query},
        )

        progress.update(task, description="Complete!")

    # Extract answer from final outputs
    answer: str | None = None
    for _role_id, outputs in trace.final_outputs.items():
        if isinstance(outputs, dict):
            output_dict: dict[str, Any] = cast("dict[str, Any]", outputs)
            answer = str(output_dict.get("answer") or output_dict.get("response") or output_dict)
            break
        answer = str(outputs)

    return {
        "query": query,
        "answer": answer,
        "status": trace.status,
        "duration_ms": (trace.end_time - trace.start_time).total_seconds() * 1000
        if trace.end_time
        else 0,
        "total_cost": trace.total_cost,
        "sop_id": trace.sop_id,
        "sop_version": trace.sop_version,
        "node_executions": len(trace.node_executions),
        "trace": trace.to_dict() if verbose else None,
    }


def _display_run_results(result: dict[str, Any], verbose: bool) -> None:
    """Display execution results."""
    console.print(
        Panel.fit(
            result.get("answer", "No answer generated"),
            title="[bold cyan]Answer[/bold cyan]",
            border_style="cyan",
        )
    )
    console.print()

    # Stats
    console.print(f"[dim]Status: {result.get('status', 'unknown')}[/dim]")
    console.print(f"[dim]Duration: {result.get('duration_ms', 0):.2f}ms[/dim]")
    console.print(f"[dim]Cost: ${result.get('total_cost', 0):.6f}[/dim]")
    console.print(f"[dim]Nodes executed: {result.get('node_executions', 0)}[/dim]")

    if verbose and result.get("trace"):
        console.print()
        console.print("[bold]Execution Trace:[/bold]")
        for node in result["trace"].get("node_executions", []):
            console.print(f"  • {node['role_id']}: {node['duration_ms']:.2f}ms")


if __name__ == "__main__":
    main()
