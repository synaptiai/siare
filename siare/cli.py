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

import sys
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

try:
    import questionary
except ImportError:
    questionary = None  # type: ignore[assignment]

from siare import __version__

console = Console()


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
def evolve(config: str, generations: int | None, metric: str | None) -> None:
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

    # Interactive prompts if options not provided
    if generations is None or metric is None:
        require_questionary()

    if generations is None:
        generations = int(
            questionary.text(  # type: ignore[union-attr]
                "Generations to run:",
                default="10",
                validate=lambda x: x.isdigit() and int(x) > 0,
            ).ask()
            or "10"
        )

    if metric is None:
        metric = (
            questionary.select(  # type: ignore[union-attr]
                "Primary metric to optimize:",
                choices=[
                    "answer_accuracy",
                    "latency",
                    "cost",
                    "faithfulness",
                ],
            ).ask()
            or "answer_accuracy"
        )

    console.print()
    console.print(f"[bold]Running evolution[/bold] ({generations} generations, optimizing {metric})")
    console.print()

    # TODO: Implement actual evolution loop
    # For now, show progress simulation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading configuration...", total=None)
        import time

        time.sleep(0.5)
        progress.update(task, description="Evolution not yet implemented - coming soon!")
        time.sleep(1)

    console.print()
    console.print(
        "[yellow]Note:[/yellow] Full evolution support coming in v1.1.0\n"
        "For now, see [cyan]examples/quickstart/[/cyan] for usage patterns."
    )


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
    help="Path to evolved SOP (default: .siare/best_sop.yaml)",
)
@click.argument("query", required=False)
def run(config: str, sop: str | None, query: str | None) -> None:
    """Execute a pipeline against a query.

    Runs the evolved (or default) SOP against a question.
    """
    if query is None:
        require_questionary()
        query = questionary.text("Your question:").ask()  # type: ignore[union-attr]

    if not query:
        console.print("[red]Error:[/red] No query provided.")
        sys.exit(1)

    console.print()
    console.print(f"[bold]Query:[/bold] {query}")
    console.print()

    # TODO: Implement actual execution
    console.print(
        "[yellow]Note:[/yellow] Full execution support coming in v1.1.0\n"
        "For now, see [cyan]examples/quickstart/[/cyan] for programmatic usage."
    )


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
    config: dict[str, Any] = {}

    # Project type
    project_type = questionary.select(  # type: ignore[union-attr]
        "What type of RAG system?",
        choices=[
            questionary.Choice("Customer Support (answer questions from docs)", value="support"),
            questionary.Choice("Research Assistant (multi-hop Q&A)", value="research"),
            questionary.Choice("Custom (define your own agents)", value="custom"),
        ],
    ).ask()

    # LLM provider
    llm_provider = questionary.select(  # type: ignore[union-attr]
        "LLM Provider?",
        choices=[
            questionary.Choice("OpenAI (requires OPENAI_API_KEY)", value="openai"),
            questionary.Choice("Ollama (local, free)", value="ollama"),
            questionary.Choice("Anthropic (requires ANTHROPIC_API_KEY)", value="anthropic"),
        ],
    ).ask()

    # Model selection based on provider
    if llm_provider == "openai":
        model = questionary.select(  # type: ignore[union-attr]
            "OpenAI Model?",
            choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
        ).ask()
    elif llm_provider == "ollama":
        model = questionary.text(  # type: ignore[union-attr]
            "Ollama model name:",
            default="llama3.2",
        ).ask()
    else:  # anthropic
        model = questionary.select(  # type: ignore[union-attr]
            "Anthropic Model?",
            choices=["claude-3-haiku-20240307", "claude-3-sonnet-20240229", "claude-3-opus-20240229"],
        ).ask()

    # Vector store
    vector_store = questionary.select(  # type: ignore[union-attr]
        "Vector store?",
        choices=[
            questionary.Choice("In-memory (simple, no setup)", value="memory"),
            questionary.Choice("ChromaDB (persistent)", value="chromadb"),
        ],
    ).ask()

    config = {
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

    return config


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


if __name__ == "__main__":
    main()
