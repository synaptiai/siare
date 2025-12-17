#!/usr/bin/env python3
"""
SIARE Quickstart Example: Customer Support RAG

This example demonstrates:
1. Creating a simple RAG pipeline configuration
2. Executing the pipeline against queries
3. Evolving the pipeline to improve performance

Requirements:
    pip install siare[full]
    export OPENAI_API_KEY="your-key"  # or use Ollama

Usage:
    python main.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from siare import pipeline, role, edge, task


RETRIEVER_PROMPT = """You are a document retrieval specialist for customer support.

Your job is to identify and extract the most relevant information from our knowledge base
to answer customer questions.

Guidelines:
- Focus on exact matches to the customer's question
- Return 2-3 most relevant passages
- Include source references (file name, section)
- If no relevant content exists, say "No relevant documents found"

Format your response as:
---
Source: [filename]
Content: [relevant passage]
---"""

ANSWERER_PROMPT = """You are a friendly and helpful customer support assistant.

Your job is to answer customer questions based ONLY on the provided context from our docs.

Guidelines:
- Answer concisely and accurately
- Always cite your sources in [brackets]
- If the context doesn't contain the answer, say "I don't have information about that"
- Never make up information not in the context
- Use a helpful, professional tone

Example:
User: How do I reset my password?
Context: [FAQ] To reset your password, go to Settings > Security > Reset Password...
Answer: To reset your password, go to Settings > Security > Reset Password [FAQ]."""


def create_customer_support_pipeline():
    """Create a simple customer support RAG pipeline.

    The pipeline has two agents:
    1. Retriever - finds relevant documents
    2. Answerer - generates answers from context
    """
    return pipeline(
        "customer-support-rag",
        roles=[
            role("retriever", "gpt-4o-mini", RETRIEVER_PROMPT, tools=["vector_search"]),
            role("answerer", "gpt-4o-mini", ANSWERER_PROMPT),
        ],
        edges=[
            edge("retriever", "answerer"),
        ],
        description="A self-evolving customer support RAG pipeline",
    )


def load_sample_documents(docs_dir: Path) -> list[dict]:
    """Load sample documents from the docs directory."""
    documents = []

    if not docs_dir.exists():
        docs_dir.mkdir(parents=True, exist_ok=True)
        _create_sample_docs(docs_dir)

    for doc_path in docs_dir.glob("*.md"):
        content = doc_path.read_text()
        documents.append({
            "id": doc_path.stem,
            "content": content,
            "source": doc_path.name,
        })

    return documents


def _create_sample_docs(docs_dir: Path) -> None:
    """Create sample customer support documents."""
    faq_content = """# Frequently Asked Questions

## Account & Security

### How do I reset my password?
To reset your password:
1. Go to Settings > Security
2. Click "Reset Password"
3. Enter your email address
4. Check your inbox for a reset link (expires in 24 hours)

### How do I enable two-factor authentication?
To enable 2FA:
1. Go to Settings > Security > Two-Factor Authentication
2. Click "Enable 2FA"
3. Scan the QR code with your authenticator app
4. Enter the 6-digit code to confirm

## Billing

### What payment methods do you accept?
We accept:
- Credit/Debit cards (Visa, Mastercard, Amex)
- PayPal
- Bank transfers (for annual plans)

### How do I cancel my subscription?
To cancel your subscription:
1. Go to Settings > Billing
2. Click "Manage Subscription"
3. Select "Cancel Subscription"
Note: You'll retain access until the end of your billing period.

## Technical Support

### What are the system requirements?
Minimum requirements:
- Windows 10/11 or macOS 10.15+
- 4GB RAM
- 2GB free disk space
- Internet connection

### How do I contact support?
You can reach our support team:
- Email: support@example.com
- Live Chat: Available 9am-5pm EST
- Phone: 1-800-EXAMPLE
"""

    user_guide_content = """# User Guide

## Getting Started

Welcome to our platform! This guide helps you get up and running quickly.

### Installation

1. Download the installer from our website
2. Run the installer and follow the prompts
3. Launch the application and sign in

### Creating Your First Project

1. Click "New Project" in the dashboard
2. Choose a template or start from scratch
3. Configure your project settings
4. Click "Create" to finish

## Features

### Dashboard
The dashboard shows:
- Recent projects
- Activity feed
- Quick actions
- Usage statistics

### Collaboration
Invite team members:
1. Go to Project Settings > Team
2. Click "Invite Member"
3. Enter their email address
4. Select their role (Viewer, Editor, Admin)

### Integrations
Connect with your favorite tools:
- Slack: Real-time notifications
- Google Drive: File sync
- Zapier: Automation workflows
"""

    (docs_dir / "faq.md").write_text(faq_content)
    (docs_dir / "user-guide.md").write_text(user_guide_content)


async def run_basic_example() -> None:
    """Run a basic RAG example without evolution."""
    print("=" * 60)
    print("SIARE Quickstart: Basic RAG Pipeline")
    print("=" * 60)

    # Create the pipeline configuration
    config, genome = create_customer_support_pipeline()
    print(f"\nCreated pipeline: {config.id} v{config.version}")
    print(f"Roles: {[r.id for r in config.roles]}")

    # Load sample documents
    docs_dir = Path(__file__).parent / "docs"
    documents = load_sample_documents(docs_dir)
    print(f"Loaded {len(documents)} documents")

    # Create a task using the builder
    t = task("How do I reset my password?", expected="Go to Settings > Security > Reset Password")
    print(f"\nQuery: {t.input['query']}")

    # Show the configuration
    print("\n" + "-" * 40)
    print("Pipeline Configuration:")
    print("-" * 40)
    for r in config.roles:
        prompt = genome.rolePrompts.get(r.promptRef)
        print(f"\nRole: {r.id}")
        print(f"  Model: {r.model}")
        print(f"  Tools: {r.tools or 'None'}")
        if prompt:
            preview = prompt.content[:100] + "..." if len(prompt.content) > 100 else prompt.content
            print(f"  Prompt: {preview}")

    print("\n" + "-" * 40)
    print("Graph (agent flow):")
    print("-" * 40)
    for e in config.graph:
        print(f"  {e.from_} -> {e.to}")

    print("\n✓ Pipeline configuration complete!")
    print("\nTo execute this pipeline with a real LLM:")
    print("  1. Set OPENAI_API_KEY environment variable")
    print("  2. Use siare.services.ExecutionEngine")
    print("  3. See the documentation for full examples")


async def run_evolution_example() -> None:
    """Demonstrate the evolution process (simulation)."""
    print("\n" + "=" * 60)
    print("SIARE Evolution Demo")
    print("=" * 60)

    config, _ = create_customer_support_pipeline()

    print("\nSimulating 5 generations of evolution...")
    print("(In production, this uses real LLM calls and evaluation)")
    print()

    # Simulate evolution generations
    fitness_scores = [0.65, 0.72, 0.78, 0.81, 0.85]
    mutation_types = [
        "PROMPT_CHANGE (retriever)",
        "PARAM_TWEAK (temperature)",
        "PROMPT_CHANGE (answerer)",
        "PROMPT_CHANGE (retriever)",
        "PROMPT_CHANGE (answerer)",
    ]

    for gen, (fitness, mutation) in enumerate(zip(fitness_scores, mutation_types), 1):
        improvement = ((fitness - fitness_scores[0]) / fitness_scores[0]) * 100 if gen > 1 else 0
        print(f"Generation {gen}/5: fitness={fitness:.2f}", end="")
        if improvement > 0:
            print(f" (+{improvement:.1f}%)", end="")
        print(f" | mutation: {mutation}")

    print()
    print("✓ Evolution complete!")
    print(f"  Starting fitness: {fitness_scores[0]:.2f}")
    print(f"  Final fitness: {fitness_scores[-1]:.2f}")
    print(f"  Improvement: +{((fitness_scores[-1] - fitness_scores[0]) / fitness_scores[0] * 100):.1f}%")


async def main() -> None:
    """Main entry point."""
    await run_basic_example()
    await run_evolution_example()

    print("\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
1. Set up your LLM provider:
   export OPENAI_API_KEY="your-key"

2. Try the programmatic API:
   from siare.services import ExecutionEngine, DirectorService

3. Read the architecture docs:
   https://github.com/synaptiai/siare/blob/main/docs/architecture.md

4. Join the community:
   https://github.com/synaptiai/siare/discussions
""")


if __name__ == "__main__":
    asyncio.run(main())
