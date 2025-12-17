#!/usr/bin/env python3
"""
SIARE Advanced Example: Clinical Trials Research Assistant

This example demonstrates a complex multi-agent RAG pipeline for biomedical research.

Features:
- Multi-hop reasoning across trial databases
- Conditional execution based on query type
- Agent topology that can evolve over time
- Quality-Diversity optimization for diverse strategies

Requirements:
    pip install siare[full]
    export OPENAI_API_KEY="your-key"

Usage:
    python main.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from siare import pipeline, role, edge


# Define prompts as constants for readability
ROUTER_PROMPT = """You are a clinical research query classifier.

Analyze the user's query and classify it into one of these categories:
- "search": Looking for specific trials (e.g., "Find trials for...")
- "analyze": Asking about findings or efficacy (e.g., "What are the results of...")
- "safety": Asking about adverse events or safety (e.g., "What are the risks of...")
- "compare": Comparing multiple treatments or trials
- "general": General questions about clinical research

Output format:
CATEGORY: [category]
KEYWORDS: [key medical terms]
INTENT: [brief description of what user wants]"""

TRIAL_SEARCHER_PROMPT = """You are a clinical trial database specialist.

Your job is to search and retrieve relevant clinical trial information.

When searching:
1. Identify key criteria (condition, phase, intervention, status)
2. Search for matching trials
3. Return structured trial information

Output format for each trial:
---
NCT ID: [identifier]
Title: [full title]
Phase: [phase]
Status: [recruiting/completed/etc]
Condition: [medical condition]
Intervention: [treatment being tested]
Summary: [2-3 sentence summary]
---"""

STUDY_ANALYZER_PROMPT = """You are a clinical study analyst specializing in efficacy analysis.

Extract and summarize key findings from clinical trial results:
1. Primary endpoints and outcomes
2. Statistical significance (p-values, confidence intervals)
3. Effect sizes and clinical relevance
4. Study limitations

Be precise with numbers and always indicate sample sizes.

Output format:
FINDINGS:
- [finding 1 with statistics]
- [finding 2 with statistics]

LIMITATIONS:
- [limitation 1]
- [limitation 2]

CONCLUSION: [1-2 sentence summary]"""

SAFETY_REVIEWER_PROMPT = """You are a clinical safety data specialist.

Analyze and summarize safety information from clinical trials:
1. Common adverse events (with frequencies)
2. Serious adverse events (SAEs)
3. Treatment discontinuations due to AEs
4. Safety signals and warnings

Always include:
- Frequency (% of patients)
- Severity grade when available
- Comparison to control/placebo

Output format:
COMMON AEs (>5%):
- [AE1]: [X%] vs [Y%] placebo
- [AE2]: [X%] vs [Y%] placebo

SERIOUS AEs:
- [SAE1]: [details]

SAFETY CONCLUSION: [summary]"""

SYNTHESIZER_PROMPT = """You are a clinical research synthesizer.

Your job is to combine findings from multiple sources into a coherent summary.

Guidelines:
1. Integrate information from all provided inputs
2. Resolve any contradictions by noting the source
3. Highlight consensus findings
4. Identify gaps in the evidence
5. Maintain scientific accuracy

Structure your synthesis:
1. OVERVIEW: What was examined
2. KEY FINDINGS: Most important results
3. SAFETY PROFILE: Summarized safety data
4. EVIDENCE QUALITY: Strength of evidence
5. CONCLUSIONS: Clinical implications"""

CITATION_WRITER_PROMPT = """You are a scientific citation specialist.

Add proper citations to the research summary:
1. Use consistent citation format (e.g., [Author et al., Year])
2. Link claims to specific studies
3. Add a references section at the end
4. Flag any unsupported claims

Output the summary with inline citations and a references list."""


def create_clinical_trials_pipeline():
    """Create a multi-agent clinical trials research pipeline.

    Architecture:
    - Query Router: Classifies query type (search, analyze, safety)
    - Trial Searcher: Finds relevant clinical trials
    - Study Analyzer: Extracts key findings from studies
    - Safety Reviewer: Focuses on adverse events and safety data
    - Synthesizer: Combines findings into coherent summary
    - Citation Writer: Adds proper academic citations
    """
    return pipeline(
        "clinical-trials-assistant",
        roles=[
            role("router", "gpt-4o-mini", ROUTER_PROMPT),
            role("trial_searcher", "gpt-4o-mini", TRIAL_SEARCHER_PROMPT, tools=["vector_search", "clinical_trials_api"]),
            role("study_analyzer", "gpt-4o-mini", STUDY_ANALYZER_PROMPT),
            role("safety_reviewer", "gpt-4o-mini", SAFETY_REVIEWER_PROMPT),
            role("synthesizer", "gpt-4o", SYNTHESIZER_PROMPT),  # Stronger model for synthesis
            role("citation_writer", "gpt-4o-mini", CITATION_WRITER_PROMPT),
        ],
        edges=[
            # Router connects to specialized agents with conditions
            edge("router", "trial_searcher", condition="'search' in output or 'compare' in output"),
            edge("router", "study_analyzer", condition="'analyze' in output or 'compare' in output"),
            edge("router", "safety_reviewer", condition="'safety' in output"),
            # All specialists feed into synthesizer
            edge("trial_searcher", "synthesizer"),
            edge("study_analyzer", "synthesizer"),
            edge("safety_reviewer", "synthesizer"),
            # Synthesizer to citation writer
            edge("synthesizer", "citation_writer"),
        ],
        description="Multi-agent pipeline for clinical trials research",
        entry_point="router",  # Explicit entry point
    )


def load_sample_trials() -> list[dict[str, Any]]:
    """Load sample clinical trial data."""
    sample_data_path = Path(__file__).parent / "sample_data" / "trials.json"

    if sample_data_path.exists():
        with open(sample_data_path) as f:
            return json.load(f)

    # Create sample data if it doesn't exist
    sample_trials = [
        {
            "nct_id": "NCT04123456",
            "title": "Phase 3 Study of Drug X in Advanced Breast Cancer",
            "phase": "Phase 3",
            "status": "Completed",
            "condition": "Breast Cancer",
            "intervention": "Drug X (oral, 100mg daily)",
            "enrollment": 450,
            "primary_outcome": "Progression-free survival",
            "results": {
                "pfs_months": 12.3,
                "pfs_control": 8.1,
                "hazard_ratio": 0.65,
                "p_value": 0.002,
            },
            "adverse_events": [
                {"name": "Fatigue", "rate": 0.32, "control_rate": 0.18},
                {"name": "Nausea", "rate": 0.24, "control_rate": 0.12},
                {"name": "Neutropenia", "rate": 0.18, "control_rate": 0.04},
            ],
        },
        {
            "nct_id": "NCT04234567",
            "title": "Immunotherapy Combination in Triple-Negative Breast Cancer",
            "phase": "Phase 2",
            "status": "Active",
            "condition": "Triple-Negative Breast Cancer",
            "intervention": "Pembrolizumab + Chemotherapy",
            "enrollment": 180,
            "primary_outcome": "Objective response rate",
            "results": {
                "orr": 0.42,
                "orr_control": 0.28,
                "dcr": 0.68,
            },
            "adverse_events": [
                {"name": "Immune-related AEs", "rate": 0.35, "control_rate": 0.05},
                {"name": "Fatigue", "rate": 0.45, "control_rate": 0.38},
            ],
        },
        {
            "nct_id": "NCT04345678",
            "title": "CDK4/6 Inhibitor in HR+ Metastatic Breast Cancer",
            "phase": "Phase 3",
            "status": "Completed",
            "condition": "HR+ Metastatic Breast Cancer",
            "intervention": "Palbociclib + Letrozole",
            "enrollment": 666,
            "primary_outcome": "Progression-free survival",
            "results": {
                "pfs_months": 24.8,
                "pfs_control": 14.5,
                "hazard_ratio": 0.58,
                "p_value": 0.001,
            },
            "adverse_events": [
                {"name": "Neutropenia", "rate": 0.79, "control_rate": 0.06},
                {"name": "Leukopenia", "rate": 0.39, "control_rate": 0.02},
                {"name": "Fatigue", "rate": 0.37, "control_rate": 0.28},
            ],
        },
    ]

    # Save for future use
    sample_data_path.parent.mkdir(parents=True, exist_ok=True)
    with open(sample_data_path, "w") as f:
        json.dump(sample_trials, f, indent=2)

    return sample_trials


async def demonstrate_pipeline() -> None:
    """Demonstrate the clinical trials pipeline."""
    print("=" * 70)
    print("SIARE Advanced Example: Clinical Trials Research Assistant")
    print("=" * 70)

    # Create the pipeline
    config, genome = create_clinical_trials_pipeline()
    print(f"\nPipeline: {config.id} v{config.version}")
    print(f"Agents: {len(config.roles)}")
    print(f"Graph edges: {len(config.graph)}")

    # Show agent topology
    print("\n" + "-" * 40)
    print("Agent Topology:")
    print("-" * 40)
    for r in config.roles:
        tools_str = f" [tools: {', '.join(r.tools)}]" if r.tools else ""
        print(f"  • {r.id} ({r.model}){tools_str}")

    print("\nGraph Connections:")
    for e in config.graph:
        condition_str = f" [if: {e.condition}]" if e.condition else ""
        print(f"  {e.from_} → {e.to}{condition_str}")

    # Load sample data
    trials = load_sample_trials()
    print(f"\nLoaded {len(trials)} sample clinical trials")

    # Sample queries
    sample_queries = [
        "Find Phase 3 trials for breast cancer with CDK4/6 inhibitors",
        "What are the common adverse events in immunotherapy trials for breast cancer?",
        "Compare progression-free survival across recent breast cancer trials",
    ]

    print("\n" + "-" * 40)
    print("Sample Queries This Pipeline Handles:")
    print("-" * 40)
    for i, query in enumerate(sample_queries, 1):
        print(f"  {i}. {query}")

    print("\n" + "=" * 70)
    print("Evolution Demonstration (Simulated)")
    print("=" * 70)

    # Simulate evolution results
    print("\nInitial pipeline: 6 agents")
    print("After 20 generations of evolution...")
    print()

    evolution_results = [
        ("Generation 1", 0.72, "PROMPT_CHANGE (router)"),
        ("Generation 5", 0.78, "PARAM_TWEAK (synthesizer temp)"),
        ("Generation 10", 0.82, "REMOVE_ROLE (redundant reviewer)"),
        ("Generation 15", 0.86, "PROMPT_CHANGE (study_analyzer)"),
        ("Generation 20", 0.89, "REWIRE_GRAPH (parallel execution)"),
    ]

    for gen, fitness, mutation in evolution_results:
        print(f"  {gen}: accuracy={fitness:.2f} | {mutation}")

    print("\n" + "-" * 40)
    print("Final Optimized Pipeline:")
    print("-" * 40)
    print("  • Reduced from 6 to 4 agents (removed redundant safety reviewer)")
    print("  • Parallel execution of search and analyze agents")
    print("  • Optimized prompts for better grounding")
    print()
    print("Improvements:")
    print("  • Accuracy: 0.72 → 0.89 (+24%)")
    print("  • Latency: 8.2s → 4.1s (-50%)")
    print("  • Cost: $0.12 → $0.07 (-42%)")


async def main() -> None:
    """Main entry point."""
    await demonstrate_pipeline()

    print("\n" + "=" * 70)
    print("Try It Yourself")
    print("=" * 70)
    print("""
1. Set up your environment:
   export OPENAI_API_KEY="your-key"

2. Customize the pipeline in main.py

3. Run with real evolution:
   from siare.services import DirectorService, ExecutionEngine
   # See quickstart example for the evolution loop

4. Explore the QD grid:
   from siare.services import QDGridManager
   qd = QDGridManager(dimensions=["accuracy", "latency"])
   # Maintains diverse high-performing solutions

Learn more:
  https://github.com/synaptiai/siare/blob/main/docs/architecture.md
""")


if __name__ == "__main__":
    asyncio.run(main())
