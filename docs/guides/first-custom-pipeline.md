---
layout: default
title: First Custom Pipeline
parent: Guides
nav_order: 1
---

# Your First Custom Pipeline

This guide walks you through building a domain-specific RAG pipeline from scratch. By the end, you'll have a working pipeline tailored to your data and use case.

**Prerequisites:**
- Completed [Quick Start](../QUICKSTART.md)
- Understanding of your domain requirements
- Sample queries and expected answers for testing

**Time:** 30-45 minutes

---

## Overview

Building a custom pipeline involves five steps:

1. **Define your domain** — What problem are you solving?
2. **Design your agents** — What roles do you need?
3. **Write your prompts** — How should each agent behave?
4. **Configure evaluation** — How do you measure success?
5. **Enable evolution** — Let SIARE optimize automatically

---

## Step 1: Define Your Domain

Before writing any code, answer these questions:

| Question | Example Answer |
|----------|----------------|
| What data sources do you have? | PDF documents, SQL database, API endpoints |
| What questions will users ask? | "Find clinical trials for patients with diabetes" |
| What makes a good answer? | Accurate, cited, comprehensive, concise |
| What constraints exist? | Max 2 seconds latency, $0.05 per query budget |

### Example: Legal Document QA

For this guide, we'll build a legal document Q&A system:

- **Data**: Contract documents (PDFs converted to text)
- **Queries**: "What are the termination clauses in contract X?"
- **Quality**: Accurate clause extraction with page citations
- **Constraints**: Must flag if answer is uncertain

---

## Step 2: Design Your Agents

Start simple and add complexity as needed. A good starting point:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Retriever  │ ──▶ │   Analyst   │ ──▶ │  Responder  │
└─────────────┘     └─────────────┘     └─────────────┘
```

### Agent Roles for Legal QA

| Agent | Responsibility | Why Needed |
|-------|----------------|------------|
| **Retriever** | Find relevant contract sections | Narrows search space |
| **Analyst** | Extract specific clauses | Domain expertise |
| **Responder** | Format answer with citations | User-facing output |

### Define in Code

```python
from typing import List, Optional
from siare.core.models import (
    ProcessConfig,
    RoleConfig,
    GraphEdge,
    RoleInput,
    PromptGenome,
    RolePrompt,
    PromptConstraints,
)


def create_legal_qa_pipeline(model: str = "gpt-4o-mini") -> ProcessConfig:
    """Create a 3-agent legal document QA pipeline.

    Args:
        model: LLM model to use for all agents

    Returns:
        ProcessConfig defining the pipeline
    """
    return ProcessConfig(
        id="legal_qa_v1",
        version="1.0.0",
        models={model: model},
        tools=[],
        roles=[
            RoleConfig(
                id="retriever",
                model=model,
                tools=["vector_search"],  # Will use vector search adapter
                promptRef="legal_retriever_prompt",
                inputs=[RoleInput(from_="user_input")],
                outputs=["relevant_sections"],
            ),
            RoleConfig(
                id="analyst",
                model=model,
                tools=[],
                promptRef="legal_analyst_prompt",
                inputs=[
                    RoleInput(from_="user_input"),
                    RoleInput(from_="retriever"),
                ],
                outputs=["extracted_clauses", "confidence"],
            ),
            RoleConfig(
                id="responder",
                model=model,
                tools=[],
                promptRef="legal_responder_prompt",
                inputs=[
                    RoleInput(from_="user_input"),
                    RoleInput(from_="analyst"),
                ],
                outputs=["answer"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="retriever"),
            GraphEdge(from_="retriever", to="analyst"),
            GraphEdge(from_="analyst", to="responder"),
        ],
    )
```

---

## Step 3: Write Your Prompts

Prompts define how each agent behaves. Good prompts are:

- **Specific**: Clear about the task
- **Structured**: Define expected output format
- **Constrained**: Include guardrails for safety

### Create the PromptGenome

```python
def create_legal_qa_prompts() -> PromptGenome:
    """Create prompts for the legal QA pipeline.

    Returns:
        PromptGenome containing all role prompts
    """
    return PromptGenome(
        id="legal_qa_prompts_v1",
        version="1.0.0",
        rolePrompts={
            "legal_retriever_prompt": RolePrompt(
                id="legal_retriever_prompt",
                content="""You are a legal document retrieval specialist.

TASK: Find sections of contracts that are relevant to the user's question.

USER QUESTION: {query}
AVAILABLE DOCUMENTS: {documents}

INSTRUCTIONS:
1. Identify which documents are most relevant
2. Extract the specific sections that address the question
3. Include page numbers and section headers

OUTPUT FORMAT (JSON):
{
  "relevant_sections": [
    {
      "document": "Contract name",
      "section": "Section title",
      "page": 5,
      "content": "Exact text..."
    }
  ]
}""",
                constraints=PromptConstraints(
                    mustNotChange=["OUTPUT FORMAT (JSON):"],
                    allowedChanges=["formatting", "examples"],
                ),
            ),
            "legal_analyst_prompt": RolePrompt(
                id="legal_analyst_prompt",
                content="""You are a legal analyst specializing in contract law.

TASK: Analyze the retrieved sections and extract specific legal clauses.

USER QUESTION: {query}
RETRIEVED SECTIONS: {relevant_sections}

INSTRUCTIONS:
1. Identify clauses that directly answer the question
2. Note any ambiguities or missing information
3. Assess your confidence level (high/medium/low)

OUTPUT FORMAT (JSON):
{
  "extracted_clauses": [
    {
      "clause_type": "Termination",
      "text": "Either party may terminate...",
      "source": "Contract A, Section 5.2, Page 12"
    }
  ],
  "ambiguities": ["List any unclear points..."],
  "confidence": "high|medium|low"
}""",
            ),
            "legal_responder_prompt": RolePrompt(
                id="legal_responder_prompt",
                content="""You are a legal assistant providing clear, accurate answers.

TASK: Synthesize the analyst's findings into a user-friendly response.

USER QUESTION: {query}
ANALYST FINDINGS: {extracted_clauses}
CONFIDENCE LEVEL: {confidence}

INSTRUCTIONS:
1. Answer the question directly and concisely
2. Include citations to specific sections
3. If confidence is low, clearly state limitations
4. NEVER invent or assume information not in the findings

OUTPUT FORMAT:
[Your answer with inline citations like (Contract A, Section 5.2)]

CONFIDENCE DISCLAIMER (if applicable):
[State any limitations or uncertainties]""",
                constraints=PromptConstraints(
                    mustNotChange=["NEVER invent or assume information"],
                ),
            ),
        },
    )
```

### Prompt Design Tips

| Tip | Example |
|-----|---------|
| **Be explicit about format** | "Return JSON with keys: answer, confidence" |
| **Include safety guardrails** | "NEVER invent information not in documents" |
| **Define uncertainty handling** | "If unsure, say 'I cannot determine...'" |
| **Use constraints** | `mustNotChange` for critical instructions |

---

## Step 4: Configure Evaluation

Define how to measure pipeline quality. SIARE supports three metric types:

| Type | Description | Example |
|------|-------------|---------|
| **LLM Judge** | LLM evaluates output quality | Accuracy, relevance, completeness |
| **Programmatic** | Code-based checks | Contains citations, JSON valid |
| **Runtime** | Execution metrics | Latency, cost, token count |

### Define Your Metrics

```python
from typing import Dict, Any
from siare.core.models import (
    MetricConfig,
    MetricType,
    AggregationMethod,
    ExecutionTrace,
)


def define_legal_qa_metrics() -> list[MetricConfig]:
    """Define evaluation metrics for legal QA.

    Returns:
        List of MetricConfig for the pipeline
    """
    return [
        # LLM Judge: Is the answer accurate?
        MetricConfig(
            id="accuracy",
            type=MetricType.LLM_JUDGE,
            model="gpt-4o-mini",
            promptRef="accuracy_judge",
            inputs=["query", "answer", "ground_truth"],
            aggregationMethod=AggregationMethod.MEAN,
            weight=0.4,  # 40% of overall score
        ),
        # LLM Judge: Are citations correct?
        MetricConfig(
            id="citation_quality",
            type=MetricType.LLM_JUDGE,
            model="gpt-4o-mini",
            promptRef="citation_judge",
            inputs=["answer", "relevant_sections"],
            aggregationMethod=AggregationMethod.MEAN,
            weight=0.3,
        ),
        # Programmatic: Does it have citations?
        MetricConfig(
            id="has_citations",
            type=MetricType.PROGRAMMATIC,
            functionName="check_citations",
            aggregationMethod=AggregationMethod.MEAN,
            weight=0.1,
        ),
        # Runtime: Cost per query
        MetricConfig(
            id="cost",
            type=MetricType.RUNTIME,
            aggregationMethod=AggregationMethod.SUM,
            weight=0.1,
        ),
        # Runtime: Latency
        MetricConfig(
            id="latency",
            type=MetricType.RUNTIME,
            aggregationMethod=AggregationMethod.MEAN,
            weight=0.1,
        ),
    ]


def check_citations(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    """Programmatic metric: Check if answer contains citations.

    Args:
        trace: Execution trace with all outputs
        task_data: Original task data

    Returns:
        1.0 if citations present, 0.0 otherwise
    """
    import re
    answer = trace.outputs.get("answer", "")
    # Look for patterns like (Contract A, Section 5.2)
    has_citation = bool(re.search(r'\([^)]+,\s*[^)]+\)', answer))
    return 1.0 if has_citation else 0.0
```

### Create Judge Prompts

```python
def create_judge_prompts() -> Dict[str, str]:
    """Create prompts for LLM judge metrics.

    Returns:
        Dict mapping prompt IDs to prompt content
    """
    return {
        "accuracy_judge": """You are evaluating the accuracy of a legal QA answer.

QUESTION: {query}
EXPECTED ANSWER: {ground_truth}
ACTUAL ANSWER: {answer}

Rate the accuracy from 0.0 to 1.0:
- 1.0: Completely accurate, all key points covered
- 0.7-0.9: Mostly accurate, minor omissions
- 0.4-0.6: Partially accurate, some errors
- 0.1-0.3: Mostly inaccurate
- 0.0: Completely wrong or harmful

Return ONLY a JSON object: {"score": 0.85, "reason": "Brief explanation"}""",

        "citation_judge": """You are evaluating citation quality in a legal answer.

ANSWER: {answer}
SOURCE DOCUMENTS: {relevant_sections}

Check:
1. Are citations present?
2. Do citations point to real sections in the sources?
3. Are the cited sections relevant to the claims?

Rate from 0.0 to 1.0:
- 1.0: All claims properly cited with accurate references
- 0.5: Some citations, but incomplete or partially accurate
- 0.0: No citations or completely inaccurate citations

Return ONLY a JSON object: {"score": 0.85, "reason": "Brief explanation"}""",
    }
```

---

## Step 5: Enable Evolution

Now let SIARE automatically improve your pipeline.

### Create Task Set

Evolution needs example queries with expected answers:

```python
from siare.core.models import Task


def create_legal_qa_tasks() -> list[Task]:
    """Create evaluation task set for legal QA evolution.

    Returns:
        List of Tasks with queries and ground truth
    """
    return [
        Task(
            id="termination_1",
            input={
                "query": "What are the termination clauses in the ABC Corp contract?",
                "documents": "[Your document content here]",
            },
            ground_truth="Either party may terminate with 30 days written notice (Section 8.1). "
                         "Immediate termination allowed for material breach (Section 8.2).",
        ),
        Task(
            id="liability_1",
            input={
                "query": "What is the liability cap in the XYZ agreement?",
                "documents": "[Your document content here]",
            },
            ground_truth="Total liability is capped at $1,000,000 or the total fees paid, "
                         "whichever is less (Section 12.3).",
        ),
        # Add 10-50 more tasks for effective evolution
    ]
```

### Run Evolution

```python
from siare.services.scheduler import EvolutionScheduler
from siare.services.director import DirectorService
from siare.services.gene_pool import GenePool
from siare.services.execution_engine import ExecutionEngine
from siare.services.evaluation_service import EvaluationService
from siare.core.models import EvolutionJob


def run_legal_qa_evolution(llm_provider) -> None:
    """Run evolution to optimize the legal QA pipeline.

    Args:
        llm_provider: Configured LLM provider
    """
    # Initialize services
    gene_pool = GenePool()
    execution_engine = ExecutionEngine(llm_provider=llm_provider)
    evaluation_service = EvaluationService(llm_provider=llm_provider)
    director = DirectorService(llm_provider=llm_provider)

    # Register custom metric
    evaluation_service.register_metric_function("check_citations", check_citations)

    scheduler = EvolutionScheduler(
        gene_pool=gene_pool,
        director=director,
        execution_engine=execution_engine,
        evaluation_service=evaluation_service,
    )

    # Create and register the initial pipeline
    sop = create_legal_qa_pipeline()
    prompts = create_legal_qa_prompts()
    gene_pool.add_sop(sop, prompts)

    # Configure evolution job
    job = EvolutionJob(
        id="legal_qa_evolution",
        baseSopIds=["legal_qa_v1"],
        taskSet=create_legal_qa_tasks(),
        metricsToOptimize=define_legal_qa_metrics(),
        constraints={
            "maxCostPerTask": 0.10,      # Max $0.10 per query
            "maxLatencyMs": 5000,         # Max 5 seconds
            "minConfidenceThreshold": 0.7, # Minimum confidence
        },
        maxGenerations=20,    # Run 20 evolution cycles
        populationSize=5,     # Maintain 5 variants
    )

    # Run evolution
    print("Starting Legal QA pipeline evolution...")
    scheduler.run_evolution(job)

    # Get best solutions
    pareto_frontier = gene_pool.get_pareto_frontier(
        metrics=["accuracy", "cost"],
        domain="legal",
    )

    print(f"\nEvolution complete! Found {len(pareto_frontier)} optimal solutions.")
    for i, sop_gene in enumerate(pareto_frontier):
        print(f"\nSolution {i + 1}: {sop_gene.id} v{sop_gene.version}")
        print(f"  Accuracy: {sop_gene.metrics.get('accuracy', 0):.2%}")
        print(f"  Cost: ${sop_gene.metrics.get('cost', 0):.4f}")
        print(f"  Latency: {sop_gene.metrics.get('latency', 0):.0f}ms")
```

---

## Complete Example

Here's the full pipeline in one file:

```python
#!/usr/bin/env python3
"""Legal Document QA Pipeline - Complete Example

Run with:
    python legal_qa_pipeline.py --provider openai
    python legal_qa_pipeline.py --provider ollama --model llama3.2
"""

import argparse
from typing import Dict, Any, List

from siare.core.models import (
    ProcessConfig, RoleConfig, GraphEdge, RoleInput,
    PromptGenome, RolePrompt, PromptConstraints,
    MetricConfig, MetricType, AggregationMethod,
    ExecutionTrace, Task, EvolutionJob,
)
from siare.services.scheduler import EvolutionScheduler
from siare.services.director import DirectorService
from siare.services.gene_pool import GenePool
from siare.services.execution_engine import ExecutionEngine
from siare.services.evaluation_service import EvaluationService


# ... (Include all functions from above)


def main():
    parser = argparse.ArgumentParser(description="Legal QA Pipeline")
    parser.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model", default=None)
    parser.add_argument("--evolve", action="store_true", help="Run evolution")
    args = parser.parse_args()

    # Initialize provider (simplified)
    if args.provider == "openai":
        from siare.adapters.openai_provider import OpenAIProvider
        llm_provider = OpenAIProvider()
        model = args.model or "gpt-4o-mini"
    else:
        from siare.adapters.ollama_provider import OllamaProvider
        llm_provider = OllamaProvider()
        model = args.model or "llama3.2"

    # Create pipeline
    sop = create_legal_qa_pipeline(model)
    prompts = create_legal_qa_prompts()

    print(f"Created Legal QA pipeline with {len(sop.roles)} agents")
    print(f"Agents: {[r.id for r in sop.roles]}")

    if args.evolve:
        run_legal_qa_evolution(llm_provider)
    else:
        print("\nRun with --evolve to start evolution")


if __name__ == "__main__":
    main()
```

---

## Next Steps

| Task | Resource |
|------|----------|
| Add conditional routing | [Multi-Agent Patterns](../concepts/multi-agent-patterns.md) |
| Understand mutation types | [Mutation Operators](../reference/mutation-operators.md) |
| Write better prompts | [Prompt Engineering Guide](prompt-engineering.md) |
| Add vector search | [Custom Extensions](custom-extensions.md) |
| Deploy to production | [Deployment Guide](../DEPLOYMENT.md) |

---

## Troubleshooting

### Pipeline doesn't improve during evolution

**Causes:**
1. Task set too small (need 20+ diverse examples)
2. Metrics not sensitive enough
3. Constraints too restrictive

**Solutions:**
- Add more diverse test cases
- Adjust metric weights
- Relax constraints temporarily to explore

### Agent outputs are inconsistent

**Causes:**
1. Prompts too vague
2. No output format specified
3. Temperature too high

**Solutions:**
- Add explicit output format in prompts
- Use JSON format for structured outputs
- Lower temperature in model config

### Evolution is slow

**Causes:**
1. Large task set
2. Expensive model (GPT-4)
3. Too many generations

**Solutions:**
- Start with smaller task set (10-20)
- Use cheaper model for initial exploration
- Run fewer generations initially

---

*Questions? [Open an issue](https://github.com/synaptiai/siare/issues)*
