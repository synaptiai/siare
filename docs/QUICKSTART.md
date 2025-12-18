---
layout: default
title: Quickstart
nav_order: 2
---

# SIARE Quick Start Guide

Get a self-evolving RAG pipeline running in under 10 minutes.

**What you'll accomplish:**
1. Install SIARE and configure an LLM provider
2. Run a demo that shows multi-agent execution and evaluation
3. Understand the output and what comes next

## Prerequisites

Choose your path based on your setup:

### Option A: Local Development with Ollama (Free)

**Requirements:**
- Python 3.10 or higher
- Docker or Ollama installed locally
- 8GB+ RAM recommended
- No API keys required

**Setup:**
1. Install Ollama: https://ollama.ai/download
2. Pull a model: `ollama pull llama3.2`
3. Verify Ollama is running: `ollama list`

### Option B: Cloud LLM with OpenAI (API Key Required)

**Requirements:**
- Python 3.10 or higher
- OpenAI API key (https://platform.openai.com/api-keys)
- Active OpenAI account with credits

**Setup:**
1. Get your API key from OpenAI dashboard
2. Keep it ready for Step 2

---

## Quick Start

### Step 1: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/synaptiai/siare.git
cd siare

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Your Environment

#### For Option A (Ollama - Local):

```bash
# Start Ollama server (if not already running)
ollama serve

# In another terminal, pull the model
ollama pull llama3.2

# Verify the model is available
ollama list
# You should see llama3.2 in the list
```

#### For Option B (OpenAI - Cloud):

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="sk-your-api-key-here"

# On Windows PowerShell:
# $env:OPENAI_API_KEY="sk-your-api-key-here"

# Verify it's set
echo $OPENAI_API_KEY
```

### Step 3: Run the Demo

Execute the quickstart demo with your chosen provider:

```bash
# Option A: Using Ollama (local)
python -m siare.demos.agentic_rag_quickstart --provider ollama

# Option B: Using OpenAI (cloud)
python -m siare.demos.agentic_rag_quickstart --provider openai
```

The demo will run 3 iterations by default. You can customize this:

```bash
# Run 5 iterations
python -m siare.demos.agentic_rag_quickstart --provider ollama --iterations 5

# Run quietly (minimal output)
python -m siare.demos.agentic_rag_quickstart --provider openai --quiet
```

### Step 4: Understand the Output

When you run the demo, you'll see output like this:

```
============================================================
  SIARE Quickstart Demo
============================================================
Provider: ollama
Model: llama3.2
Iterations: 3

[1/4] Services initialized
[2/4] Created RAG pipeline with 2 roles

--- Iteration 1/3 ---
  Task 1: completed
  Task 2: completed
  Task 3: completed
  Iteration accuracy: 78.33%

--- Iteration 2/3 ---
  Task 1: completed
  Task 2: completed
  Task 3: completed
  Iteration accuracy: 81.67%

--- Iteration 3/3 ---
  Task 1: completed
  Task 2: completed
  Task 3: completed
  Iteration accuracy: 83.33%

[3/4] Completed 3 iterations
[4/4] Results:
  Initial score: 78.33%
  Final score: 83.33%
  Delta: +5.00%

============================================================
```

#### What Each Section Means:

1. **Services initialized**: LLM provider, execution engine, and evaluation service are ready
2. **Created RAG pipeline with 2 roles**:
   - `retriever`: Finds relevant documents
   - `answerer`: Generates answers based on retrieved documents
3. **Iteration results**: Each iteration runs 3 sample tasks and calculates accuracy
4. **Final results**: Shows performance across all iterations

**Important Note**: This demo runs the same pipeline multiple times to demonstrate execution and evaluation. For actual SOP evolution (where the system improves itself), use the `EvolutionScheduler` (see [Running Full Evolution](#run-full-evolution-self-improvement) below).

---

## What's Next?

You've successfully run a multi-agent RAG pipeline! Here's where to go based on your goal:

| I want to... | Do this |
|--------------|---------|
| Customize prompts and add agents | Continue to [Step 5: Customize Your Pipeline](#step-5-customize-your-pipeline) below |
| Run automatic evolution | Jump to [Run Full Evolution](#run-full-evolution-self-improvement) |
| Understand how evolution works | Read [Evolution Lifecycle](concepts/evolution-lifecycle.md) |
| Build a domain-specific pipeline | Follow [First Custom Pipeline](guides/first-custom-pipeline.md) |
| Deploy to production | See [Deployment Guide](DEPLOYMENT.md) |

---

## Step 5: Customize Your Pipeline

The demo uses a simple 2-role pipeline. Let's customize it step by step.

### 5.1 Add a New Agent (Ranker)

Add a document ranker between retriever and answerer:

```python
from typing import List
from siare.core.models import ProcessConfig, RoleConfig, GraphEdge, RoleInput


def create_three_stage_rag(model: str) -> ProcessConfig:
    """Create a 3-stage RAG pipeline: retriever → ranker → answerer.

    Args:
        model: Model identifier (e.g., "llama3.2" for Ollama, "gpt-4o-mini" for OpenAI)

    Returns:
        ProcessConfig defining the multi-agent pipeline
    """
    return ProcessConfig(
        id="three_stage_rag",
        version="1.0.0",
        models={model: model},
        tools=[],
        roles=[
            RoleConfig(
                id="retriever",
                model=model,
                tools=[],
                promptRef="retriever_prompt",
                inputs=[RoleInput(from_="user_input")],
                outputs=["documents"],
            ),
            RoleConfig(
                id="ranker",  # New role: re-ranks retrieved documents
                model=model,
                tools=[],
                promptRef="ranker_prompt",
                inputs=[RoleInput(from_=["user_input", "retriever"])],
                outputs=["ranked_docs"],
            ),
            RoleConfig(
                id="answerer",
                model=model,
                tools=[],
                promptRef="answerer_prompt",
                inputs=[RoleInput(from_=["user_input", "ranker"])],
                outputs=["answer"],
            ),
        ],
        graph=[
            GraphEdge(from_="user_input", to="retriever"),
            GraphEdge(from_=["user_input", "retriever"], to="ranker"),
            GraphEdge(from_=["user_input", "ranker"], to="answerer"),
        ],
    )


# Usage
sop = create_three_stage_rag("llama3.2")
print(f"Pipeline has {len(sop.roles)} roles: {[r.id for r in sop.roles]}")
```

**What changed:** We added a `ranker` role that sits between retriever and answerer. The ranker receives both the original query and retrieved documents, then outputs re-ranked documents for the answerer.

### 5.2 Customize Agent Prompts

Each role references a prompt by ID. Create the prompts that define agent behavior:

```python
from siare.core.models import PromptGenome, RolePrompt


def create_three_stage_prompts() -> PromptGenome:
    """Create prompts for the three-stage RAG pipeline.

    Returns:
        PromptGenome containing all role prompts
    """
    return PromptGenome(
        id="three_stage_genome",
        version="1.0.0",
        rolePrompts={
            "retriever_prompt": RolePrompt(
                id="retriever_prompt",
                content="""You are an expert document retrieval agent.
Given a query, find and return the TOP 5 most relevant passages.

Query: {query}

Focus on semantic relevance and recency.
Return as JSON: {"documents": [...]}""",
            ),
            "ranker_prompt": RolePrompt(
                id="ranker_prompt",
                content="""You are a document relevance ranker.
Re-rank the following documents by relevance to the query.

Query: {query}
Documents: {documents}

Return the TOP 3 documents in order of relevance.
Return as JSON: {"ranked_docs": [...]}""",
            ),
            "answerer_prompt": RolePrompt(
                id="answerer_prompt",
                content="""You are a precise question-answering agent.
Use ONLY the provided documents to answer. If unsure, say so.

Question: {query}
Documents: {ranked_docs}

Provide a concise, evidence-based answer with citations.""",
            ),
        },
    )


# Usage
genome = create_three_stage_prompts()
print(f"Prompts for: {list(genome.rolePrompts.keys())}")
```

**What changed:** We added a `ranker_prompt` and updated `answerer_prompt` to use `{ranked_docs}` instead of `{documents}`.

### 5.3 Add a Custom Evaluation Metric

Define domain-specific metrics to evaluate your pipeline:

```python
from typing import Dict, List, Any
from siare.services.evaluation_service import EvaluationService
from siare.core.models import ExecutionTrace


def term_coverage_metric(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    """Check if the answer contains required domain terms.

    Args:
        trace: Execution trace containing role outputs
        task_data: Task data including required_terms list

    Returns:
        Score between 0.0 and 1.0 indicating term coverage
    """
    answer: str = trace.outputs.get("answer", "")
    required_terms: List[str] = task_data.get("required_terms", [])

    if not required_terms:
        return 1.0  # No terms required = perfect score

    matches = sum(1 for term in required_terms if term.lower() in answer.lower())
    return matches / len(required_terms)


def citation_metric(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    """Check if the answer includes source citations.

    Args:
        trace: Execution trace containing role outputs
        task_data: Task data (unused but required by interface)

    Returns:
        1.0 if citations present, 0.0 otherwise
    """
    answer: str = trace.outputs.get("answer", "")
    # Simple heuristic: check for citation patterns like [1], [Source], etc.
    import re
    has_citations = bool(re.search(r'\[[\w\d]+\]', answer))
    return 1.0 if has_citations else 0.0


# Register custom metrics with the evaluation service
# evaluation_service = EvaluationService(llm_provider=your_provider)
# evaluation_service.register_metric_function("term_coverage", term_coverage_metric)
# evaluation_service.register_metric_function("has_citations", citation_metric)
```

**What changed:** We defined two custom metrics. `term_coverage_metric` checks if required domain terms appear in answers. `citation_metric` verifies that answers include source citations.

---

## Run Full Evolution (Self-Improvement)

This is where SIARE's power shines: automatic improvement of your pipeline through evolutionary optimization.

```python
from typing import List
from siare.services.scheduler import EvolutionScheduler
from siare.services.director import DirectorService
from siare.services.gene_pool import GenePool
from siare.services.execution_engine import ExecutionEngine
from siare.services.evaluation_service import EvaluationService
from siare.core.models import (
    EvolutionJob,
    MetricConfig,
    AggregationMethod,
    MetricType,
    Task,
    SOPGene,
)


def run_evolution_example(llm_provider) -> List[SOPGene]:
    """Run an evolution job to automatically improve a RAG pipeline.

    Args:
        llm_provider: Configured LLM provider (OpenAI or Ollama)

    Returns:
        List of Pareto-optimal SOPs discovered during evolution
    """
    # Initialize core services
    gene_pool = GenePool()
    execution_engine = ExecutionEngine(llm_provider=llm_provider)
    evaluation_service = EvaluationService(llm_provider=llm_provider)
    director = DirectorService(llm_provider=llm_provider)

    scheduler = EvolutionScheduler(
        gene_pool=gene_pool,
        director=director,
        execution_engine=execution_engine,
        evaluation_service=evaluation_service,
    )

    # Define your task set (questions with ground truth answers)
    task_set: List[Task] = [
        Task(id="q1", input={"query": "What is SIARE?"}, groundTruth={"answer": "Self-Improving Agentic RAG Engine"}),
        Task(id="q2", input={"query": "How does evolution work?"}, groundTruth={"answer": "Mutation and selection"}),
        # Add more tasks for better evolution...
    ]

    # Configure the evolution job
    job = EvolutionJob(
        id="rag_evolution_001",
        baseSopIds=["three_stage_rag"],  # Start from our 3-stage pipeline
        taskSet=task_set,
        metricsToOptimize=[
            MetricConfig(
                id="accuracy",
                type=MetricType.LLM_JUDGE,
                model="gpt-4o-mini",
                promptRef="accuracy_judge",
                inputs=["query", "answer", "groundTruth"],
                aggregationMethod=AggregationMethod.MEAN,
            ),
            MetricConfig(
                id="cost",
                type=MetricType.RUNTIME,
                aggregationMethod=AggregationMethod.SUM,
            ),
        ],
        constraints={
            "maxCostPerTask": 0.10,  # Max $0.10 per task
            "minSafetyScore": 0.90,  # Minimum 90% safety
        },
        maxGenerations=10,  # Run 10 evolution cycles
        populationSize=5,   # Maintain 5 SOP variants
    )

    # Run evolution (this may take several minutes)
    print("Starting evolution...")
    scheduler.run_evolution(job)
    print("Evolution complete!")

    # Get the Pareto-optimal solutions (best trade-offs between accuracy and cost)
    pareto_frontier: List[SOPGene] = gene_pool.get_pareto_frontier(
        metrics=["accuracy", "cost"],
        domain="rag",
    )

    # Display results
    print(f"\nFound {len(pareto_frontier)} Pareto-optimal solutions:")
    for sop in pareto_frontier:
        print(f"  SOP {sop.id} v{sop.version}")
        print(f"    Accuracy: {sop.metrics.get('accuracy', 0):.2%}")
        print(f"    Cost: ${sop.metrics.get('cost', 0):.4f}")

    return pareto_frontier
```

**What happens during evolution:**

1. **Execute**: Each SOP variant runs on your task set
2. **Evaluate**: Metrics are computed (accuracy, cost, latency, etc.)
3. **Diagnose**: AI Director analyzes failures and identifies weaknesses
4. **Mutate**: Director proposes improvements (better prompts, new agents, rewired graphs)
5. **Select**: Best solutions are kept, forming the next generation

The result is a **Pareto frontier** — a set of solutions that represent optimal trade-offs. You can then choose the SOP that best fits your needs (highest accuracy, lowest cost, or balanced).

---

## Troubleshooting

### Ollama Issues

**Problem**: `RuntimeError: Ollama not running at http://localhost:11434`

**Solution**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not running, start it
ollama serve

# Verify the model is pulled
ollama list
# If llama3.2 is missing:
ollama pull llama3.2
```

**Problem**: Slow performance with Ollama

**Solution**:
- Ensure you have at least 8GB RAM available
- Try a smaller model: `ollama pull llama3.2:1b`
- Use the smaller model: `--model llama3.2:1b`

### OpenAI Issues

**Problem**: `RuntimeError: OPENAI_API_KEY environment variable not set`

**Solution**:
```bash
# Set the key
export OPENAI_API_KEY="sk-your-key-here"

# Verify it's set
echo $OPENAI_API_KEY
```

**Problem**: `AuthenticationError` or `401 Unauthorized`

**Solution**:
- Verify your API key is valid at https://platform.openai.com/api-keys
- Check that your OpenAI account has available credits
- Ensure the key is properly exported (no quotes issues, no spaces)

### General Issues

**Problem**: `ModuleNotFoundError: No module named 'siare'`

**Solution**:
```bash
# Make sure you're in the project directory
pwd  # Should show .../siare

# Ensure dependencies are installed
pip install -r requirements.txt

# Install in development mode (optional)
pip install -e .
```

**Problem**: Tests failing or import errors

**Solution**:
```bash
# Verify Python version (must be 3.10+)
python --version

# Reinstall dependencies in a fresh venv
deactivate  # if in a venv
rm -rf venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Learn More

### Key Concepts

| Concept | Description |
|---------|-------------|
| **ProcessConfig (SOP)** | Defines your multi-agent pipeline structure |
| **PromptGenome** | Contains all prompts used by agents |
| **ExecutionEngine** | Runs your pipeline as a directed acyclic graph (DAG) |
| **EvaluationService** | Measures pipeline performance across multiple metrics |
| **Director** | AI brain that diagnoses issues and proposes improvements |
| **GenePool** | Stores all SOP versions with their performance history |

### Example Use Cases

| Domain | What SIARE Does |
|--------|-----------------|
| **Customer Support RAG** | Automatically improve answer quality and reduce costs |
| **Legal Document Analysis** | Evolve pipelines for better compliance detection |
| **Research Paper Search** | Optimize retrieval and summarization strategies |
| **Clinical Trials** | Match patients to trials with evolving criteria |

### Documentation

| Topic | Link |
|-------|------|
| System Architecture | [SYSTEM_ARCHITECTURE.md](architecture/SYSTEM_ARCHITECTURE.md) |
| Data Models | [DATA_MODELS.md](architecture/DATA_MODELS.md) |
| Configuration Reference | [CONFIGURATION.md](CONFIGURATION.md) |
| Deployment Guide | [DEPLOYMENT.md](DEPLOYMENT.md) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## Summary: What You've Learned

You've successfully:

1. ✅ Installed SIARE and configured an LLM provider (Ollama or OpenAI)
2. ✅ Run a multi-agent RAG pipeline
3. ✅ Understood how to add agents, customize prompts, and add metrics
4. ✅ Seen how to run automatic evolution for self-improvement

### Continue Your Journey

| Next Step | Guide |
|-----------|-------|
| Build a domain-specific pipeline | [First Custom Pipeline](guides/first-custom-pipeline.md) |
| Understand evolution deeply | [Evolution Lifecycle](concepts/evolution-lifecycle.md) |
| Learn multi-agent design patterns | [Multi-Agent Patterns](concepts/multi-agent-patterns.md) |
| Add custom metrics and tools | [Custom Extensions](guides/custom-extensions.md) |
| Deploy to production | [Deployment Guide](DEPLOYMENT.md) |

---

*Questions? [Open an issue](https://github.com/synaptiai/siare/issues) or [start a discussion](https://github.com/synaptiai/siare/discussions).*
