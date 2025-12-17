# SIARE Quickstart: Customer Support RAG

This example demonstrates how to build a self-evolving customer support RAG system
that answers questions from your documentation.

## What You'll Build

A RAG pipeline that:
1. Retrieves relevant documents based on user questions
2. Generates accurate answers grounded in your docs
3. **Evolves to improve over time** using SIARE's Quality-Diversity optimization

## Prerequisites

```bash
pip install siare[full]
export OPENAI_API_KEY="your-key-here"  # or use Ollama for local inference
```

## Quick Start (5 minutes)

### 1. Create Your Pipeline

```python
from siare import pipeline, role, edge, task

# Create a 2-agent RAG pipeline in just a few lines
config, genome = pipeline(
    "customer-support-rag",
    roles=[
        role("retriever", "gpt-4o-mini", "You are a document retrieval specialist...", tools=["vector_search"]),
        role("answerer", "gpt-4o-mini", "You are a helpful customer support assistant..."),
    ],
    edges=[
        edge("retriever", "answerer"),
    ],
)
```

### 2. Execute the Pipeline

```python
from siare.services import ExecutionEngine, LLMProvider

# Initialize LLM provider
llm = LLMProvider(
    provider="openai",
    model="gpt-4o-mini",
)

# Create execution engine
engine = ExecutionEngine(llm_provider=llm)

# Define a task
t = task("How do I reset my password?", expected="Go to Settings > Security")

# Execute
trace = await engine.execute(config, genome, t)
print(trace.final_output)
```

### 3. Evolve Your Pipeline

```python
from siare.services import DirectorService, GenePool, EvaluationService

# Initialize services
director = DirectorService(llm_provider=llm)
gene_pool = GenePool()
evaluator = EvaluationService()

# Evolution loop
for generation in range(10):
    trace = await engine.execute(config, genome, t)
    evaluation = await evaluator.evaluate(trace, t)
    print(f"Gen {generation}: accuracy={evaluation.metrics.get('accuracy', 0):.2f}")

    diagnosis = await director.diagnose(evaluation)
    config, genome = await director.mutate_sop(config, genome, diagnosis)
    gene_pool.add(config, evaluation)

# Get best performing config
best = gene_pool.get_pareto_frontier()[0]
```

## File Structure

```
examples/quickstart/
├── README.md           # This file
├── main.py             # Complete working example
├── siare.yaml          # Sample configuration
└── docs/               # Sample documents
    ├── faq.md
    └── user-guide.md
```

## Running the Example

```bash
cd examples/quickstart
python main.py
```

## What Happens During Evolution

SIARE automatically:
1. **Diagnoses** why the current pipeline fails (e.g., "retriever misses relevant docs")
2. **Proposes mutations** to fix issues (e.g., adjust retrieval prompt)
3. **Evaluates** the mutated pipeline against your metrics
4. **Selects** the best-performing variants using Quality-Diversity

Mutation types include:
- `PROMPT_CHANGE` - Improve agent prompts
- `PARAM_TWEAK` - Adjust model parameters
- `ADD_ROLE` - Add new agents
- `REMOVE_ROLE` - Simplify the pipeline
- `REWIRE_GRAPH` - Change how agents communicate

## Next Steps

- See [main.py](main.py) for the complete working example
- Read the [Architecture Guide](../../docs/architecture.md) for deeper understanding
- Try the [Clinical Trials Example](../clinical_trials/) for advanced multi-agent patterns
