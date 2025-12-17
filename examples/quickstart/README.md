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
from siare import ProcessConfig, RoleConfig, RolePrompt, GraphEdge, PromptGenome, Task

# Define prompts for each role
genome = PromptGenome(
    id="customer_support_genome",
    version="1.0.0",
    rolePrompts={
        "retriever_prompt": RolePrompt(
            id="retriever_prompt",
            content="You are a document retrieval assistant...",
        ),
        "answerer_prompt": RolePrompt(
            id="answerer_prompt",
            content="You are a helpful customer support assistant...",
        ),
    },
)

# Define the pipeline configuration
config = ProcessConfig(
    id="customer-support-rag",
    version="1.0.0",
    models={"default": "gpt-4o-mini"},
    tools=["vector_search"],
    roles=[
        RoleConfig(
            id="retriever",
            model="gpt-4o-mini",
            tools=["vector_search"],
            promptRef="retriever_prompt",
        ),
        RoleConfig(
            id="answerer",
            model="gpt-4o-mini",
            tools=None,
            promptRef="answerer_prompt",
        ),
    ],
    graph=[
        GraphEdge(from_="user_input", to="retriever"),
        GraphEdge(from_="retriever", to="answerer"),
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
    api_key="your-api-key",  # or set OPENAI_API_KEY env var
)

# Create execution engine
engine = ExecutionEngine(llm_provider=llm)

# Define a task
task = Task(
    id="q1",
    input={"query": "How do I reset my password?"},
    groundTruth={"answer": "Go to Settings > Security > Reset Password"},
)

# Execute
trace = await engine.execute(config, genome, task)
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
    # Execute current config
    trace = await engine.execute(config, genome, task)

    # Evaluate performance
    evaluation = await evaluator.evaluate(trace, task)
    print(f"Gen {generation}: accuracy={evaluation.metrics.get('accuracy', 0):.2f}")

    # Diagnose weaknesses
    diagnosis = await director.diagnose(evaluation)

    # Mutate to improve
    config, genome = await director.mutate_sop(config, genome, diagnosis)

    # Track in gene pool
    gene_pool.add(config, evaluation)

# Get best performing config
best = gene_pool.get_pareto_frontier()[0]
print(f"Best config: {best.sop.id} v{best.sop.version}")
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
