---
layout: default
title: Home
nav_order: 1
description: "SIARE - Self-Improving Agentic RAG Engine"
permalink: /
---

# SIARE Documentation

**Self-Improving Agentic RAG Engine** - Evolve multi-agent RAG pipelines using Quality-Diversity optimization.

SIARE treats your RAG pipeline configuration as a searchable space and uses AI-driven evolution to discover optimal strategies for your specific domain and tasks.

---

## Quick Links

| Getting Started | Core Concepts | Guides |
|-----------------|---------------|--------|
| [Installation]({% link QUICKSTART.md %}#installation) | [System Architecture]({% link architecture/SYSTEM_ARCHITECTURE.md %}) | [First Custom Pipeline]({% link guides/first-custom-pipeline.md %}) |
| [Quickstart]({% link QUICKSTART.md %}) | [Data Models]({% link architecture/DATA_MODELS.md %}) | [Prompt Engineering]({% link guides/prompt-engineering.md %}) |
| [Configuration]({% link CONFIGURATION.md %}) | [Glossary]({% link GLOSSARY.md %}) | [Custom Extensions]({% link guides/custom-extensions.md %}) |

---

## What is SIARE?

Traditional RAG systems require manual tuning of retrieval strategies, prompt templates, and agent configurations. SIARE automates this through **evolutionary optimization**:

1. **Define** your multi-agent pipeline (roles, tools, graph structure)
2. **Provide** evaluation tasks and metrics
3. **Evolve** - SIARE mutates and evaluates configurations to find what works best
4. **Deploy** the optimal pipeline for your domain

### Key Features

- **Quality-Diversity Optimization** - MAP-Elites algorithm maintains diverse high-performing solutions
- **6 Mutation Types** - Prompt changes, parameter tweaks, topology rewiring, crossover
- **Multi-Agent DAGs** - Conditional execution paths with role-based specialization
- **Extensible Adapters** - Vector search, web search, custom tools
- **Hook System** - Observe and extend core behavior without modifying source

---

## Installation

```bash
pip install siare
```

For full features including LLM providers and embeddings:

```bash
pip install siare[full]
```

See [Quickstart]({% link QUICKSTART.md %}) for detailed setup instructions.

---

## Example

```python
from siare import pipeline, role, edge, task

# Define a simple RAG pipeline
config = pipeline(
    name="my-rag",
    roles=[
        role("retriever", "Find relevant documents", model="gpt-4o-mini"),
        role("synthesizer", "Generate answer from context", model="gpt-4o"),
    ],
    edges=[
        edge("retriever", "synthesizer"),
    ],
)

# Define evaluation tasks
tasks = [
    task("What is machine learning?", expected="supervised and unsupervised..."),
]

# Evolve the pipeline
from siare.services import DirectorService, ExecutionEngine

director = DirectorService(llm_provider)
engine = ExecutionEngine(llm_provider)

# Run evolution loop...
```

See [Use Cases]({% link guides/USE_CASES.md %}) for complete examples.

---

## Documentation Sections

### Getting Started
- [Quickstart Guide]({% link QUICKSTART.md %}) - Installation and first pipeline
- [Configuration]({% link CONFIGURATION.md %}) - All configuration options
- [Troubleshooting]({% link TROUBLESHOOTING.md %}) - Common issues and solutions

### Architecture
- [System Architecture]({% link architecture/SYSTEM_ARCHITECTURE.md %}) - How SIARE works
- [Data Models]({% link architecture/DATA_MODELS.md %}) - Core data structures

### Guides
- [First Custom Pipeline]({% link guides/first-custom-pipeline.md %}) - Build your first pipeline
- [Prompt Engineering]({% link guides/prompt-engineering.md %}) - Optimize prompts
- [Custom Extensions]({% link guides/custom-extensions.md %}) - Extend SIARE
- [Use Cases]({% link guides/USE_CASES.md %}) - Real-world examples

### Reference
- [Glossary]({% link GLOSSARY.md %}) - Key terms and concepts
- [Why SIARE?]({% link WHY_SIARE.md %}) - Comparison with alternatives
- [Contributing]({% link CONTRIBUTING.md %}) - How to contribute

---

## License

SIARE is open source under the [MIT License](https://github.com/synaptiai/siare/blob/main/LICENSE).
