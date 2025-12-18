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
| [Installation](QUICKSTART.html#installation) | [System Architecture](architecture/SYSTEM_ARCHITECTURE.html) | [First Custom Pipeline](guides/first-custom-pipeline.html) |
| [Quickstart](QUICKSTART.html) | [Data Models](architecture/DATA_MODELS.html) | [Prompt Engineering](guides/prompt-engineering.html) |
| [Configuration](CONFIGURATION.html) | [Glossary](GLOSSARY.html) | [Custom Extensions](guides/custom-extensions.html) |

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

See [Quickstart](QUICKSTART.html) for detailed setup instructions.

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

See [Use Cases](guides/USE_CASES.html) for complete examples.

---

## Documentation Sections

### Getting Started
- [Quickstart Guide](QUICKSTART.html) - Installation and first pipeline
- [Configuration](CONFIGURATION.html) - All configuration options
- [Troubleshooting](TROUBLESHOOTING.html) - Common issues and solutions

### Architecture
- [System Architecture](architecture/SYSTEM_ARCHITECTURE.html) - How SIARE works
- [Data Models](architecture/DATA_MODELS.html) - Core data structures

### Guides
- [First Custom Pipeline](guides/first-custom-pipeline.html) - Build your first pipeline
- [Prompt Engineering](guides/prompt-engineering.html) - Optimize prompts
- [Custom Extensions](guides/custom-extensions.html) - Extend SIARE
- [Use Cases](guides/USE_CASES.html) - Real-world examples

### Reference
- [Glossary](GLOSSARY.html) - Key terms and concepts
- [Why SIARE?](WHY_SIARE.html) - Comparison with alternatives
- [Contributing](CONTRIBUTING.html) - How to contribute

---

## License

SIARE is open source under the [MIT License](https://github.com/synaptiai/siare/blob/main/LICENSE).
