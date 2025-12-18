# SIARE - Self-Improving Agentic RAG Engine

[![PyPI version](https://img.shields.io/pypi/v/siare.svg)](https://pypi.org/project/siare/)
[![PyPI downloads](https://img.shields.io/pypi/dm/siare.svg)](https://pypi.org/project/siare/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/siare.svg)](https://pypi.org/project/siare/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![CI](https://github.com/synaptiai/siare/actions/workflows/ci.yml/badge.svg)](https://github.com/synaptiai/siare/actions/workflows/ci.yml)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://synaptiai.github.io/siare/)

**Stop tuning RAG pipelines. Let them evolve.**

SIARE is the first self-improving RAG engine that treats pipeline configuration as evolvable genetic material. Instead of manually tuning prompts, retrieval strategies, and agent topologies, SIARE uses **Quality-Diversity optimization** to automatically discover and maintain diverse high-performing multi-agent RAG strategies.

## The Problem

Building RAG systems today means endless iteration:
- Tweak prompts → benchmark → repeat
- Try different chunking strategies → benchmark → repeat
- Adjust retrieval parameters → benchmark → repeat
- Add more agents → debug interactions → repeat

This process is **expensive**, **brittle**, and **never-ending** as your data and requirements change.

## The Solution

SIARE treats your RAG pipeline as a **living system that evolves**:

1. **Define your goals** - accuracy, latency, cost, or custom metrics
2. **Let evolution work** - SIARE mutates prompts, tools, and even agent topology
3. **Get diverse solutions** - Quality-Diversity ensures you have multiple good options
4. **Adapt continuously** - As your data changes, your pipeline evolves

## Quick Start

```bash
pip install siare

# Initialize a new project
siare init

# Run evolution (10 generations)
siare evolve

# Query your evolved pipeline
siare run "How do I reset my password?"
```

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                      SIARE Evolution Loop                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Execute │───▶│ Evaluate │───▶│ Diagnose │───▶│  Mutate  │  │
│  │   SOP    │    │  Metrics │    │ Weakness │    │   SOP    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲                                                │         │
│       │                                                │         │
│       └────────────────────────────────────────────────┘         │
│                                                                  │
│  Mutation Types:                                                 │
│  • PROMPT_CHANGE  - Evolve prompts based on failure patterns    │
│  • PARAM_TWEAK    - Adjust model parameters (temp, tokens)      │
│  • ADD_ROLE       - Add new agents to the pipeline              │
│  • REMOVE_ROLE    - Simplify by removing agents                 │
│  • REWIRE_GRAPH   - Change how agents connect and communicate   │
│  • CROSSOVER      - Combine successful strategies               │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### 🧬 Topology Evolution
Unlike prompt-only optimizers, SIARE can evolve the **structure** of your pipeline - adding, removing, and rewiring agents to find optimal architectures.

### 🎯 Quality-Diversity Optimization
Using MAP-Elites, SIARE maintains a diverse population of high-performing solutions. You don't just get one answer - you get a range of options trading off different metrics.

### 🔌 Extensible Hooks
Enterprise features integrate cleanly via hooks without modifying core code:
- Audit logging
- Usage billing
- Approval gates
- Custom metrics

### 📊 Built-in Benchmarks
Evaluate your pipelines against standard datasets:
- HotpotQA (multi-hop reasoning)
- Natural Questions
- Custom evaluation suites

## Installation

```bash
# Core package
pip install siare

# With LLM providers
pip install siare[llm]

# With embeddings support
pip install siare[embeddings]

# Everything
pip install siare[full]
```

## Programmatic Usage

```python
from siare import ProcessConfig, Role, GraphEdge
from siare.services import DirectorService, ExecutionEngine, GenePool

# Define your pipeline
config = ProcessConfig(
    name="customer-support",
    version="1.0.0",
    roles=[
        Role(
            name="retriever",
            model="gpt-4o-mini",
            system_prompt="Find relevant documents...",
            tools=["vector_search"],
        ),
        Role(
            name="answerer",
            model="gpt-4o-mini",
            system_prompt="Answer based on retrieved context...",
        ),
    ],
    graph=[
        GraphEdge(source="retriever", target="answerer"),
    ],
)

# Run evolution
director = DirectorService(llm_provider)
gene_pool = GenePool()

for generation in range(10):
    # Execute and evaluate
    trace = await engine.execute(config, task)
    evaluation = await evaluator.evaluate(trace)

    # Diagnose and mutate
    diagnosis = await director.diagnose(evaluation)
    mutated = await director.mutate_sop(config, diagnosis)

    # Track in gene pool
    gene_pool.add(mutated, evaluation)

# Get best solution
best = gene_pool.get_pareto_frontier()[0]
```

## Why SIARE?

| Feature | SIARE | DSPy | AutoRAG | LangChain |
|---------|-------|------|---------|-----------|
| Prompt optimization | ✅ | ✅ | ✅ | ❌ |
| Parameter tuning | ✅ | ❌ | ✅ | ❌ |
| **Topology evolution** | ✅ | ❌ | ❌ | ❌ |
| **Quality-Diversity** | ✅ | ❌ | ❌ | ❌ |
| Multi-agent support | ✅ | Limited | ❌ | ✅ |
| Extensible hooks | ✅ | ❌ | ❌ | ✅ |

## Examples

- [Customer Support](examples/quickstart/) - Simple Q&A over documents
- [Clinical Trials](examples/clinical_trials/) - Complex multi-agent research assistant

## Documentation

- [Quickstart Guide](https://synaptiai.github.io/siare/QUICKSTART.html)
- [Configuration](https://synaptiai.github.io/siare/CONFIGURATION.html)
- [System Architecture](https://synaptiai.github.io/siare/architecture/SYSTEM_ARCHITECTURE.html)
- [Data Models](https://synaptiai.github.io/siare/architecture/DATA_MODELS.html)
- [Guides](https://synaptiai.github.io/siare/guides/)

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) for details.

---

Built with ❤️ by [Synapti.ai](https://synapti.ai)
