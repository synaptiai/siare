# Benchmarks Guide

SIARE includes a comprehensive benchmarking suite for measuring evolution effectiveness and comparing variation strategies.

## Overview

The benchmark system has three tiers:

| Tier | Suite | Purpose | Datasets |
|------|-------|---------|----------|
| 1 | Self-Improvement | Core value demonstration (Gen 0 vs Gen N) | BEIR, Natural Questions |
| 2 | Quality Gate | Pre-production validation with statistical rigor | HotpotQA |
| 3 | Publication | Academic-grade with ablation studies | FRAMES |

## Quick Start

### Self-Improvement Benchmark

The flagship benchmark. Shows that SIARE improves RAG performance by evolving prompts and topology while keeping the model constant.

```bash
# Quick test (3 generations, 20 samples — minutes)
python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
    --provider ollama --model llama3.2:1b \
    --reasoning-model llama3.1:8b --quick

# Full run (10 generations, 50 samples)
python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
    --provider openai --model gpt-4o-mini \
    --reasoning-model gpt-4o --generations 10 --samples 50
```

### Agentic Comparison Benchmark

Compares all 3 variation modes on the same dataset to measure the impact of the hybrid agentic evolution feature.

```bash
# Quick comparison
python -m siare.benchmarks.scripts.run_agentic_comparison \
    --provider ollama --model llama3.2:1b \
    --reasoning-model llama3.1:8b --quick

# Full comparison with specific modes
python -m siare.benchmarks.scripts.run_agentic_comparison \
    --provider openai --model gpt-4o-mini \
    --reasoning-model gpt-4o \
    --modes single_turn,adaptive --generations 10
```

## Variation Modes

The benchmark suite supports three variation modes that control how the evolution loop generates mutations:

| Mode | Description | Cost | Best For |
|------|-------------|------|----------|
| `single_turn` | Classic Diagnostician + Architect (2 LLM calls per mutation) | Low | Fast iterations, tight budgets |
| `agentic` | Multi-turn AgenticDirector with tools (diagnose, propose, validate, iterate) | Higher | Deep optimization, complex pipelines |
| `adaptive` | Starts single_turn, escalates to agentic when stagnation detected | Variable | **Recommended** — best of both |

Use the `--variation-mode` flag to select:

```bash
python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
    --provider ollama --model llama3.2:1b \
    --reasoning-model llama3.1:8b \
    --variation-mode adaptive --quick
```

### Agentic-Specific Options

When using `agentic` or `adaptive` mode:

| Flag | Default | Description |
|------|---------|-------------|
| `--variation-mode` | None (single_turn) | `single_turn`, `agentic`, or `adaptive` |
| `--max-inner-iterations` | 5 | Max diagnose-propose-validate cycles per mutation |
| `--agent-model` | same as `--reasoning-model` | LLM model for the AgenticDirector and Supervisor |

## Topology Evolution

By default, benchmarks only evolve prompts and parameters. Enable topology evolution to also evolve the agent graph structure:

```bash
python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
    --provider ollama --model llama3.1:8b \
    --reasoning-model llama3.1:8b \
    --enable-topology-evolution --max-roles 6
```

Topology mutations include: `ADD_ROLE`, `REMOVE_ROLE`, `REWIRE_GRAPH`.

## Available Datasets

| Dataset | Tier | Samples | Description |
|---------|------|---------|-------------|
| BEIR | 1 | Varies | 15 zero-shot retrieval benchmarks (MS MARCO, TREC, etc.) |
| Natural Questions | 1 | 300K+ | Google's real user questions from Wikipedia |
| HotpotQA | 2 | 113K | Multi-hop questions requiring reasoning over 2+ articles |
| FRAMES | 3 | 824 | Complex multi-hop questions from Google Research |

Select via `--dataset-tier 1|2|3`.

## Pre-built SOPs

Benchmarks use pre-built SOP configurations:

| SOP | Roles | Use Case |
|-----|-------|----------|
| Simple QA | answerer | Baseline sanity check (no retrieval) |
| RAG Retriever | retriever, answerer | Standard 2-agent RAG |
| Multihop RAG | decomposer, router, retriever, synthesizer | Complex multi-step reasoning |
| Evolvable RAG | configurable | Parameterized RAG with tunable settings |

The self-improvement benchmark uses **Multihop RAG** by default.

## Metrics

### QA Metrics
- `benchmark_accuracy`: Exact + partial match against ground truth
- `benchmark_f1`: Token-level F1 score
- `benchmark_partial_match`: Bidirectional substring matching

### Retrieval Metrics (RAG benchmarks)
- `ndcg@k`: Normalized Discounted Cumulative Gain
- `recall@k`: Recall at K
- `mrr`: Mean Reciprocal Rank
- `map`: Mean Average Precision

## Reports

All benchmarks generate reports in multiple formats:

| Format | File | Content |
|--------|------|---------|
| Markdown | `*.md` | Human-readable with tables and charts |
| JSON | `*.json` | Machine-readable raw data |
| HTML | `*.html` | Interactive charts (Plotly) |

Reports include:
- Executive summary with key improvement metrics
- Performance comparison table (initial vs evolved)
- Learning curve visualization
- Statistical significance tests (Wilcoxon signed-rank)
- Prompt diff analysis
- Configuration summary

### Agentic Comparison Reports

The comparison benchmark generates additional sections:
- Side-by-side mode comparison table
- Winner analysis (quality, speed, efficiency)
- Agentic-specific metrics (inner iterations, supervisor redirections)
- Mode recommendations based on data

## Statistical Rigor

| Feature | Tier 1 | Tier 2 | Tier 3 |
|---------|--------|--------|--------|
| Wilcoxon signed-rank test | Yes | Yes | Yes |
| Confidence intervals | No | Yes | Yes |
| Bonferroni correction | No | Yes | No |
| FDR correction | No | No | Yes |
| Ablation studies | No | No | Yes |
| Power analysis | No | No | Yes |

Minimum 50 samples recommended for meaningful statistical power (SE ~ 0.07, detects ~14% improvements).

## Reproducibility

Set `--random-seed` for deterministic runs:

```bash
python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
    --provider ollama --model llama3.2:1b \
    --reasoning-model llama3.1:8b \
    --random-seed 42 --quick
```

Benchmarks support checkpoint/resume for crash recovery:

```bash
# Initial run (saves checkpoints)
python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
    --provider ollama --model llama3.2:1b \
    --reasoning-model llama3.1:8b \
    --output-dir results/my_run

# Resume from checkpoint
python -m siare.benchmarks.scripts.run_self_improvement_benchmark \
    --provider ollama --model llama3.2:1b \
    --reasoning-model llama3.1:8b \
    --output-dir results/my_run --resume
```

## Programmatic Usage

```python
from siare.benchmarks.self_improvement_benchmark import (
    SelfImprovementBenchmark,
    SelfImprovementConfig,
)
from siare.core.models import AgenticVariationConfig

# Configure with agentic variation
config = SelfImprovementConfig(
    max_generations=10,
    model="gpt-4o-mini",
    reasoning_model="gpt-4o",
    agentic_config=AgenticVariationConfig(
        mode="adaptive",
        maxInnerIterations=5,
        agentModel="gpt-4o",
    ),
)

benchmark = SelfImprovementBenchmark(
    config=config,
    llm_provider=my_provider,
    base_sop=my_sop,
    base_genome=my_genome,
)

result = benchmark.run()
print(result.summary())
```

## Output Directory Structure

```
benchmarks/results/
├── self_improvement/
│   ├── self_improvement_20260327_120000.md
│   ├── self_improvement_20260327_120000.json
│   └── checkpoint.json
└── agentic_comparison/
    ├── comparison_20260327_130000.md
    ├── comparison_20260327_130000.json
    ├── single_turn/
    │   └── self_improvement_*.json
    ├── agentic/
    │   └── self_improvement_*.json
    └── adaptive/
        └── self_improvement_*.json
```
