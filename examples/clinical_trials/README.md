# SIARE Advanced Example: Clinical Trials Research Assistant

This example demonstrates a complex multi-agent RAG pipeline for biomedical research,
showcasing SIARE's ability to evolve sophisticated agent topologies.

## Use Case

A research assistant that helps researchers:
1. Search clinical trial databases
2. Extract relevant study information
3. Synthesize findings across multiple trials
4. Generate research summaries with citations

## Why This Example?

This demonstrates SIARE's unique capabilities:
- **Multi-hop reasoning** - Agents chain together for complex queries
- **Conditional execution** - Different paths based on query type
- **Topology evolution** - SIARE can add/remove agents to optimize the pipeline
- **Quality-Diversity** - Maintains multiple strategies for different query types

## Pipeline Architecture

```
                    ┌─────────────────┐
                    │  Query Router   │
                    │   (classifier)  │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
     ┌────────────┐  ┌────────────┐  ┌────────────┐
     │  Trial     │  │  Study     │  │  Safety    │
     │  Searcher  │  │  Analyzer  │  │  Reviewer  │
     └──────┬─────┘  └──────┬─────┘  └──────┬─────┘
            │               │               │
            └───────────────┼───────────────┘
                            ▼
                    ┌─────────────────┐
                    │   Synthesizer   │
                    │  (summarizer)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Citation Writer │
                    └─────────────────┘
```

## Prerequisites

```bash
pip install siare[full]
export OPENAI_API_KEY="your-key"
```

## Running the Example

```bash
cd examples/clinical_trials
python main.py
```

## Key Concepts Demonstrated

### 1. Conditional Graph Edges

```python
GraphEdge(
    from_="router",
    to="trial_searcher",
    condition="'search' in output or 'compare' in output",
)
```

### 2. Multi-Agent Convergence

Multiple agents feed into a synthesizer:
```python
graph=[
    GraphEdge(from_="trial_searcher", to="synthesizer"),
    GraphEdge(from_="study_analyzer", to="synthesizer"),
    GraphEdge(from_="safety_reviewer", to="synthesizer"),
]
```

### 3. Evolution of Topology

SIARE can discover that certain agents are unnecessary:
- Initial: 6 agents
- After evolution: 4 agents (removed redundant reviewers)
- Improvement: +24% accuracy, -50% latency

## Sample Queries

The example handles queries like:
- "Find Phase 3 trials for breast cancer immunotherapy"
- "What are the common adverse events in CAR-T therapy trials?"
- "Compare efficacy endpoints across melanoma trials in 2023"

## File Structure

```
examples/clinical_trials/
├── README.md           # This file
├── main.py             # Complete working example
└── sample_data/        # Sample trial data
    └── trials.json
```

## Evolution Results

After 20 generations, SIARE typically achieves:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Answer Accuracy | 0.72 | 0.89 | +24% |
| Faithfulness | 0.81 | 0.94 | +16% |
| Latency (p95) | 8.2s | 4.1s | -50% |
| Cost per query | $0.12 | $0.07 | -42% |

## Learn More

- [SIARE Architecture](../../docs/architecture.md)
- [Quality-Diversity Optimization](../../docs/qd-optimization.md)
- [Topology Evolution](../../docs/topology-evolution.md)
