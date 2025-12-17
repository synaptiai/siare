# SIARE Glossary

Definitions of key terms used throughout the SIARE documentation.

---

## Core Concepts

### SOP (Standard Operating Procedure)
In SIARE, an SOP is a complete pipeline configuration that defines how agents collaborate. Represented as `ProcessConfig` in code. An SOP includes roles, graph structure, prompts, and constraints.

**Also called:** Pipeline, ProcessConfig

### Role
An individual agent within a pipeline. Each role has a model, tools, prompts, and defined inputs/outputs. In user-facing documentation, roles may be called "agents" for clarity.

**Code:** `RoleConfig`
**User-facing:** Agent

### Director
The AI service that analyzes pipeline performance and proposes improvements. Acts as the "brain" of the evolution system - both diagnosing weaknesses and suggesting mutations.

**Code:** `DirectorService`

### Gene Pool
Storage for SOP versions and their performance history. Tracks ancestry (parent-child relationships), computes Pareto frontiers, and maintains diversity via QD Grid.

**Code:** `GenePool`

### Mutation
A targeted change to an SOP designed to improve performance. SIARE supports 7 mutation types: PROMPT_CHANGE, PARAM_TWEAK, ADD_ROLE, REMOVE_ROLE, REWIRE_GRAPH, CROSSOVER, META_PROMPT_CHANGE.

**Code:** `MutationType`, `MutationProposal`

---

## Evolution Terminology

### Evolution Job
A complete evolution run with defined budget, constraints, and phases. Orchestrated by the `EvolutionScheduler`.

### Generation
One iteration of the evolution loop: execute → evaluate → diagnose → mutate.

### Pareto Frontier
The set of SOPs where no single SOP dominates all others across all metrics. Represents optimal trade-offs (e.g., high accuracy vs. low cost).

### QD Grid (Quality-Diversity Grid)
A data structure that maintains diverse solutions across behavioral dimensions. Prevents evolution from converging to a single local optimum.

### Fitness
The overall performance score of an SOP, typically a weighted combination of multiple metrics.

---

## Metrics & Evaluation

### EvaluationVector
A collection of metric scores for a single SOP execution. Contains individual metric values plus an aggregate score.

**Code:** `EvaluationVector`

### LLM Judge
A metric that uses an LLM to evaluate quality. Common for subjective measures like "answer quality" or "relevance."

**Code:** `MetricType.LLM_JUDGE`

### Programmatic Metric
A metric computed by code (not LLM). Examples: latency, cost, word count.

**Code:** `MetricType.PROGRAMMATIC`

### Runtime Metric
A metric that measures execution characteristics like latency or API cost.

**Code:** `MetricType.RUNTIME`

---

## Pipeline Structure

### Graph
The DAG (Directed Acyclic Graph) that defines execution order and data flow between roles.

**Code:** `GraphEdge`

### Conditional Edge
A graph edge that only executes if a condition is met. Enables dynamic routing based on intermediate outputs.

**Code:** `GraphEdge.condition`

### PromptGenome
The collection of prompts for all roles in an SOP. The "DNA" that gets mutated during evolution.

**Code:** `PromptGenome`

### PromptConstraints
Rules that limit what evolution can change in a prompt. Protects critical instructions from modification.

**Code:** `PromptConstraints`

---

## Tools & Adapters

### Tool Adapter
A class that wraps external functionality (APIs, databases, search engines) for use by agents.

**Base class:** `ToolAdapter`

### Tool Registry
A global registry of available tool adapters. Adapters register via the `@register_adapter` decorator.

**Code:** `ToolRegistry`

---

## Prompt Evolution

### TextGrad
A prompt optimization strategy that uses LLM-generated critiques as "gradients" to improve prompts.

**Code:** `TextGradStrategy`

### EvoPrompt
A prompt optimization strategy based on evolutionary algorithms (genetic algorithms or differential evolution).

**Code:** `EvoPromptStrategy`

### MetaPrompt
A prompt optimization strategy that uses LLM meta-analysis to identify patterns and propose improvements.

**Code:** `MetaPromptStrategy`

---

## API & Infrastructure

### Execution Trace
A complete record of one pipeline run, including all role inputs/outputs, timing, and costs.

**Code:** `ExecutionTrace`

### ConfigStore
Persistent storage for SOPs, prompts, and configurations.

**Code:** `ConfigStore`

### Circuit Breaker
Fault tolerance mechanism that prevents repeated failures from overwhelming the system.

**Code:** `CircuitBreaker`

---

## Terminology Mapping

| User-Facing Term | Code Term | Description |
|------------------|-----------|-------------|
| Agent | Role | An individual component in the pipeline |
| Pipeline | SOP / ProcessConfig | The complete configuration |
| Evolution | Mutation + Selection | The optimization process |
| Score | EvaluationVector | Performance metrics |
| Memory | Context | Information passed between agents |

---

## See Also

- [Architecture](architecture/SYSTEM_ARCHITECTURE.md) — System design
- [Data Models](architecture/DATA_MODELS.md) — Complete model reference
- [Mutation Operators](reference/mutation-operators.md) — All 7 mutation types
