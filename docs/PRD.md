<!-- PRD.md -->

# Generic Self-Improving Agentic RAG Engine (SIARE)
_configurable, multi-domain, self-evolving agentic RAG platform_

---

## 1. Summary

We want a **generic engine** that can:

- Execute **configurable agentic RAG pipelines** (SOP-driven multi-agent graphs).
- Evaluate their behavior across **multiple metrics** (accuracy, cost, safety, domain constraints, etc.).
- Run an **outer evolutionary loop** that:
  - Diagnoses weaknesses using metrics,
  - Mutates the pipeline configuration (SOP),
  - Maintains a **gene pool** of configurations and their performance,
  - Surfaces **Pareto-optimal strategies** for humans to select from.

The engine must be **domain-agnostic**: the same core can support clinical trial design, construction tender analysis, legal review, trading intelligence, internal support bots, etc., by swapping in different:

- Knowledge adapters (vector stores, SQL, APIs),
- SOP configs (roles, tools, graphs),
- Metric suites.

---

## 2. Background & Motivation

Most RAG systems today are:

- Hand-tuned for one narrow use case.
- Hard-coded as a single agent or a brittle chain.
- “Improved” via ad-hoc prompt tweaks and manual intuition.

Agentic RAG plus an explicit self-improvement loop reframes this as:

- A **search problem in configuration space** (SOP space),
- Optimized against a **multi-objective evaluation vector**,
- Guided by a **Director** process that reasons about performance and proposes pipeline changes.

The aim is a **platform**, not another bespoke workflow.

---

## 3. Goals

### G1. Domain-agnostic core

A single engine that can support multiple domains via configuration:

- No domain logic hardcoded into the core.
- Domain-specific behavior comes from configs, adapters, and metric definitions.

### G2. Configurable multi-agent RAG

- Represent complex pipelines as **Standard Operating Procedures (SOPs)**:
  - Roles, models, prompts,
  - Tools,
  - Graph structure (DAG, branching),
  - Tunable hyperparameters.
- Execute these SOPs reliably with traceable behavior.

### G3. Multi-metric evaluation

- Support **pluggable metrics**:
  - LLM-as-Judge metrics,
  - Programmatic metrics (e.g. accuracy vs labels, constraint checks),
  - Runtime metrics (cost, latency),
  - Human feedback metrics.
- Aggregate results into structured **EvaluationVectors**.

### G4. Self-improvement loop

- Implement an **evolution engine** that:
  - Runs pipelines on tasks,
  - Evaluates them,
  - Diagnoses weaknesses,
  - Proposes new SOP variants,
  - Maintains a **gene pool** and **Pareto frontier** across metrics.

### G5. Advanced evolution capabilities (v2+)

- Continuous evolution over many generations.
- Distilled **Director policy model** for cheaper/faster mutation proposals.
- Topology search: add/remove agents and edges under constraints.
- Genetic operators: mutation + crossover.
- Live data adapters (APIs, streaming) instead of static snapshots.
- Tight integration of **human expert feedback** as a first-class signal.

### G6. Safety, cost, and governance

- Treat safety, compliance, cost, and latency as **first-class objectives or constraints**.
- Ensure human decision-makers remain in control of what’s actually deployed.

---

## 4. Non-Goals

- Training custom foundation models from scratch.
- Fully replacing domain expert judgment.
- Building a full MLOps platform or feature store.
- Heavy end-user UI; focus is on API + minimal operator console.

---

## 5. Target Users & Personas

### 5.1 Platform / Infra teams

- Operate the engine.
- Define global SLAs, cost budgets, and safety constraints.
- Integrate with tracing, logging, and model providers.

### 5.2 Domain solution teams

- Own specific use cases (e.g., clinical trial design, NS 3420 tender analysis, legal review, trading intel).
- Define domain-specific:

  - SOP configs (agent graphs),
  - Knowledge adapters,
  - Metric suites,
  - Human feedback workflows.

### 5.3 Domain experts (reviewers)

- Provide labeled data and expert scores.
- Review Pareto-front configurations before deployment.
- Give feedback that shapes the evolution and Director policy.

---

## 6. Representative Use Cases

> These are examples to validate generality, not baked-in domains.

### 6.1 Clinical trial protocol assistant

- Generate and critique inclusion/exclusion criteria.
- Check ethics and regulatory alignment against guidelines.
- Estimate cohort feasibility from patient data.

### 6.2 Construction tender risk analysis

- Analyze tender documents against NS 3420 and other norms.
- Identify technical and contractual risks per discipline.
- Generate structured risk reports and questions for the client.

### 6.3 Legal contract review

- Highlight risky clauses and suggest edits.
- Check alignment with internal policy libraries and regulations.
- Produce review summaries and redline suggestions.

### 6.4 Trading / market intelligence

- Fuse news, macro data, sentiment, and portfolio context.
- Produce structured signals and narratives for PMs and analysts.
- Balance latency vs depth of reasoning.

---

## 7. Product Scope & Capabilities

### 7.1 Core capabilities (v1)

1. **SOP-based agentic execution**

   - Load a `ProcessConfig` (SOP) describing:
     - Roles, models, prompts,
     - Tools and bindings,
     - Graph structure (DAG),
     - Hyperparameters.
   - Execute the graph deterministically given inputs and config.
   - Produce a **final answer** and a **trace** (for evaluation and debugging).

2. **Knowledge adapter & tool abstraction**

   - Built-in adapters for:
     - Vector search,
     - SQL/analytics DBs,
     - HTTP/REST APIs,
     - File/document stores,
     - Web search.
   - Expose these as **tools** referenced by roles in the SOP config.

3. **Evaluation framework**

   - Metric registry supporting:
     - LLM-as-Judge metrics (with model + prompt + schema),
     - Programmatic metrics (Python functions, SQL checks),
     - Runtime metrics (cost, latency).
   - Combine metric outputs into an **EvaluationVector** per run.

4. **Gene pool & Pareto analysis**

   - Persist SOP versions and their EvaluationVectors.
   - Compute **Pareto frontiers** across selected metric subsets.
   - Provide APIs to:
     - List all SOPs and their aggregated metrics,
     - List Pareto-optimal SOPs for a given domain and metric set.

5. **Director v1 (LLM-based)**

   - **Diagnostician:**
     - Input: EvaluationVector (+ optional trace summaries).
     - Output: structured Diagnosis (primary weakness, root causes, recommendations).
   - **Architect:**
     - Input: current SOP + Diagnosis.
     - Output: batch of mutated SOP configs (configuration deltas with rationale).

6. **Observability**

   - Trace each run:
     - Agent invocations and prompts,
     - Tool calls,
     - Timings and token usage,
     - Errors.
   - Attach traces to evaluations and gene pool entries.

### 7.2 Advanced capabilities (v2+)

1. **Continuous evolution**

   - Scheduler for repeated evolution cycles over:
     - Offline benchmarks,
     - Sampled live traffic.
   - Budget and convergence controls.

2. **Director policy distillation**

   - Train a smaller policy model from Director history:
     - `(ProcessConfig, metrics_before, Δmetrics)` → `(mutation proposals)`.
   - Use small model for frequent suggestions; large LLM for difficult/novel cases.

3. **Topology search**

   - Allow controlled changes to:
     - Role sets,
     - Edges,
     - Branching logic.
   - Use a library of agent templates and graph motifs to keep search tractable.

4. **Genetic operators (crossover)**

   - Combine segments of two successful SOPs into a new candidate.
   - Operate over typed segments (retrieval, analysis, critique, synthesis).

5. **Live data adapters**

   - API and streaming connectors with:
     - Auth, retry, and caching,
     - Circuit-breaking and failover to snapshots.

6. **Human-in-the-loop evaluation**

   - UI/API for experts to score outputs on defined axes.
   - Integrate human scores as metrics in EvaluationVectors.
   - Use human ratings when training the Director policy.

---

## 8. Functional Requirements

### 8.1 Must-Haves (v1)

- **FR1:** Load, validate, and execute SOP configs describing multi-agent DAGs.
- **FR2:** Bind SOP roles to concrete tools/adapters via configuration.
- **FR3:** Register and run multiple metrics per pipeline execution.
- **FR4:** Persist SOP versions, evaluations, and ancestry in a gene pool.
- **FR5:** Compute Pareto frontiers over arbitrary subsets of metrics.
- **FR6:** Provide LLM-based Diagnostician and Architect:
  - Output structured diagnoses and at least parameter/prompt-level mutations.
- **FR7:** Provide an API to:
  - Run a specific SOP on given input(s),
  - Run an evolution round over a task set.
- **FR8:** Emit traces for all runs in a format consumable by tracing tools.

### 8.2 Should-Haves (v1–v2)

- **FR9:** Support multiple domains via “domain packages” (configs + metrics).
- **FR10:** Support cost and latency as first-class metrics.
- **FR11:** Support basic safety checks as metrics and/or hard constraints.

### 8.3 Nice-to-Haves (v2+)

- **FR12:** Topology-level mutations (add/remove roles/edges) with constraints.
- **FR13:** Director policy model trained from history.
- **FR14:** Human feedback interface and human metrics.
- **FR15:** Live data connectors with configurable caching/policies.

---

## 9. Non-Functional Requirements

- **NFR1: Extensibility**
  - Easy to add:
    - New metric types,
    - New tool/adapter types,
    - New domain packages.

- **NFR2: Performance**
  - Support mid-scale batch evolution (1000s of runs) with:
    - Concurrency,
    - Backpressure,
    - Job-queueing.

- **NFR3: Cost-awareness**
  - Define per-job and per-generation budgets.
  - Abort or degrade gracefully when budget is exceeded.

- **NFR4: Reliability**
  - Clear error handling.
  - No corruption of gene pool on partial failures.

- **NFR5: Security & compliance**
  - Secret management for adapters.
  - Access-control hooks for domain data access.

---

## 10. Success Metrics

- Reduced manual tuning time for new use cases.
- Improvement over naive/baseline pipelines on domain-specific metrics.
- Stability and diversity of Pareto fronts over time.
- Fraction of Director-generated SOP variants that domain experts accept.
- Cost per unit of performance improvement (efficiency of the evolution loop).

---

## 11. Risks & Open Questions

- **Metric alignment**
  - Risk: optimizing for the wrong proxies.
  - Need explicit, versioned metric sets and regular alignment reviews.

- **LLM-Judge reliability**
  - Risk: biased or gameable evaluators.
  - Mitigation: use diverse judges + programmatic checks + hidden evals.

- **Search-space explosion**
  - Especially with topology search and crossover.
  - Require constrained mutation spaces and budgets.

- **Human feedback cost**
  - Must be selective and high-signal; use active learning strategies.

- **Policy representation**
  - How best to encode ProcessConfigs and metric histories for the Director policy; likely a structured JSON-to-embedding pipeline.