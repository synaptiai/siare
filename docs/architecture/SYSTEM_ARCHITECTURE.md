# SIARE System Architecture

**Version:** 2.0 (Consolidated)
**Status:** Production Reference

This document consolidates the complete system architecture for SIARE (Self-Improving Agentic RAG Engine).

> **Open-Core Architecture**: SIARE follows an open-core model:
> - **siare** (this package): MIT-licensed core with evolution engine, gene pool, QD grid, execution, evaluation
> - **siare-cloud** (enterprise): Proprietary features including auth, billing, audit, approval workflows
>
> Sections marked with 🔒 describe enterprise features not included in this open-source package.

---

## 1. Overview & Design Principles

SIARE treats RAG pipeline configuration as a searchable space and uses AI-driven evolution to discover optimal strategies.

### Core Principles

1. **Config over code**: Pipelines defined as data (SOP configs), not hard-coded flows
2. **Separation of concerns**: Execution, Evaluation, Optimization, Domain adapters
3. **Multi-objective by default**: Accuracy, safety, cost, latency are explicit metrics
4. **Human-in-the-loop**: Humans approve which SOPs are promoted and deployed
5. **Evolution as first-class**: Built to iterate on its own configurations
6. **Quality-Diversity**: Maintain diverse, high-performing strategies (MAP-Elites)

### Data Flow

```
ProcessConfig (SOP) → ExecutionEngine → EvaluationVector → Director → Mutated SOP → GenePool
```

---

## 2. Core Components

### 2.1 Config Store

Persists and versions all configuration artifacts:
- `ProcessConfig` (SOPs) - Multi-agent pipeline definitions
- `PromptGenome` - Evolvable prompts per role
- `MetaConfig` - Director/judge prompts
- `MetricConfig` - Evaluation metrics
- `ToolConfig` - Tool adapter configurations

### 2.2 Execution Engine (Guild Runner)

Builds and runs multi-agent DAGs:
- Nodes = `RoleConfig`s bound with `RolePrompt`s
- Edges = `GraphEdge` entries with optional conditions
- Conditional execution via safe eval (regex-validated, keyword blacklist)
- Emits execution traces with node inputs/outputs, timings, tokens

### 2.3 Evaluation Service

Runs metrics over execution traces:
- **LLM Judge**: Model-based evaluation with textual critiques
- **Programmatic**: Custom functions (accuracy, compliance checks)
- **Runtime**: Latency, tokens, cost derived from traces
- **Human**: Slots for expert ratings

**Statistical Aggregation:**
- Bootstrap confidence intervals (95% CI)
- Benjamini-Hochberg correction for multiple comparisons
- Configurable per metric: mean, median, min, max, p95
- Task weighting support

### 2.4 Director Service

AI brain that evolves SOPs through diagnosis and mutation:

**Diagnostician:**
- Analyzes `EvaluationVector` scores + artifacts
- Produces `Diagnosis` with weaknesses, strengths, root causes

**Architect:**
- Generates `SOPMutation`s based on diagnosis
- Validates constraints before mutations
- Supports 6 mutation types

### 2.5 Gene Pool

Stores SOP versions and performance:
- Ancestry tracking (parent → child lineage)
- Semantic versioning (patch/minor/major)
- Aggregated metrics per SOP
- QD grid integration

### 2.6 QD Grid Manager (MAP-Elites)

Maintains diverse population of high-performing SOPs:
- Discretizes feature space into cells
- Maintains highest-quality SOP per occupied cell
- Enables diversity-aware sampling

### 2.7 Evolution Scheduler

Orchestrates the evolution loop:
- Manages job lifecycle and phase transitions
- Selects parents using configurable strategies
- Coordinates execution → evaluation → mutation cycle
- Tracks budget and convergence

---

## 3. Evolution Engine

### 3.1 Evolution Loop

```
Generation N
    │
    ▼
1. Parent Selection (from QD Grid + Pareto)
    │
    ▼
2. Mutation (Director generates SOP variants)
    │
    ▼
3. Evaluation (Run on task set, compute metrics)
    │
    ▼
4. Gene Pool Update (Add to pool, update QD Grid & Pareto)
    │
    ▼
5. Termination Check (Budget, convergence, max generations)
    │
    ▼
Next Generation
```

### 3.2 Selection Strategies (8 Implemented)

| Strategy | Description |
|----------|-------------|
| **Tournament** | k-way tournament selection |
| **Roulette Wheel** | Probability proportional to fitness |
| **Rank-Based** | Linear ranking with selection pressure |
| **Elitist** | Always include top performers |
| **Fitness Proportionate** | Selection proportional to fitness |
| **Truncation** | Select top k% only |
| **Stochastic Universal Sampling** | Evenly spaced selection points |
| **Lexicase** | Filter by random metric order |

### 3.3 Mutation Types (6 Implemented)

| Type | Scope | Description |
|------|-------|-------------|
| `PROMPT_CHANGE` | Prompts | Modify role prompts |
| `PARAM_TWEAK` | Parameters | Adjust hyperparameters (temperature, max_tokens) |
| `ADD_ROLE` | Topology | Add new specialist role |
| `REMOVE_ROLE` | Topology | Remove underperforming role |
| `REWIRE_GRAPH` | Topology | Change edge connections |
| `CROSSOVER` | Structure | Combine two SOPs (stubbed) |

### 3.4 Multi-Phase Evolution

```python
phases = [
  {
    "name": "Phase 1: Prompt Refinement",
    "allowedMutationTypes": ["prompt_change", "param_tweak"],
    "maxGenerations": 20
  },
  {
    "name": "Phase 2: Structural Exploration",
    "allowedMutationTypes": ["add_role", "rewire_graph", "prompt_change"],
    "maxGenerations": 30
  },
  {
    "name": "Phase 3: Fine-Tuning",
    "allowedMutationTypes": ["param_tweak", "prompt_change"],
    "maxGenerations": 10
  }
]
```

---

## 4. Quality-Diversity Grid

### 4.1 Feature Space

Each SOP characterized by:

**Complexity** (normalized 0-1):
```
complexity = 0.4 * numRoles + 0.3 * avgChainDepth + 0.2 * numEdges + 0.1 * avgPromptLength
```

**Diversity Embedding** (384-dim):
- Sentence embedding of concatenated prompts
- Captures semantic diversity

**Domain Features** (configurable):
- Domain-specific metrics as features
- Example: ethics_focus, regulatory_coverage, safety_emphasis

### 4.2 Grid Operations

- **Cell Assignment**: Map SOP features to grid cell
- **Elite Replacement**: Higher quality replaces current elite
- **Sampling**: Uniform, curiosity-driven, or quality-weighted

### 4.3 Pareto Frontier

Computes non-dominated SOPs across metrics:
- Incremental update for efficiency
- Periodic full recomputation
- Integration with QD grid for selection

---

## 5. Metric Aggregation & Statistics

### 5.1 Aggregation Methods

| Method | Use Case |
|--------|----------|
| `mean` | Average performance (default) |
| `median` | Robust to outliers |
| `min` | Worst-case performance |
| `max` | Best-case performance |
| `p95` | High percentile |

### 5.2 Statistical Rigor

**Bootstrap Confidence Intervals:**
```python
# 10,000 bootstrap samples
# 95% confidence interval (2.5th, 97.5th percentile)
ci_low, ci_high = bootstrap_ci(scores, n_bootstrap=10000)
```

**Multiple Comparison Correction:**
```python
# Benjamini-Hochberg procedure for FDR control
adjusted_p_values = benjamini_hochberg(raw_p_values)
```

**Task Weighting:**
- Tasks with `weight` field contribute proportionally
- Higher weight = more influence on aggregate

---

## 6. Error Handling & Fault Tolerance

### 6.1 Implemented Components

| Component | Purpose |
|-----------|---------|
| **CircuitBreaker** | 3-state pattern (CLOSED, OPEN, HALF_OPEN) |
| **RetryHandler** | Exponential backoff with jitter |
| **CheckpointManager** | Job checkpoint save/restore |

### 6.2 Transaction Semantics

- SOPGene insertion: Atomic via database transaction
- EvaluationVector writes: Batch-wise with all-or-nothing
- Failed runs stored with `status: "failed"`
- Gene Pool write failures: Rollback entire generation

### 6.3 Recovery Strategy

| Failure Type | Response |
|--------------|----------|
| Execution failure | Retry 3x with exponential backoff |
| Evaluation failure | Mark run as failed, continue batch |
| Director failure | Halt generation, preserve state, alert |
| Gene Pool write failure | Rollback entire generation |

---

## 7. Safety & Governance 🔒

> **Enterprise Feature**: Full safety governance is available in siare-cloud. The open-source core includes basic safety validation.

### 7.1 Deployment Gates

**Multi-Stage Approval:**
1. Auto-validation (schema checks, safety thresholds)
2. Domain expert review (performance, risk)
3. Production approval (authorized stakeholders)

### 7.2 Safety Validator

- Toxicity detection
- Bias detection
- PII detection
- Constraint validation (`mustNotChange`)

### 7.3 Emergency Controls

- **Kill Switch**: Emergency SOP deactivation
- **Auto-Rollback**: Revert to last known good version
- **Runtime Monitoring**: Continuous safety metric tracking

---

## 8. Governance & Deployment 🔒

> **Enterprise Feature**: Approval workflows, deployment services, and auth are available in siare-cloud.

### 8.1 Approval Workflow Engine

Multi-stage approval for SOP deployments:

```
Deployment Request
    │
    ▼
Stage 1: Auto-Validation (safety checks, schema)
    │
    ▼
Stage 2: Domain Expert Review
    │
    ▼
Stage 3: Production Approval
    │
    ▼
Deployed
```

**Approval Types:**
- `AUTO`: Automated validation (safety thresholds, schema checks)
- `SINGLE`: One approver from required roles
- `MULTI`: Multiple approvers (quorum-based)
- `UNANIMOUS`: All required approvers must approve

**Key Features:**
- Configurable approval chains per environment
- Timeout and escalation policies
- Conditional approvals with comments
- Approval history and audit trail

### 8.2 Deployment Service

Manages SOP lifecycle across environments:

**Deployment States:**
```
CREATED → PENDING → VALIDATING → APPROVED → DEPLOYING → DEPLOYED → ACTIVE
                                    ↓
                                 REJECTED
                                    ↓
                                 DENIED
```

**Environment Support:**
- `development`: Unrestricted deployment
- `staging`: Domain expert approval required
- `production`: Full approval workflow + safety validation

**Version Management:**
- Immutable deployment versions
- Rollback to previous versions
- A/B deployment support (planned)

### 8.3 Authentication & Authorization

**Authentication Service:**
- API key authentication (`SIARE_API_KEY` environment variable)
- JWT token support (configurable)
- OAuth/SAML integration (planned)
- Backward compatible demo mode

**Authorization Service:**
- Resource-based access control (RBAC)
- Scope-based permissions (domain, resource ID)
- Conditional permissions (time, MFA)
- Role inheritance

**Predefined Roles:**
| Role | Capabilities |
|------|-------------|
| `admin` | Full system access |
| `domain_owner` | Full access to assigned domains |
| `researcher` | Run evolution, no deployment |
| `viewer` | Read-only access |

### 8.4 Prompt Evolution Framework

Three optimization strategies for automatic prompt improvement:

**TextGrad Strategy:**
- Textual gradient descent with LLM-generated critiques
- Analyzes failure patterns and generates targeted improvements
- Best for: Iterative refinement of specific issues

**EvoPrompt Strategy:**
- Evolutionary algorithms (GA/DE) for prompt populations
- Maintains population diversity through crossover and mutation
- Best for: Exploring diverse prompt variations

**MetaPrompt Strategy:**
- LLM meta-analysis for targeted improvements
- High-level structural changes based on failure analysis
- Best for: Significant prompt restructuring

**Adaptive Strategy Selector:**
- Auto-selects optimal strategy based on failure patterns
- Considers: failure type, iteration count, improvement rate
- Falls back to ensemble approach for complex failures

**Key Components:**
- `PromptParser`: Section-based prompt parsing (markdown/LLM)
- `FeedbackExtractor`: Extract actionable feedback from critiques
- `SectionMutator`: Apply targeted mutations to prompt sections
- `ConstraintValidator`: Ensure mutations respect `mustNotChange` constraints

---

## 9. Observability & Operations

> **Note**: Audit Service (9.1) and Safety Monitor (9.2) are enterprise features 🔒. Checkpoint Manager (9.3) is included in the open-source core.

### 9.1 Audit Service 🔒

Comprehensive logging of system operations:

**Audit Events:**
- Authentication events (login, logout, token refresh)
- Authorization decisions (permitted/denied)
- SOP modifications (create, update, deploy)
- Evolution job lifecycle events
- Approval workflow events

**Event Classification:**
- `SECURITY`: Authentication, authorization, access
- `OPERATION`: CRUD operations, job management
- `SYSTEM`: Health, errors, recovery

### 9.2 Safety Monitor 🔒

Runtime safety metric tracking:

**Monitored Metrics:**
- Response toxicity scores
- Bias detection results
- PII exposure risk
- Safety threshold violations

**Alert Conditions:**
- Metric degradation (>10% below baseline)
- Threshold violation (configurable)
- Anomaly detection (statistical)

### 9.3 Checkpoint Manager

Job checkpointing for fault tolerance:

**Checkpoint Contents:**
- Current generation state
- Gene pool snapshot
- QD grid state
- Evolution job configuration

**Recovery Semantics:**
- Resume from last checkpoint
- Partial generation recovery
- Automatic checkpoint on phase completion

---

## 10. API Reference Summary 🔒

> **Enterprise Feature**: The REST API server is available in siare-cloud. The open-source core provides CLI (`siare init`, `siare evolve`, `siare run`) and Python library interfaces.

### 10.1 Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/run` | POST | Execute single SOP |
| `/v1/run/batch` | POST | Batch execution |
| `/v1/jobs` | POST | Start evolution job |
| `/v1/jobs/{id}` | GET | Job status |
| `/v1/jobs/{id}/stream` | WebSocket | Real-time updates |
| `/v1/gene-pool` | GET | Query SOPs |
| `/v1/frontier` | GET | Pareto-optimal SOPs |
| `/v1/qd-grid` | GET | QD grid data |

### 10.2 Configuration Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/sops` | GET/POST | SOP management |
| `/v1/prompts` | GET/POST | PromptGenome management |
| `/v1/metrics` | GET/POST | MetricConfig management |
| `/v1/tools` | GET/POST | ToolConfig management |

### 10.3 Governance Endpoints 🔒

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/deployments` | POST | Initiate promotion |
| `/v1/deployments/{id}/approve` | POST | Approve stage |
| `/v1/deployments/{id}/rollback` | POST | Rollback |
| `/v1/kill-switch/{sopId}` | POST | Emergency stop |

### 10.4 Authentication Endpoints 🔒

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/auth/validate` | GET | Validate API key |
| `/v1/auth/me` | GET | Current user info |

### 10.5 Prompt Evolution Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/prompts/{id}/optimize` | POST | Trigger prompt optimization |
| `/v1/prompts/{id}/history` | GET | Prompt evolution history |

---

## 11. Data Models Reference

See `docs/architecture/DATA_MODELS.md` for complete Pydantic model definitions.

### Key Models

- `ProcessConfig`: SOP definition with roles, graph, tools
- `RoleConfig`: Individual agent configuration
- `GraphEdge`: DAG edge with optional condition
- `PromptGenome`: Evolvable prompts per role
- `MetricConfig`: Evaluation metric definition
- `EvaluationVector`: Scores + artifacts from evaluation
- `SOPGene`: Gene pool entry with ancestry
- `Diagnosis`: Director analysis output
- `SOPMutation`: Mutation proposal with rationale

---

## 12. Extension Points

### Adding a Tool Adapter

```python
from siare.adapters.base import ToolAdapter, register_adapter

@register_adapter("my_tool")
class MyToolAdapter(ToolAdapter):
    def initialize(self) -> None:
        self.is_initialized = True

    def execute(self, inputs: dict) -> dict:
        return {"result": "..."}

    def validate_inputs(self, inputs: dict) -> list[str]:
        return []
```

### Adding a Metric

```python
def my_metric(trace: ExecutionTrace, task_data: dict) -> float:
    return 0.85  # Score between 0 and 1

evaluation_service.register_metric_function("my_metric", my_metric)
```

### Adding a Mutation Type

1. Add to `MutationType` enum in `siare/core/models.py`
2. Implement `_apply_<type>_mutation()` in `siare/services/director.py`
3. Add to mutation dispatch in `mutate_sop()`
4. Write tests

**Last Updated:** 2025-12-17
