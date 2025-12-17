# Implementing Domain Use Cases in SIARE

This guide explains how to implement domain-specific use cases in SIARE, using the **clinical trial design** use case as a concrete example. It provides a general pattern that can be applied to any domain requiring autonomous, self-improving multi-agent RAG systems.

## Table of Contents

1. [Understanding SIARE's Architecture](#understanding-siares-architecture)
2. [The Clinical Trials Use Case](#the-clinical-trials-use-case)
3. [General Pattern for Use Case Implementation](#general-pattern-for-use-case-implementation)
4. [Step-by-Step Implementation Guide](#step-by-step-implementation-guide)
5. [Best Practices](#best-practices)
6. [Example Use Cases](#example-use-cases)

---

## Understanding SIARE's Architecture

SIARE (Self-Improving Agentic RAG Engine) consists of two main loops:

### Inner Loop: Multi-Agent Execution

```
Task Input
    │
    ↓
ProcessConfig (SOP)
    │
    ├─→ Agent 1 (uses tools, produces outputs)
    ├─→ Agent 2 (uses outputs from Agent 1)
    ├─→ Agent 3 (parallel with Agent 2)
    │      │
    │      ↓
    └─→ Agent 4 (synthesizes all outputs)
           │
           ↓
      Final Output
```

**Key components:**
- **ProcessConfig (SOP)**: Defines agent roles, tools, and collaboration graph
- **Roles**: Individual agents with specific responsibilities
- **Tools**: Adapters for external knowledge (vector DB, SQL, web search, APIs)
- **Graph**: DAG structure with optional conditional edges

### Outer Loop: Self-Evolution

```
Evaluation Results (multi-dimensional metrics)
    │
    ↓
Director Service (AI-powered)
    │
    ├─→ Diagnostician: Analyzes performance, identifies weaknesses
    ├─→ Architect: Proposes mutations targeting gaps
    └─→ Validator: Ensures constraints are met
           │
           ↓
    Mutated SOPs
           │
           ↓
    Re-evaluate on task set
           │
           ↓
    Gene Pool (stores all versions with ancestry)
           │
           ↓
    QD Grid (maintains diversity)
           │
           ↓
    Pareto Frontier (multi-objective optimization)
           │
           ↓
    Human Decision Maker (selects optimal configuration)
```

**Key components:**
- **Director**: AI agent that evolves SOPs (mutations, diagnosis, constraints)
- **Gene Pool**: Version control for SOPs with performance history
- **QD Grid**: Quality-Diversity grid for maintaining diverse solutions
- **Metrics**: Multi-dimensional evaluation (accuracy, cost, latency, domain-specific)

---

## The Clinical Trials Use Case

### Problem Statement

**Manual tuning of clinical trial design systems is extremely difficult.** The system must balance:
- Scientific rigor (evidence-based design)
- Regulatory compliance (FDA requirements)
- Ethical adequacy (patient safety, informed consent)
- Feasibility (recruitment, budget, timeline)

Each trial is unique, and configurations that work in testing often fail with unseen production data.

### Solution Architecture

#### Inner Loop: Specialist Agent Guild

```
Trial Request (indication, intervention, phase, population)
    │
    ├─→ Medical Researcher
    │     ↓ (searches PubMed)
    │   Evidence Base, Literature Analysis
    │     │
    │     ├─→ Regulatory Specialist
    │     │     ↓ (queries FDA guidelines)
    │     │   Compliance Checklist, Requirements
    │     │
    │     ├─→ Ethics Officer (parallel)
    │     │     ↓ (references ethics frameworks)
    │     │   Safety Measures, Informed Consent
    │     │
    │     └─→ Data Analyst (parallel)
    │           ↓ (queries clinical database)
    │         Feasibility Data, Recruitment Projections
    │               │
    │               ↓
    └──────────→ Trial Coordinator
                  ↓ (synthesizes all inputs)
              Trial Design, Protocol Draft
                  │
                  ↓ (conditional on complexity)
              Quality Reviewer
                  ↓
              Final Protocol
```

**Key insight:** Specialists collaborate as a "guild," each contributing domain expertise.

#### Outer Loop: Performance-Driven Evolution

**Scenario:** After 10 evaluations, the Diagnostician notices:
- Scientific Accuracy: 0.88 ✓
- Regulatory Compliance: 0.90 ✓
- Ethical Adequacy: 0.89 ✓
- Feasibility: 0.65 ⚠️ **← Weakest dimension**

**Diagnosis:** Recruitment projections are overly optimistic, leading to unrealistic timelines.

**Mutations proposed:**
1. **PROMPT_CHANGE** for Data Analyst: Add emphasis on historical recruitment challenges
2. **PARAM_TWEAK**: Reduce `min_patient_count` threshold
3. **ADD_ROLE**: Insert Feasibility Reviewer for independent validation

**Result:** After evolution, 3 Pareto-optimal configurations emerge:
- **Config A**: Max scientific rigor (accuracy 0.92, feasibility 0.70)
- **Config B**: Balanced (accuracy 0.88, feasibility 0.78)
- **Config C**: Max feasibility (accuracy 0.85, feasibility 0.85, cost 0.8×)

**Human decision:** Selects Config C for a resource-constrained trial.

### Why This Works in SIARE

1. **Modularity**: Each specialist is an independent role, easy to modify
2. **Composability**: Graph structure allows flexible collaboration patterns
3. **Evolvability**: Director can mutate any aspect (prompts, tools, topology)
4. **Multi-objective**: Pareto frontier respects real-world trade-offs
5. **Safety**: Constraints prevent removal of critical components (mandatory roles, immutable prompt keywords)

---

## General Pattern for Use Case Implementation

Follow this pattern for any domain:

### Phase 1: Domain Analysis

1. **Identify the complex task** that requires multiple perspectives or knowledge sources
2. **Decompose into specialist roles** based on expertise areas
3. **Map knowledge sources** to tool adapters (APIs, databases, documents)
4. **Define evaluation dimensions** reflecting domain priorities
5. **Establish safety constraints** (mandatory components, immutable principles)

### Phase 2: Architecture Design

1. **Design the Inner Loop**
   - Define specialist agent roles with clear responsibilities
   - Map dependencies between roles (who needs whose outputs?)
   - Select appropriate LLM models for each role (complex vs. simple tasks)
   - Identify conditional execution paths (when should certain roles run?)

2. **Design the Outer Loop**
   - Define multi-dimensional metrics aligned with domain goals
   - Set metric weights reflecting domain priorities
   - Specify evolution constraints (what must not change?)
   - Plan evolution phases (optimize specialists → refine coordination → balance trade-offs)

### Phase 3: Implementation

1. **Create tool adapters** for domain-specific knowledge sources
2. **Implement ProcessConfig** (SOP) with roles, graph, prompts
3. **Implement evaluation metrics** (programmatic or LLM-based)
4. **Create PromptGenome** with role prompts and constraints
5. **Assemble DomainPackage** tying everything together
6. **Write evaluation tasks** representing realistic scenarios

### Phase 4: Validation

1. **Run baseline evaluation** on task set
2. **Execute evolution** over multiple generations
3. **Analyze Pareto frontier** for meaningful trade-offs
4. **Validate with domain experts**
5. **Deploy and monitor** in production

---

## Step-by-Step Implementation Guide

### Step 1: Define Your Domain Use Case

**Questions to answer:**
- What complex task requires multiple perspectives?
- What are the key dimensions of quality?
- What knowledge sources are needed?
- What are the safety-critical constraints?

**Example (Clinical Trials):**
- Task: Design clinical trial protocols
- Dimensions: Scientific accuracy, regulatory compliance, ethical adequacy, feasibility
- Knowledge: PubMed, FDA guidelines, ethics frameworks, clinical databases
- Constraints: Cannot remove patient safety, FDA compliance, informed consent requirements

### Step 2: Design Specialist Agents

**Template:**

```python
RoleConfig(
    id="<role_name>",
    model="<llm_model>",  # gpt-5, gpt-5-mini, etc.
    tools=["<tool_id>"],  # Tools this role can use
    promptRef="<prompt_id>",
    inputs=[{"from": "<upstream_role_or_user_input>"}],
    outputs=["<output_1>", "<output_2>"],  # What this role produces
)
```

**Best practices:**
- **Single responsibility**: Each role has one clear job
- **Clear outputs**: Name outputs descriptively (e.g., `literature_analysis`, not `result`)
- **Appropriate models**: Use cheaper models (gpt-5-mini) for straightforward tasks
- **Tool selection**: Only assign tools the role actually needs

**Example:**

```python
RoleConfig(
    id="medical_researcher",
    model="gpt-5",
    tools=["pubmed_search"],
    promptRef="medical_researcher_prompt",
    inputs=[{"from": "user_input"}],
    outputs=["literature_analysis", "evidence_base", "research_gaps"],
)
```

### Step 3: Design the Collaboration Graph

**Template:**

```python
graph = [
    # Sequential dependency
    GraphEdge(from_="role_a", to="role_b"),

    # Multiple inputs
    GraphEdge(from_=["role_a", "role_b"], to="role_c"),

    # Conditional execution
    GraphEdge(
        from_="role_a",
        to="role_b",
        condition="role_a.complexity == 'high'"
    ),
]
```

**Best practices:**
- **Minimize sequential dependencies**: Allow parallel execution where possible
- **Use conditionals wisely**: Skip expensive roles when not needed
- **Validate DAG structure**: Ensure no cycles, all roles reachable

**Example:**

```python
graph = [
    # Parallel specialist analysis
    GraphEdge(from_="user_input", to="medical_researcher"),
    GraphEdge(from_=["user_input", "medical_researcher"], to="regulatory_specialist"),
    GraphEdge(from_=["user_input", "medical_researcher"], to="ethics_officer"),
    GraphEdge(from_=["user_input", "medical_researcher"], to="data_analyst"),

    # Synthesis
    GraphEdge(
        from_=["user_input", "medical_researcher", "regulatory_specialist",
               "ethics_officer", "data_analyst"],
        to="trial_coordinator"
    ),

    # Conditional final review
    GraphEdge(
        from_=["trial_coordinator", "regulatory_specialist", "ethics_officer"],
        to="quality_reviewer",
        condition="trial_coordinator.complexity == 'high'"
    ),
]
```

### Step 4: Create Tool Adapters

**Template:**

```python
from siare.adapters.base import ToolAdapter, register_adapter

@register_adapter("my_tool")
class MyToolAdapter(ToolAdapter):
    def __init__(self, config: dict[str, Any]):
        super().__init__(config)
        # Extract config parameters
        self.api_key = config.get("api_key")
        self.base_url = config.get("base_url")

    def initialize(self) -> None:
        """Set up connections, load resources"""
        self.session = requests.Session()
        self.is_initialized = True

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Main execution logic"""
        query = inputs.get("query")

        # Call external API, query database, etc.
        results = self._call_api(query)

        return {
            "results": results,
            "status": "success",
        }

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """Validate inputs before execution"""
        errors = []
        if "query" not in inputs:
            errors.append("Missing required field: query")
        return errors

    def cleanup(self) -> None:
        """Close connections, free resources"""
        if self.session:
            self.session.close()
```

**Best practices:**
- **Error handling**: Return error messages, don't raise exceptions
- **Rate limiting**: Respect API rate limits (use `time.sleep()` if needed)
- **Caching**: Cache expensive API calls when appropriate
- **Validation**: Check inputs before expensive operations

### Step 5: Implement Evaluation Metrics

**Template:**

```python
from siare.core.models import MetricResult

def calculate_my_metric(
    output_a: Dict[str, Any],
    output_b: Dict[str, Any],
    ground_truth: Dict[str, Any] = None,
) -> MetricResult:
    """
    Calculate metric score

    Args:
        output_a: Output from role A
        output_b: Output from role B
        ground_truth: Expected results (if available)

    Returns:
        MetricResult with score 0-1
    """
    score = 0.0
    details = {}

    # Scoring logic
    # Component 1 (weight: 0.4)
    component_1_score = calculate_component_1(output_a)
    score += component_1_score * 0.4
    details["component_1"] = component_1_score

    # Component 2 (weight: 0.3)
    if ground_truth:
        component_2_score = calculate_component_2(output_b, ground_truth)
        score += component_2_score * 0.3
    else:
        score += 0.15  # Partial credit if no ground truth

    # Component 3 (weight: 0.3)
    component_3_score = calculate_component_3(output_a, output_b)
    score += component_3_score * 0.3
    details["component_3"] = component_3_score

    return MetricResult(
        metricId="my_metric",
        score=round(score, 3),
        source="programmatic",
        details=details,
    )
```

**Best practices:**
- **Weighted components**: Break metrics into sub-components with explicit weights
- **Ground truth optional**: Support evaluation with and without ground truth
- **Details dictionary**: Include diagnostic information for debugging
- **Score range**: Always return 0-1 (SIARE normalizes to this range)

### Step 6: Write Role Prompts with Constraints

**Template:**

```python
from siare.core.models import RolePrompt, PromptConstraints

prompt = RolePrompt(
    id="specialist_prompt",
    content="""You are a [ROLE TITLE] specializing in [DOMAIN].

Your responsibilities:
1. [Responsibility 1]
2. [Responsibility 2]
3. [Responsibility 3]

Given [INPUTS], provide:
- [output_1]: [Description of what this output should contain]
- [output_2]: [Description]

Use the [TOOL_NAME] tool to [PURPOSE].
Focus on [KEY PRIORITIES].

CRITICAL: [Safety-critical requirements]
""",
    constraints=PromptConstraints(
        mustNotChange=["critical", "phrase", "list"],
        domainTips=[
            "Specific guidance for this role",
            "Best practices to emphasize",
            "Common pitfalls to avoid",
        ],
    ),
)
```

**Best practices:**
- **Clear structure**: Use numbered lists for responsibilities and outputs
- **Explicit tool usage**: Tell the agent which tools to use and why
- **Safety-critical keywords**: Mark with "CRITICAL:" prefix
- **Constraints**: Use `mustNotChange` for phrases that ensure safety/compliance
- **Domain tips**: Guide the Director on how to improve this prompt

### Step 7: Assemble the Domain Package

**Template:**

```python
from siare.core.models import DomainPackage, DomainConfig, EvolutionConstraints

def create_my_domain_package() -> DomainPackage:
    # 1. Define SOPs (ProcessConfigs)
    my_sop = ProcessConfig(
        id="my_sop",
        version="1.0.0",
        description="SOP for my domain",
        models={"default": "gpt-5", "analysis": "gpt-5-mini"},
        tools=["tool_1", "tool_2"],
        roles=[...],  # From Step 2
        graph=[...],  # From Step 3
        hyperparameters={...},
    )

    # 2. Define Prompts (PromptGenome)
    my_prompts = PromptGenome(
        id="my_prompts",
        version="1.0.0",
        rolePrompts={...},  # From Step 6
    )

    # 3. Define Metrics
    metrics = [
        MetricConfig(
            id="metric_1",
            type=MetricType.PROGRAMMATIC,
            fnRef="my_module.calculate_metric_1",
            inputs=["output_a", "output_b"],
            weight=0.4,
        ),
        MetricConfig(
            id="metric_2",
            type=MetricType.LLM_JUDGE,
            model="gpt-5",
            promptRef="metric_2_judge",
            inputs=["output_c"],
            weight=0.3,
        ),
    ]

    # 4. Define Tools
    tools = [
        ToolConfig(
            id="tool_1",
            type=ToolType.CUSTOM,
            config={...},
        ),
    ]

    # 5. Define Tasks
    tasks = TaskSet(
        id="eval_tasks",
        domain="my_domain",
        version="1.0.0",
        tasks=[
            Task(
                id="task_1",
                input={...},
                groundTruth={...},
                metadata=TaskMetadata(
                    category="easy",
                    difficulty="medium",
                    importance=1.0,
                ),
            ),
        ],
    )

    # 6. Define Evolution Strategy
    domain_config = DomainConfig(
        defaultEvolutionConfig={
            "phases": [
                {
                    "name": "optimize_specialists",
                    "allowedMutationTypes": ["prompt_change", "param_tweak"],
                    "selectionStrategy": "qd_uniform",
                    "parentsPerGeneration": 4,
                    "maxGenerations": 15,
                },
                {
                    "name": "refine_coordination",
                    "allowedMutationTypes": ["prompt_change", "rewire_graph"],
                    "selectionStrategy": "pareto",
                    "parentsPerGeneration": 3,
                    "maxGenerations": 10,
                },
            ],
        },
        aggregationConfig={
            "quality_weights": {
                "metric_1": 0.4,
                "metric_2": 0.3,
                "metric_3": 0.3,
            },
        },
        recommendedConstraints=EvolutionConstraints(
            budgetLimit=BudgetLimit(
                maxEvaluations=100,
                maxLLMCalls=500,
                maxCost=50.0,
            ),
            maxRoles=10,
            maxEdges=20,
            allowedTools=["tool_1", "tool_2"],
            mandatoryRoles=["critical_role_1", "critical_role_2"],
            promptConstraints=PromptConstraints(
                mustNotChange=["safety", "critical", "keywords"],
            ),
        ),
    )

    # 7. Assemble Package
    return DomainPackage(
        id="my_domain",
        name="My Domain Name",
        version="1.0.0",
        description="Package for [domain description]",
        sopTemplates=["my_sop"],
        promptGenomes=["my_prompts"],
        metaConfigs=["my_meta"],
        toolConfigs=["tool_1", "tool_2"],
        metricConfigs=["metric_1", "metric_2"],
        evaluationTasks=["eval_tasks"],
        domainConfig=domain_config,
        maintainer="Your Name",
        documentation="...",
        tags=["domain", "tag1", "tag2"],
    )
```

### Step 8: Create Demonstration Script

**Template:**

```python
import asyncio
import logging

async def run_demo():
    # 1. Setup
    package = create_my_domain_package()
    tasks = create_example_tasks()

    # 2. Inner Loop Demo
    logger.info("INNER LOOP: Specialist Agent Collaboration")
    for task in tasks[:1]:  # Demo with first task
        results = await run_inner_loop(package, task)
        display_results(results)

    # 3. Outer Loop Demo
    logger.info("OUTER LOOP: Self-Evolution")
    demonstrate_diagnosis(results)
    demonstrate_mutations(results, package)

    # 4. Human-in-the-Loop Demo
    logger.info("HUMAN-IN-THE-LOOP: Pareto Frontier")
    demonstrate_pareto_frontier()

if __name__ == "__main__":
    asyncio.run(run_demo())
```

---

## Best Practices

### Architecture Design

1. **Start simple, iterate**: Begin with 3-4 roles, add complexity as needed
2. **Parallel when possible**: Minimize sequential dependencies for speed
3. **Conditional execution**: Skip expensive roles when not needed
4. **Clear interfaces**: Define explicit inputs/outputs between roles

### Prompt Engineering

1. **Structure over prose**: Use numbered lists, clear sections
2. **Explicit tool usage**: Tell agents which tools to use and when
3. **Output format**: Specify exactly what each output should contain
4. **Safety first**: Mark critical requirements with "CRITICAL:" or "IMPORTANT:"

### Evaluation Metrics

1. **Multi-dimensional**: Don't rely on a single metric
2. **Weighted components**: Make trade-offs explicit
3. **Ground truth optional**: Support evaluation without perfect labels
4. **Programmatic + LLM**: Mix fast programmatic checks with nuanced LLM judges

### Evolution Strategy

1. **Phase-based evolution**: Start with specialists, then coordination, then global optimization
2. **Conservative constraints**: Protect safety-critical components
3. **Diverse selection**: Use QD Grid to maintain diversity
4. **Human oversight**: Always provide Pareto frontier for human decision

### Testing

1. **Unit test adapters**: Test tool adapters in isolation
2. **Integration test SOP**: Test full pipeline execution
3. **Metric validation**: Verify metrics produce expected scores
4. **Evolution simulation**: Test mutation logic without full evolution

---

## Example Use Cases

### Use Case 1: Legal Contract Review

**Problem:** Review complex contracts for risks, compliance, and negotiation opportunities.

**Specialist Agents:**
- Legal Analyst: Identifies legal risks and compliance issues
- Financial Analyst: Assesses financial implications
- Risk Manager: Evaluates liability and mitigation strategies
- Domain Expert: Provides industry-specific context
- Contract Coordinator: Synthesizes recommendations

**Knowledge Sources:**
- Legal precedent database (SQL)
- Regulatory text corpus (vector search)
- Company contract templates (vector search)
- Financial models (API)

**Metrics:**
- Risk identification completeness (0.3)
- Regulatory compliance accuracy (0.3)
- Negotiation recommendation quality (0.2)
- Financial impact assessment (0.2)

### Use Case 2: Software Architecture Design

**Problem:** Design optimal software architecture for complex requirements.

**Specialist Agents:**
- Requirements Analyst: Extracts functional and non-functional requirements
- Security Architect: Identifies security requirements and patterns
- Performance Engineer: Designs for scalability and performance
- Integration Specialist: Plans external system integrations
- Architecture Coordinator: Creates coherent architecture

**Knowledge Sources:**
- Architecture pattern library (vector search)
- Security best practices (vector search)
- Performance benchmarks (SQL database)
- API documentation (web search)

**Metrics:**
- Requirements coverage (0.25)
- Security posture (0.25)
- Scalability potential (0.20)
- Implementation feasibility (0.20)
- Cost efficiency (0.10)

### Use Case 3: Financial Report Analysis

**Problem:** Analyze financial reports for investment decisions.

**Specialist Agents:**
- Financial Analyst: Analyzes financial statements and ratios
- Market Analyst: Assesses market position and competition
- Risk Analyst: Evaluates financial risks
- Industry Expert: Provides sector-specific insights
- Investment Advisor: Synthesizes investment recommendation

**Knowledge Sources:**
- Financial statements database (SQL)
- Market data API (real-time)
- News and filings (web search)
- Industry reports (vector search)

**Metrics:**
- Financial analysis accuracy (0.3)
- Risk assessment completeness (0.25)
- Market context quality (0.20)
- Recommendation clarity (0.15)
- Timeliness (0.10)

### Use Case 4: Scientific Literature Review

**Problem:** Conduct comprehensive literature reviews for research.

**Specialist Agents:**
- Search Strategist: Designs optimal search strategy
- Paper Screener: Filters relevant papers
- Methodology Critic: Assesses research quality
- Data Synthesizer: Extracts and synthesizes findings
- Review Coordinator: Creates structured review

**Knowledge Sources:**
- Academic databases (PubMed, arXiv, etc.)
- Citation graph (network database)
- Research methodology guides (vector search)

**Metrics:**
- Search comprehensiveness (0.25)
- Paper relevance (0.25)
- Methodological rigor (0.20)
- Synthesis quality (0.20)
- Citation completeness (0.10)

---

## Summary: The SIARE Use Case Pattern

Every SIARE use case follows this pattern:

```
1. ANALYZE DOMAIN
   ├─ Complex task decomposition
   ├─ Identify knowledge sources
   ├─ Define quality dimensions
   └─ Establish safety constraints

2. DESIGN ARCHITECTURE
   ├─ Specialist agents (3-6 roles)
   ├─ Collaboration graph (DAG)
   ├─ Tool adapters
   └─ Evaluation metrics

3. IMPLEMENT COMPONENTS
   ├─ ProcessConfig (SOP)
   ├─ PromptGenome (role prompts)
   ├─ Tool adapters
   ├─ Metric functions
   └─ DomainPackage

4. EVOLVE AND OPTIMIZE
   ├─ Run baseline evaluation
   ├─ Execute evolution (Director)
   ├─ Analyze Pareto frontier
   └─ Select optimal configuration

5. DEPLOY AND MONITOR
   ├─ API deployment
   ├─ Production monitoring
   ├─ Continuous evolution
   └─ Human feedback loop
```

The key insight: **Treat the RAG pipeline configuration as a searchable space, and use AI-driven evolution to discover optimal strategies.**

---

## Next Steps

1. **Choose your domain**: Identify a complex task in your organization
2. **Follow the pattern**: Use this guide to implement your use case
3. **Start simple**: Begin with 3-4 agents, iterate based on results
4. **Evolve**: Let SIARE discover optimal configurations
5. **Deploy**: Use the API to integrate into production systems

For more examples, see:
- `siare/domains/rag_package.py` - Generic RAG use case
- `siare/domains/clinical_trials_package.py` - Clinical trials use case
- `examples/` - Demonstration scripts and metrics

**Questions?** Check the main documentation:
- `docs/architecture/SYSTEM_ARCHITECTURE.md` - System architecture
- `docs/PRD.md` - Product requirements
- `CLAUDE.md` - Development guide
