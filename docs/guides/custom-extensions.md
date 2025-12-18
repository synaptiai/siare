---
layout: default
title: Custom Extensions
parent: Guides
nav_order: 3
---

# Custom Extensions

Guide to extending SIARE with custom metrics, tools, and constraints.

## Overview

SIARE is designed for extensibility. You can add:

| Extension | Purpose |
|-----------|---------|
| **Custom Metrics** | Domain-specific evaluation |
| **Tool Adapters** | Connect external services |
| **Constraints** | Enforce domain rules |
| **Prompt Strategies** | Custom evolution approaches |

---

## Custom Metrics

### Programmatic Metrics

Define code-based evaluation functions:

```python
from typing import Dict, Any
from siare.core.models import ExecutionTrace
from siare.services.evaluation_service import EvaluationService


def citation_accuracy(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    """Check if citations in the answer are accurate.

    Args:
        trace: Execution trace containing role outputs
        task_data: Task data with ground truth

    Returns:
        Score between 0.0 and 1.0
    """
    answer = trace.outputs.get("answer", "")
    documents = trace.outputs.get("documents", [])

    # Extract citations from answer (e.g., [DocA, Section 1])
    import re
    citations = re.findall(r'\[([^\]]+)\]', answer)

    if not citations:
        return 0.0

    # Verify each citation exists in documents
    valid_count = 0
    for citation in citations:
        for doc in documents:
            if citation in str(doc):
                valid_count += 1
                break

    return valid_count / len(citations)


def answer_length(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    """Check if answer length is within acceptable range.

    Returns 1.0 if within range, scaled down otherwise.
    """
    answer = trace.outputs.get("answer", "")
    word_count = len(answer.split())

    min_words = task_data.get("min_words", 50)
    max_words = task_data.get("max_words", 500)

    if min_words <= word_count <= max_words:
        return 1.0
    elif word_count < min_words:
        return word_count / min_words
    else:
        return max_words / word_count


# Register metrics with evaluation service
eval_service = EvaluationService(llm_provider=provider)
eval_service.register_metric_function("citation_accuracy", citation_accuracy)
eval_service.register_metric_function("answer_length", answer_length)
```

### LLM Judge Metrics

Create custom LLM-based evaluators:

```python
from siare.core.models import MetricConfig, MetricType, AggregationMethod

# Define the judge prompt
judge_prompt = """
You are evaluating the quality of a legal document summary.

ORIGINAL DOCUMENT: {document}
SUMMARY: {answer}

Evaluate on these criteria:
1. ACCURACY (40%): Does the summary accurately represent the document?
2. COMPLETENESS (30%): Are all key points included?
3. CLARITY (20%): Is the summary clear and well-organized?
4. CONCISENESS (10%): Is it appropriately brief without losing content?

Calculate a weighted score from 0.0 to 1.0.

Return ONLY a JSON object:
{"score": 0.85, "reasoning": "Brief explanation..."}
"""

# Register the judge prompt
config_store.add_prompt("legal_summary_judge", judge_prompt)

# Create metric config
legal_summary_metric = MetricConfig(
    id="legal_summary_quality",
    type=MetricType.LLM_JUDGE,
    model="gpt-4o-mini",
    promptRef="legal_summary_judge",
    inputs=["document", "answer"],
    aggregationMethod=AggregationMethod.MEAN,
    weight=0.5,
)
```

### Composite Metrics

Combine multiple metrics:

```python
def composite_quality_score(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    """Combine multiple quality checks into one score."""
    scores = []

    # Check 1: Has citations
    answer = trace.outputs.get("answer", "")
    has_citations = 1.0 if "[" in answer and "]" in answer else 0.0
    scores.append(("citations", has_citations, 0.3))

    # Check 2: Appropriate length
    words = len(answer.split())
    length_ok = 1.0 if 50 <= words <= 500 else 0.5
    scores.append(("length", length_ok, 0.2))

    # Check 3: Contains required keywords
    required = task_data.get("required_terms", [])
    if required:
        found = sum(1 for term in required if term.lower() in answer.lower())
        keyword_score = found / len(required)
    else:
        keyword_score = 1.0
    scores.append(("keywords", keyword_score, 0.5))

    # Weighted average
    total_weight = sum(w for _, _, w in scores)
    weighted_sum = sum(s * w for _, s, w in scores)

    return weighted_sum / total_weight
```

---

## Tool Adapters

### Creating a Custom Adapter

Extend the `ToolAdapter` base class:

```python
from typing import Dict, List, Any
from siare.adapters.base import ToolAdapter, register_adapter


@register_adapter("custom_database")
class CustomDatabaseAdapter(ToolAdapter):
    """Adapter for querying a custom database."""

    def __init__(self, connection_string: str, **kwargs):
        super().__init__(**kwargs)
        self.connection_string = connection_string
        self.connection = None

    def initialize(self) -> None:
        """Initialize the database connection."""
        import your_db_library
        self.connection = your_db_library.connect(self.connection_string)
        self.is_initialized = True

    def execute(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a database query.

        Args:
            inputs: Must contain 'query' key

        Returns:
            Dict with 'results' key containing query results
        """
        if not self.is_initialized:
            raise RuntimeError("Adapter not initialized")

        query = inputs.get("query")
        if not query:
            raise ValueError("Missing required 'query' input")

        results = self.connection.execute(query)

        return {
            "results": results,
            "row_count": len(results),
        }

    def validate_inputs(self, inputs: Dict[str, Any]) -> List[str]:
        """Validate inputs before execution.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if "query" not in inputs:
            errors.append("Missing required 'query' input")

        if "query" in inputs and not isinstance(inputs["query"], str):
            errors.append("'query' must be a string")

        return errors

    def cleanup(self) -> None:
        """Clean up resources."""
        if self.connection:
            self.connection.close()
            self.connection = None
```

### Using the Adapter

```python
from siare.adapters import ToolRegistry

# Register and initialize
registry = ToolRegistry()
registry.register("custom_database", CustomDatabaseAdapter(
    connection_string="your://connection/string"
))

# Use in a role
role = RoleConfig(
    id="data_retriever",
    model="gpt-4o-mini",
    tools=["custom_database"],
    promptRef="data_retriever_prompt",
    inputs=[RoleInput(from_="user_input")],
    outputs=["data_results"],
)
```

### Built-in Adapters

SIARE includes several adapters:

| Adapter | Purpose |
|---------|---------|
| `vector_search` | Semantic search over embeddings |
| `sql` | SQL database queries |
| `web_search` | Web search APIs |

```python
from siare.adapters import VectorSearchAdapter, SQLAdapter

# Vector search
vector_adapter = VectorSearchAdapter(
    backend="chroma",
    embedding_model="all-MiniLM-L6-v2",
    persist_directory="./data/vectors",
)

# SQL
sql_adapter = SQLAdapter(
    connection_string="sqlite:///./data/app.db",
    read_only=True,
    max_rows=1000,
)
```

---

## Custom Constraints

### Prompt Constraints

Protect parts of prompts from evolution:

```python
from siare.core.models import PromptConstraints, RolePrompt

prompt = RolePrompt(
    id="safety_prompt",
    content="""
## ROLE
You are a helpful assistant.

## SAFETY RULES (PROTECTED)
1. Never provide harmful information
2. Always recommend professional consultation
3. Flag suspicious requests

## TASK
[Task description that can be evolved]

## OUTPUT FORMAT
[Format that can be evolved]
""",
    constraints=PromptConstraints(
        mustNotChange=[
            "SAFETY RULES (PROTECTED)",
            "Never provide harmful",
            "Always recommend professional",
            "Flag suspicious",
        ],
        allowedChanges=["TASK", "OUTPUT FORMAT"],
        maxLength=4000,
    ),
)
```

### Topology Constraints

Limit pipeline structure:

```python
from siare.core.models import EvolutionConstraints

constraints = EvolutionConstraints(
    # Role limits
    maxRoles=10,
    minRoles=2,
    requiredRoles=["retriever", "answerer"],

    # Model limits
    allowedModels=["gpt-4o-mini", "gpt-4o"],

    # Cost limits
    maxCostPerTask=0.10,

    # Safety
    requireSafetyCheck=True,
)
```

### Custom Constraint Validators

Create domain-specific validation:

```python
from typing import List, Optional
from siare.core.models import ProcessConfig, ConstraintViolation


def validate_medical_constraints(sop: ProcessConfig) -> List[ConstraintViolation]:
    """Validate medical domain constraints.

    Returns:
        List of constraint violations
    """
    violations = []

    # Check for required disclaimer role
    role_ids = {role.id for role in sop.roles}
    if "disclaimer" not in role_ids:
        violations.append(ConstraintViolation(
            constraint_type="required_role",
            violation_description="Medical pipelines must include disclaimer role",
            role_id="disclaimer",
            severity="error",
        ))

    # Check all prompts contain safety language
    for role in sop.roles:
        # Would need to load prompt content here
        pass

    return violations


# Register with director
from siare.services.director import DirectorService

class MedicalDirector(DirectorService):
    def validate_constraints(self, sop: ProcessConfig, constraints) -> List[str]:
        # Run standard validation
        errors = super().validate_constraints(sop, constraints)

        # Add medical-specific validation
        violations = validate_medical_constraints(sop)
        errors.extend([v.violation_description for v in violations])

        return errors
```

---

## Custom Prompt Evolution Strategies

### Creating a Strategy

```python
from typing import Dict, Any, Optional
from siare.services.prompt_evolution.strategies.base import PromptEvolutionStrategy
from siare.core.models import PromptOptimizationStrategyType


class DomainExpertStrategy(PromptEvolutionStrategy):
    """Domain-specific prompt evolution using expert knowledge."""

    strategy_type = PromptOptimizationStrategyType.CUSTOM

    def __init__(self, domain: str, expert_rules: Dict[str, str], **kwargs):
        super().__init__(**kwargs)
        self.domain = domain
        self.expert_rules = expert_rules

    def optimize(
        self,
        prompt: str,
        failure_context: Dict[str, Any],
        constraints: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Apply domain expert rules to improve prompt.

        Args:
            prompt: Current prompt content
            failure_context: Information about failures
            constraints: Optional constraints to respect

        Returns:
            Improved prompt
        """
        improved = prompt

        # Apply domain-specific rules
        for pattern, replacement in self.expert_rules.items():
            if pattern in failure_context.get("failure_patterns", []):
                improved = self._apply_rule(improved, pattern, replacement)

        return improved

    def _apply_rule(self, prompt: str, pattern: str, rule: str) -> str:
        """Apply a specific domain rule to the prompt."""
        # Add rule as instruction if not present
        if rule not in prompt:
            # Find INSTRUCTIONS section and add rule
            if "## INSTRUCTIONS" in prompt:
                prompt = prompt.replace(
                    "## INSTRUCTIONS",
                    f"## INSTRUCTIONS\n- {rule}"
                )
            else:
                prompt = f"{prompt}\n\nADDITIONAL RULE: {rule}"

        return prompt

    def should_handle(self, failure_patterns: List[str]) -> bool:
        """Check if this strategy should handle these failures."""
        domain_patterns = [p for p in failure_patterns if self.domain in p.lower()]
        return len(domain_patterns) > 0


# Register the strategy
from siare.services.prompt_evolution import PromptOptimizationFactory

factory = PromptOptimizationFactory()
factory.register_strategy(
    "domain_expert",
    DomainExpertStrategy(
        domain="legal",
        expert_rules={
            "missing_citation": "Always cite the specific section and page number",
            "ambiguous_term": "Define legal terms before using them",
            "incomplete_analysis": "Address all parties mentioned in the document",
        }
    )
)
```

### Combining Strategies

```python
from siare.services.prompt_evolution import AdaptiveStrategySelector

# Configure adaptive selection
selector = AdaptiveStrategySelector(
    strategies={
        "textgrad": TextGradStrategy(learning_rate=0.1),
        "evoprompt": EvoPromptStrategy(population_size=10),
        "domain_expert": DomainExpertStrategy(domain="legal", expert_rules={...}),
    },
    initial_weights={
        "textgrad": 0.4,
        "evoprompt": 0.3,
        "domain_expert": 0.3,
    },
    auto_adjust=True,  # Adjust weights based on success
)

# Selector automatically chooses best strategy
strategy = selector.select(failure_patterns)
improved_prompt = strategy.optimize(prompt, failure_context)
```

---

## Custom Domain Packages

### Creating a Domain Package

Domain packages bundle domain-specific SOPs, prompts, metrics, and constraints. Use a factory function pattern:

```python
# siare/domains/legal_domain.py

from siare.core.models import (
    BudgetLimit,
    DomainConfig,
    DomainPackage,
    EvolutionConstraints,
    GraphEdge,
    MetricConfig,
    MetricType,
    ProcessConfig,
    PromptConstraints,
    PromptGenome,
    RoleConfig,
    RoleInput,
    RolePrompt,
    ToolConfig,
    ToolType,
)


def create_legal_domain_package() -> DomainPackage:
    """
    Create pre-configured Legal domain package.

    Returns a DomainPackage with:
    - Default SOP for legal document analysis
    - Domain-specific prompts with constraints
    - Metrics for accuracy, citation quality, completeness
    - Evolution constraints (e.g., mandatory disclaimer role)
    """

    # SOP Template
    legal_sop = ProcessConfig(
        id="legal_rag_default",
        version="1.0.0",
        description="Legal document analysis with citation verification",
        models={"default": "gpt-4o", "retrieval": "gpt-4o-mini"},
        tools=["vector_search", "case_database"],
        roles=[
            RoleConfig(
                id="retriever",
                model="retrieval",
                tools=["vector_search"],
                promptRef="legal_retriever_prompt",
                inputs=[RoleInput.model_validate({"from": "user_input"})],
                outputs=["relevant_documents", "case_citations"],
            ),
            RoleConfig(
                id="analyzer",
                model="default",
                tools=[],
                promptRef="legal_analyzer_prompt",
                inputs=[RoleInput.model_validate({"from": ["user_input", "retriever"]})],
                outputs=["legal_analysis", "precedent_summary"],
            ),
            RoleConfig(
                id="disclaimer",
                model="retrieval",
                tools=[],
                promptRef="legal_disclaimer_prompt",
                inputs=[RoleInput.model_validate({"from": "analyzer"})],
                outputs=["final_response", "disclaimer_text"],
            ),
        ],
        graph=[
            GraphEdge(from_="retriever", to="analyzer"),
            GraphEdge(from_="analyzer", to="disclaimer"),
        ],
    )

    # Prompts with domain-specific constraints
    legal_prompts = PromptGenome(
        id="legal_prompts_default",
        version="1.0.0",
        rolePrompts={
            "legal_retriever_prompt": RolePrompt(
                id="legal_retriever_prompt",
                content="""You are a legal research specialist.

Search for relevant case law, statutes, and legal precedents.
Always cite sources with proper legal citations (e.g., 410 U.S. 113).

Provide:
- relevant_documents: Key documents with citations
- case_citations: Properly formatted legal citations
""",
                constraints=PromptConstraints(
                    mustNotChange=["legal citations", "source verification"],
                    domainTips=["Use Bluebook citation format", "Include jurisdiction"],
                ),
            ),
            "legal_analyzer_prompt": RolePrompt(
                id="legal_analyzer_prompt",
                content="""You are a legal analyst. Analyze the retrieved documents
and provide a comprehensive legal analysis.

IMPORTANT: Distinguish between binding precedent and persuasive authority.
""",
                constraints=PromptConstraints(
                    mustNotChange=["precedent analysis", "jurisdiction awareness"],
                ),
            ),
            "legal_disclaimer_prompt": RolePrompt(
                id="legal_disclaimer_prompt",
                content="""Add appropriate legal disclaimers to the response.

MUST include:
- "This is not legal advice"
- "Consult a licensed attorney"
- Jurisdiction limitations
""",
                constraints=PromptConstraints(
                    mustNotChange=["disclaimer", "not legal advice"],
                ),
            ),
        },
    )

    # Domain-specific metrics
    metrics = [
        MetricConfig(
            id="citation_accuracy",
            type=MetricType.LLM_JUDGE,
            model="gpt-4o",
            promptRef="citation_accuracy",
            inputs=["case_citations", "legal_analysis"],
            weight=0.35,
        ),
        MetricConfig(
            id="legal_reasoning",
            type=MetricType.LLM_JUDGE,
            model="gpt-4o",
            promptRef="legal_reasoning",
            inputs=["legal_analysis", "precedent_summary"],
            weight=0.35,
        ),
        MetricConfig(
            id="completeness",
            type=MetricType.LLM_JUDGE,
            model="gpt-4o-mini",
            promptRef="completeness",
            inputs=["final_response"],
            weight=0.20,
        ),
        MetricConfig(
            id="cost",
            type=MetricType.RUNTIME,
            fnRef="calculate_cost",
            inputs=[],
            weight=0.10,
        ),
    ]

    # Evolution constraints
    domain_config = DomainConfig(
        recommendedConstraints=EvolutionConstraints(
            budgetLimit=BudgetLimit(
                maxEvaluations=100,
                maxLLMCalls=500,
                maxCost=50.0,
            ),
            maxRoles=6,
            mandatoryRoles=["disclaimer"],  # Always require disclaimer
            allowedTools=["vector_search", "case_database"],
        ),
    )

    # Assemble and return the package
    return DomainPackage(
        id="legal_domain",
        name="Legal Document Analysis",
        version="1.0.0",
        description="Domain package for legal RAG with citation verification and mandatory disclaimers",
        sopTemplates=["legal_rag_default"],
        promptGenomes=["legal_prompts_default"],
        metricConfigs=["citation_accuracy", "legal_reasoning", "completeness", "cost"],
        domainConfig=domain_config,
        maintainer="Your Team",
        # Internal references (collected into package)
        _sop=legal_sop,
        _prompts=legal_prompts,
        _metrics=metrics,
    )
```

### Using a Domain Package

```python
from siare.domains.legal_domain import create_legal_domain_package
from siare.services.config_store import ConfigStore
from siare.services.scheduler import EvolutionScheduler

# Create the domain package
package = create_legal_domain_package()

# Access package components
print(f"Domain: {package.name} v{package.version}")
print(f"Description: {package.description}")

# Register with config store
config_store = ConfigStore()
config_store.register_domain_package(package)

# Start evolution with domain configuration
scheduler = EvolutionScheduler(config_store=config_store)
job = scheduler.start_evolution_job(
    sop_id="legal_rag_default",
    tasks=[{"query": "What are the precedents for software patent disputes?"}],
    constraints=package.domainConfig.recommendedConstraints,
)
```

### See Also

For a complete, production-ready domain package example, see:
- `siare/domains/clinical_trials_package.py` - Full implementation with 6 specialist agents
- [Clinical Trials Walkthrough](../examples/clinical-trials-walkthrough.md) - Step-by-step guide

---

## Best Practices

### 1. Validate Early

Check inputs before processing:

```python
def my_metric(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    # Validate inputs
    if "answer" not in trace.outputs:
        raise ValueError("Expected 'answer' in trace outputs")

    answer = trace.outputs["answer"]
    if not isinstance(answer, str):
        raise TypeError(f"Expected str, got {type(answer)}")

    # Process...
```

### 2. Handle Errors Gracefully

Return sensible defaults on failure:

```python
def citation_check(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    try:
        answer = trace.outputs.get("answer", "")
        # ... check citations ...
        return score
    except Exception as e:
        logger.warning(f"Citation check failed: {e}")
        return 0.5  # Neutral score on error
```

### 3. Document Extensions

```python
def domain_specific_metric(trace: ExecutionTrace, task_data: Dict[str, Any]) -> float:
    """Calculate domain-specific quality score.

    This metric checks for:
    1. Proper citation format [Author, Year]
    2. Required terminology usage
    3. Disclaimer presence

    Args:
        trace: Execution trace with 'answer' output
        task_data: Must contain 'required_terms' list

    Returns:
        Score from 0.0 (poor) to 1.0 (excellent)

    Raises:
        ValueError: If required inputs are missing

    Example:
        >>> trace = ExecutionTrace(outputs={"answer": "The study [Smith, 2024]..."})
        >>> task = {"required_terms": ["methodology", "results"]}
        >>> score = domain_specific_metric(trace, task)
        >>> assert 0.0 <= score <= 1.0
    """
    # Implementation...
```

### 4. Test Extensions

```python
# tests/test_custom_metrics.py

import pytest
from siare.core.models import ExecutionTrace


def test_citation_accuracy_with_valid_citations():
    trace = ExecutionTrace(
        outputs={
            "answer": "According to [DocA, §1.2], the term is 30 days.",
            "documents": [{"id": "DocA", "content": "...§1.2..."}],
        }
    )
    task_data = {}

    score = citation_accuracy(trace, task_data)

    assert score == 1.0


def test_citation_accuracy_with_no_citations():
    trace = ExecutionTrace(
        outputs={
            "answer": "The term is 30 days.",
            "documents": [{"id": "DocA", "content": "..."}],
        }
    )
    task_data = {}

    score = citation_accuracy(trace, task_data)

    assert score == 0.0
```

---

## See Also

- [Configuration Reference](../CONFIGURATION.md) — Configure extensions
- [Evolution Lifecycle](../concepts/evolution-lifecycle.md) — How extensions are used
- [First Custom Pipeline](first-custom-pipeline.md) — Build with extensions
- [Mutation Operators](../reference/mutation-operators.md) — Extend mutations
