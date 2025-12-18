---
layout: default
title: Contributing
nav_order: 10
---

# Contributing to SIARE

Thank you for your interest in contributing to SIARE! This guide will help you get started.

## Quick Start

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/siare.git
cd siare

# Set up development environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Verify setup
pytest

# Create a branch for your work
git checkout -b feature/your-feature-name
```

---

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- An OpenAI API key (for integration tests)

### Full Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/siare.git
cd siare

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Set up pre-commit hooks
pre-commit install

# Set environment variables
export OPENAI_API_KEY=your_key_here
```

### Verify Installation

```bash
# Run all tests
pytest

# Run linting
ruff check .

# Run type checking
mypy siare/

# Format code
ruff format .
```

---

## Code Quality Standards

### Style Guide

We use:
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing

```bash
# Check everything
ruff check .
mypy siare/
pytest
```

### Key Principles

1. **Type hints everywhere**
   ```python
   # Good
   def execute(self, task: Task) -> ExecutionTrace:
       ...

   # Bad
   def execute(self, task):
       ...
   ```

2. **Models in models.py**
   - All Pydantic data structures go in `siare/core/models.py`
   - No data structures scattered across modules

3. **Dependency injection in API**
   ```python
   # Good
   @app.get("/sops")
   def get_sops(gene_pool: GenePool = Depends(get_gene_pool)):
       return gene_pool.list_sops()

   # Bad - global state
   _gene_pool = GenePool()

   @app.get("/sops")
   def get_sops():
       return _gene_pool.list_sops()
   ```

4. **No silent fallbacks to mocks**
   ```python
   # Bad - silent mock
   def evaluate(self):
       if self.provider:
           return self.provider.call()
       return {"score": 0.85}  # Silent fake!

   # Good - fail loudly
   def evaluate(self):
       if not self.provider:
           raise RuntimeError("LLM provider required")
       return self.provider.call()
   ```

5. **Validate constraints before mutations**
   ```python
   errors = director.validate_constraints(sop, constraints)
   if errors:
       raise ValueError(f"Constraint violations: {errors}")
   ```

---

## Testing

### Test Organization

```
tests/
├── unit/           # Single component, mocked dependencies
├── integration/    # Multiple components, real interactions
└── conftest.py     # Shared fixtures
```

### Running Tests

```bash
# All tests
pytest

# Single file
pytest tests/unit/test_director.py -v

# Specific test
pytest tests/unit/test_director.py::test_validate_prompt_constraints -v

# With coverage
pytest --cov=siare --cov-report=html
```

### Writing Tests

Name tests as: `test_<component>_<scenario>_<expected_outcome>`

```python
def test_director_diagnose_identifies_weak_metric():
    """Director correctly identifies the weakest metric in evaluation."""
    # Arrange
    evaluation = EvaluationVector(
        metrics={"accuracy": 0.9, "latency": 0.3, "cost": 0.8}
    )

    # Act
    diagnosis = director.diagnose(evaluation)

    # Assert
    assert diagnosis.weakest_metric == "latency"
    assert "latency" in diagnosis.mutation_targets
```

### Integration Tests

Mark tests that need external services:

```python
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY required"
)
def test_execution_engine_with_real_llm():
    ...
```

---

## Pull Request Process

### Before Submitting

1. **Run all checks**
   ```bash
   ruff check .
   ruff format .
   mypy siare/
   pytest
   ```

2. **Update documentation** if adding features

3. **Add tests** for new functionality

4. **Write a clear commit message**
   ```
   feat(director): add TextGrad prompt optimization strategy

   - Implement TextGrad strategy for prompt evolution
   - Add gradient-based optimization using LLM critiques
   - Include unit tests for edge cases
   ```

### Commit Message Format

```
<type>(<scope>): <short description>

<body - what and why>

<footer - breaking changes, issue refs>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change that neither fixes nor adds
- `test`: Adding missing tests
- `chore`: Build process or auxiliary tools

### PR Template

Your PR should include:

1. **Summary**: What does this change?
2. **Motivation**: Why is this change needed?
3. **Test plan**: How was this tested?
4. **Breaking changes**: Does this break existing APIs?

---

## Adding New Features

### Adding a New Mutation Type

1. **Add to enum** in `siare/core/models.py`:
   ```python
   class MutationType(str, Enum):
       PROMPT_CHANGE = "prompt_change"
       # ... existing types
       NEW_TYPE = "new_type"  # Add your type
   ```

2. **Implement handler** in `siare/services/director.py`:
   ```python
   def _apply_new_type_mutation(self, sop: ProcessConfig, target: str) -> ProcessConfig:
       """Apply NEW_TYPE mutation."""
       # Implementation
       return mutated_sop
   ```

3. **Add to dispatch** in `mutate_sop()`:
   ```python
   if mutation.type == MutationType.NEW_TYPE:
       return self._apply_new_type_mutation(sop, mutation.target)
   ```

4. **Write tests** in `tests/unit/test_director.py`

5. **Document** in `docs/reference/mutation-operators.md`

### Adding a New Tool Adapter

1. **Create adapter** in `siare/adapters/`:
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

2. **Add tests** in `tests/unit/test_adapters.py`

3. **Document** in `docs/guides/custom-extensions.md`

### Adding a New Metric

1. **Define metric function**:
   ```python
   def my_metric(trace: ExecutionTrace, task_data: dict) -> float:
       return 0.85  # Score 0-1
   ```

2. **Register with service**:
   ```python
   evaluation_service.register_metric_function("my_metric", my_metric)
   ```

3. **Add tests**

4. **Document** in `docs/guides/custom-extensions.md`

### Adding a Prompt Evolution Strategy

1. **Extend base class** in `siare/services/prompt_evolution/strategies/`:
   ```python
   from siare.services.prompt_evolution.strategies.base import PromptEvolutionStrategy

   class MyStrategy(PromptEvolutionStrategy):
       def optimize(self, prompt: str, failure_context: dict) -> str:
           improved_prompt = self._apply_improvements(prompt, failure_context)
           return improved_prompt

       def select_strategy(self, failure_patterns: list[str]) -> bool:
           return "my_pattern" in failure_patterns
   ```

2. **Register with factory**:
   ```python
   factory = PromptOptimizationFactory()
   factory.register_strategy("my_strategy", MyStrategy)
   ```

3. **Add tests and documentation**

---

## Documentation

### Writing Documentation

- Use clear, concise language
- Include code examples
- Add cross-references with "See Also" sections
- Update the docs/README.md navigation hub

### Documentation Structure

```
docs/
├── README.md           # Navigation hub
├── QUICKSTART.md       # Getting started
├── CONFIGURATION.md    # Complete reference
├── TROUBLESHOOTING.md  # Common issues
├── DEPLOYMENT.md       # Production deployment
├── concepts/           # How things work
├── guides/             # Step-by-step tutorials
├── reference/          # API and operator reference
├── production/         # Security, cost, monitoring
└── examples/           # Worked examples
```

### Building Docs Locally

```bash
# If using MkDocs (optional)
pip install mkdocs-material
mkdocs serve
```

---

## Getting Help

### Questions

- Open a GitHub issue with the "question" label
- Check existing issues and docs first

### Reporting Bugs

Include:
1. SIARE version
2. Python version
3. Steps to reproduce
4. Expected vs actual behavior
5. Error messages/logs

### Feature Requests

Include:
1. What you're trying to accomplish
2. Current workaround (if any)
3. Proposed solution
4. Alternatives considered

---

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards others

### Enforcement

Unacceptable behavior may be reported to the maintainers. All complaints will be reviewed and investigated.

---

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License.

---

## Recognition

Contributors are recognized in:
- GitHub contributors page
- Release notes for significant contributions

Thank you for contributing to SIARE!
