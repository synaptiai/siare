# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SIARE (Self-Improving Agentic RAG Engine) is a Python framework for building self-evolving multi-agent RAG systems. It treats pipeline configurations as evolvable "genetic material" and uses Quality-Diversity optimization (MAP-Elites) to discover diverse, high-performing architectures automatically.

## Development Commands

```bash
# Install for development
pip install -e ".[dev,full]"

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=siare --cov-report=xml

# Run single test file
pytest tests/unit/test_models.py -v

# Run specific test
pytest tests/unit/test_models.py::test_process_config_validation -v

# Run only integration tests
pytest -m integration

# Linting
ruff check siare/
ruff fix siare/          # Auto-fix

# Type checking
pyright siare/

# CLI commands
siare init               # Initialize new project
siare evolve             # Run evolution
siare run "query"        # Execute pipeline
```

## Architecture

### Core Evolution Loop

```
ProcessConfig (SOP) → ExecutionEngine → ExecutionTrace
                                              ↓
                      EvaluationService → EvaluationVector
                                              ↓
                      DirectorService (single-turn) ──or── AgenticDirector (multi-turn)
                              ↓                                    ↓
                      SOPMutation (6 types)              vary() loop with tools
                              ↓                                    ↓
                      GenePool → QDGridManager (MAP-Elites)
                              ↓
                      EvolutionScheduler → next generation
                              ↓ (on stagnation)
                      SupervisorAgent → redirection directive
```

### Key Services (`siare/services/`)

- **ExecutionEngine**: Builds and runs multi-agent DAGs from ProcessConfig
- **DirectorService**: AI-driven mutation via Diagnostician (analyzes weaknesses) + Architect (generates fixes)
- **AgenticDirector**: Multi-turn variation operator with tool access — drop-in alternative to DirectorService
- **SupervisorAgent**: Analyzes evolutionary trajectory on stagnation and redirects exploration
- **KnowledgeBase**: Queryable store of RAG patterns, prompt engineering techniques, and prior run learnings
- **VariationToolRegistry**: 6 tools for the agentic director (inspect_trace, compare_sops, query_gene_pool, dry_run, validate_mutation, query_knowledge_base)
- **EvaluationService**: Multi-metric evaluation (LLM Judge, programmatic, runtime)
- **GenePool + QDGridManager**: Population management with Quality-Diversity optimization
- **EvolutionScheduler**: Orchestrates the evolution loop with 3 modes (single_turn, agentic, adaptive)

### Mutation Types

The framework supports 6 mutation types: `PROMPT_CHANGE`, `PARAM_TWEAK`, `ADD_ROLE`, `REMOVE_ROLE`, `REWIRE_GRAPH`, `CROSSOVER`

### Extensibility via Hooks (`siare/core/hooks.py`)

All major operations are extensible without modifying core code:
- `EvolutionHooks`: on_mutation_*, on_generation_complete
- `AgenticEvolutionHooks`: on_variation_start/iteration/complete, on_supervisor_redirect
- `ExecutionHooks`: on_execution_start, on_role_complete, on_execution_complete
- `EvaluationHooks`: on_evaluation_start, on_evaluation_complete
- `StorageHooks`: on_sop_saved, on_sop_loaded
- `LLMHooks`: on_llm_request, on_llm_response

### Data Models (`siare/core/models.py`)

Core Pydantic models:
- `ProcessConfig`: Complete multi-agent pipeline definition
- `RoleConfig`: Individual agent with prompts, tools, parameters
- `PromptGenome`: Evolvable prompt with mutation history
- `ExecutionTrace`: Step-by-step execution logs
- `EvaluationVector`: Multi-metric evaluation results
- `SOPMutation`: Proposed change with rationale
- `AgenticVariationConfig`: Configuration for agentic evolution modes
- `InnerLoopBudget`: Budget tracking for agentic variation sessions
- `VariationResult`: Result of an agentic variation session
- `SupervisorDirective`: Exploration redirection from the supervisor

### Adapters (`siare/adapters/`)

Tool integrations with common interface: Qdrant, Pinecone, ChromaDB (vector), DuckDuckGo (web), Wikipedia

## Code Style

- **Line length**: 100 characters
- **Type hints**: Required (pyright strict mode)
- **Models**: Use Pydantic v2 with `model_config = ConfigDict(...)`
- **Async**: Use `pytest-asyncio` for async tests
- **Test markers**: `@pytest.mark.integration` for integration tests, `@pytest.mark.slow` for slow tests

## Package Structure

```
siare/
├── __init__.py          # Public exports (builders, models, hooks)
├── builders.py          # High-level API (pipeline, role, edge, task)
├── cli.py               # Click-based CLI
├── core/                # Data models, hooks, config
├── services/            # Core business logic
├── adapters/            # External tool integrations
├── benchmarks/          # Evaluation datasets and runners
└── utils/               # Sampling, statistics, diffing utilities
```

## Python Version

Requires Python 3.12+
