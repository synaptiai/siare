---
layout: default
title: Configuration
nav_order: 3
---

# Configuration Reference

Complete reference for configuring SIARE. All settings can be specified via:

1. **YAML config file** (`config.yaml`)
2. **Environment variables** (take precedence)
3. **Python code** (for programmatic configuration)

---

## Quick Setup

```bash
# Copy the example configuration
cp config.example.yaml config.yaml

# Set required environment variables
export OPENAI_API_KEY="sk-your-key-here"

# Or for Anthropic
export ANTHROPIC_API_KEY="sk-ant-your-key-here"
```

---

## Configuration File Structure

```yaml
# config.yaml
llm_provider:
  type: "openai"
  default_model: "gpt-4o-mini"

tool_adapters:
  vector_search:
    backend: "chroma"

evolution:
  default_budget:
    maxEvaluations: 1000

storage:
  base_path: "./data"

logging:
  level: "INFO"
```

---

## LLM Provider

### OpenAI

```yaml
llm_provider:
  type: "openai"
  api_key: "${OPENAI_API_KEY}"  # Can use env var
  default_model: "gpt-4o-mini"
  judge_model: "gpt-4o"         # For evaluation
  architect_model: "gpt-4o"     # For mutations
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `type` | string | required | `"openai"`, `"anthropic"`, or `"ollama"` |
| `api_key` | string | env var | API key (prefer env var) |
| `default_model` | string | `"gpt-4o-mini"` | Default model for agents |
| `judge_model` | string | default_model | Model for LLM Judge metrics |
| `architect_model` | string | default_model | Model for Director mutations |

### Anthropic

```yaml
llm_provider:
  type: "anthropic"
  api_key: "${ANTHROPIC_API_KEY}"
  default_model: "claude-3-haiku-20240307"
```

### Ollama (Local)

```yaml
llm_provider:
  type: "ollama"
  base_url: "http://localhost:11434"
  default_model: "llama3.2"
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `OLLAMA_BASE_URL` | Ollama server URL (default: `http://localhost:11434`) |

---

## Tool Adapters

### Vector Search

```yaml
tool_adapters:
  vector_search:
    backend: "chroma"
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    dimension: 384
    backend_config:
      persist_directory: "./data/chroma"
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `backend` | string | `"memory"` | `"memory"`, `"chroma"`, `"pinecone"`, `"qdrant"` |
| `embedding_model` | string | `"all-MiniLM-L6-v2"` | Sentence transformer model |
| `dimension` | int | 384 | Embedding dimension |
| `backend_config` | object | `{}` | Backend-specific settings |

#### Backend-Specific Config

**ChromaDB:**
```yaml
backend_config:
  persist_directory: "./data/chroma"
  collection_name: "siare_docs"
```

**Pinecone:**
```yaml
backend_config:
  api_key: "${PINECONE_API_KEY}"
  environment: "us-east-1-aws"
  index_name: "siare"
```

**Qdrant:**
```yaml
backend_config:
  url: "http://localhost:6333"
  collection_name: "siare"
```

### SQL Database

```yaml
tool_adapters:
  sql:
    connection_string: "postgresql://user:pass@localhost:5432/db"
    dialect: "postgresql"
    read_only: true
    max_rows: 1000
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `connection_string` | string | required | Database connection URL |
| `dialect` | string | auto | `"postgresql"`, `"mysql"`, `"sqlite"` |
| `read_only` | bool | `true` | Prevent write operations |
| `max_rows` | int | 1000 | Max rows per query |

### Web Search

```yaml
tool_adapters:
  web_search:
    provider: "duckduckgo"
    max_results: 10
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `provider` | string | `"duckduckgo"` | `"google"`, `"bing"`, `"duckduckgo"`, `"serper"`, `"brave"` |
| `max_results` | int | 10 | Maximum search results |
| `api_key` | string | - | API key (for paid providers) |

---

## Evolution Settings

### Budget Limits

```yaml
evolution:
  default_budget:
    maxEvaluations: 1000
    maxLLMCalls: 5000
    maxCost: 100.0
    maxWallTime: 3600
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `maxEvaluations` | int | 1000 | Maximum task evaluations |
| `maxLLMCalls` | int | 5000 | Maximum LLM API calls |
| `maxCost` | float | 100.0 | Maximum cost in USD |
| `maxWallTime` | int | 3600 | Maximum runtime in seconds |

### QD Grid

```yaml
evolution:
  qd_grid:
    complexity_bins: 10
    embedding_dimensions: 2
    embedding_bins: 10
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `complexity_bins` | int | 10 | Bins for complexity dimension |
| `embedding_dimensions` | int | 2 | Number of embedding dimensions |
| `embedding_bins` | int | 10 | Bins per embedding dimension |

### Constraints

```yaml
evolution:
  default_constraints:
    maxRoles: 10
    maxEdges: 20
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `maxRoles` | int | 10 | Maximum agents in SOP |
| `maxEdges` | int | 20 | Maximum graph edges |

---

## Prompt Evolution Strategies

### TextGrad

Textual gradient descent with LLM-generated critiques.

```yaml
prompt_evolution:
  textgrad:
    learning_rate: 0.1
    backprop_depth: 3
    gradient_aggregation: "mean"
    model: "gpt-4o"
    temperature: 0.7
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `learning_rate` | float | 0.1 | How much to change prompts (0.0-1.0) |
| `backprop_depth` | int | 3 | Trace depth for gradient computation |
| `gradient_aggregation` | string | `"mean"` | `"mean"`, `"max"`, `"weighted"` |
| `model` | string | `"gpt-4"` | LLM for gradient computation |
| `temperature` | float | 0.7 | Creativity in gradient generation |

### EvoPrompt

Evolutionary algorithms for prompt populations.

```yaml
prompt_evolution:
  evoprompt:
    population_size: 10
    mutation_rate: 0.3
    crossover_rate: 0.7
    selection_method: "tournament"
    tournament_size: 3
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `population_size` | int | 10 | Population size |
| `mutation_rate` | float | 0.3 | Mutation probability |
| `crossover_rate` | float | 0.7 | Crossover probability |
| `selection_method` | string | `"tournament"` | `"tournament"`, `"roulette"` |
| `tournament_size` | int | 3 | Tournament selection size |

### MetaPrompt

LLM meta-analysis for targeted improvements.

```yaml
prompt_evolution:
  metaprompt:
    model: "gpt-4o"
    temperature: 0.8
    max_suggestions: 5
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `model` | string | `"gpt-4o"` | Model for meta-analysis |
| `temperature` | float | 0.8 | Creativity level |
| `max_suggestions` | int | 5 | Max improvement suggestions |

### Adaptive Strategy Selection

```yaml
prompt_evolution:
  adaptive:
    enabled: true
    strategy_weights:
      textgrad: 0.4
      evoprompt: 0.3
      metaprompt: 0.3
    auto_adjust: true
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enabled` | bool | `true` | Enable adaptive selection |
| `strategy_weights` | object | equal | Initial strategy weights |
| `auto_adjust` | bool | `true` | Auto-adjust weights based on success |

---

## Storage

```yaml
storage:
  base_path: "./data"
  configs: "./data/configs"
  genes: "./data/genes"
  logs: "./data/logs"
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `base_path` | string | `"./data"` | Base directory for all storage |
| `configs` | string | `"./data/configs"` | Config store path |
| `genes` | string | `"./data/genes"` | Gene pool storage path |
| `logs` | string | `"./data/logs"` | Log file directory |

---

## Logging

```yaml
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./data/logs/siare.log"
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `level` | string | `"INFO"` | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |
| `format` | string | standard | Log message format |
| `file` | string | - | Log file path (optional) |

---

## API Server

```yaml
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins:
    - "http://localhost:3000"
  rate_limit:
    requests_per_minute: 60
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `host` | string | `"0.0.0.0"` | Server bind address |
| `port` | int | 8000 | Server port |
| `workers` | int | 1 | Number of workers |
| `cors_origins` | list | `["*"]` | Allowed CORS origins |
| `rate_limit.requests_per_minute` | int | 60 | Rate limit per client |

---

## Circuit Breaker

```yaml
circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 30
  half_open_requests: 3
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `failure_threshold` | int | 5 | Failures before circuit opens |
| `recovery_timeout` | int | 30 | Seconds before half-open |
| `half_open_requests` | int | 3 | Test requests in half-open |

---

## Retry Configuration

```yaml
retry:
  max_retries: 3
  base_delay: 1.0
  max_delay: 60.0
  exponential_base: 2.0
```

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `max_retries` | int | 3 | Maximum retry attempts |
| `base_delay` | float | 1.0 | Initial delay in seconds |
| `max_delay` | float | 60.0 | Maximum delay in seconds |
| `exponential_base` | float | 2.0 | Exponential backoff base |

---

## Complete Example

```yaml
# config.yaml - Production configuration example

llm_provider:
  type: "openai"
  default_model: "gpt-4o-mini"
  judge_model: "gpt-4o"
  architect_model: "gpt-4o"

tool_adapters:
  vector_search:
    backend: "chroma"
    embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
    dimension: 384
    backend_config:
      persist_directory: "./data/chroma"

  sql:
    connection_string: "${DATABASE_URL}"
    read_only: true
    max_rows: 1000

  web_search:
    provider: "serper"
    max_results: 10

evolution:
  qd_grid:
    complexity_bins: 10
    embedding_dimensions: 2
    embedding_bins: 10

  default_budget:
    maxEvaluations: 1000
    maxLLMCalls: 5000
    maxCost: 100.0
    maxWallTime: 3600

  default_constraints:
    maxRoles: 10
    maxEdges: 20

prompt_evolution:
  adaptive:
    enabled: true
  textgrad:
    learning_rate: 0.1
    model: "gpt-4o"
  evoprompt:
    population_size: 10
    mutation_rate: 0.3

storage:
  base_path: "./data"
  configs: "./data/configs"
  genes: "./data/genes"
  logs: "./data/logs"

logging:
  level: "INFO"
  file: "./data/logs/siare.log"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  cors_origins:
    - "https://yourdomain.com"
  rate_limit:
    requests_per_minute: 60

circuit_breaker:
  failure_threshold: 5
  recovery_timeout: 30

retry:
  max_retries: 3
  base_delay: 1.0
```

---

## Python Configuration

For programmatic configuration:

```python
from siare.core.models import (
    ProcessConfig,
    EvolutionJob,
    MetricConfig,
    RetryConfig,
    CircuitBreakerConfig,
)


# Create retry config
retry_config = RetryConfig(
    max_retries=3,
    base_delay=1.0,
    max_delay=60.0,
    exponential_base=2.0,
)

# Create circuit breaker config
circuit_breaker_config = CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=30,
    half_open_requests=3,
)

# Create evolution job with budget
job = EvolutionJob(
    id="my_evolution",
    baseSopIds=["my_sop"],
    taskSet=my_tasks,
    metricsToOptimize=[...],
    budget=BudgetUsage(
        maxEvaluations=1000,
        maxLLMCalls=5000,
        maxCostUSD=100.0,
    ),
    maxGenerations=20,
    populationSize=5,
)
```

---

## Environment Variable Reference

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key | Yes (if using OpenAI) |
| `ANTHROPIC_API_KEY` | Anthropic API key | Yes (if using Anthropic) |
| `OLLAMA_BASE_URL` | Ollama server URL | No (default: localhost:11434) |
| `DATABASE_URL` | SQL database connection | No |
| `PINECONE_API_KEY` | Pinecone API key | No (if using Pinecone) |
| `SERPER_API_KEY` | Serper search API key | No (if using Serper) |
| `SIARE_CONFIG_PATH` | Custom config file path | No (default: ./config.yaml) |
| `SIARE_LOG_LEVEL` | Override log level | No |

---

## See Also

- [Quick Start](QUICKSTART.md) — Get running in 10 minutes
- [First Custom Pipeline](guides/first-custom-pipeline.md) — Build your first pipeline
- [Deployment Guide](DEPLOYMENT.md) — Production deployment
- [Troubleshooting](TROUBLESHOOTING.md) — Common issues
