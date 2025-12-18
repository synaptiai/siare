---
layout: default
title: Troubleshooting
nav_order: 4
---

# SIARE Troubleshooting Guide

This guide covers common issues and their solutions when working with SIARE (Self-Improving Agentic RAG Engine).

## First-Time Setup Issues

### Quick Checklist

If you just installed SIARE and it's not working, check these first:

| Check | Command | Expected |
|-------|---------|----------|
| Python version | `python --version` | 3.10+ |
| In virtual env? | `which python` | Points to `venv/bin/python` |
| Dependencies installed? | `pip show siare` | Shows package info |
| Ollama running? | `curl localhost:11434/api/tags` | JSON response |
| OpenAI key set? | `echo $OPENAI_API_KEY` | `sk-...` |

### Most Common First-Time Issues

1. **"No module named 'siare'"** → You're not in the virtual environment
   ```bash
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **"Ollama not running"** → Ollama server not started
   ```bash
   ollama serve  # Start server
   ollama pull llama3.2  # Pull model
   ```

3. **"OPENAI_API_KEY not set"** → Environment variable missing
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

4. **"Python 3.9 is not supported"** → Wrong Python version
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Quick Diagnostics

### Health Check Commands

Run these commands to quickly diagnose your SIARE setup:

```bash
# Check API server health (if running)
curl http://localhost:8000/health

# Check Ollama connection
curl http://localhost:11434/api/tags

# Verify environment and dependencies
python -c "from siare.demos import validate_environment; import json; print(json.dumps(validate_environment(), indent=2))"

# Test Python imports
python -c "from siare.services import DirectorService, ExecutionEngine; print('Core imports: OK')"

# Check Python version
python --version  # Must be 3.10+

# List installed packages
pip list | grep -E "(openai|anthropic|fastapi|pydantic)"
```

### Quick Diagnostic Script

Save this as `diagnose.py` and run it to check your environment:

```python
#!/usr/bin/env python
"""SIARE diagnostic script."""
import os
import sys
from siare.demos import validate_environment

def run_diagnostics():
    """Run comprehensive diagnostics."""
    print("=== SIARE Environment Diagnostics ===\n")

    # Check Python version
    print(f"Python version: {sys.version}")
    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ required")
        return
    print("Python version: OK\n")

    # Check environment
    env = validate_environment()

    print("LLM Provider Status:")
    print(f"  Ollama:  {'OK' if env['ollama_available'] else 'FAILED'}")
    print(f"  OpenAI:  {'OK' if env['openai_available'] else 'FAILED'}")

    if env['errors']:
        print("\nErrors detected:")
        for err in env['errors']:
            print(f"  - {err}")
    else:
        print("\nAll checks passed!")

    # Check environment variables
    print("\nEnvironment Variables:")
    print(f"  OPENAI_API_KEY:     {'Set' if os.getenv('OPENAI_API_KEY') else 'Not set'}")
    print(f"  ANTHROPIC_API_KEY:  {'Set' if os.getenv('ANTHROPIC_API_KEY') else 'Not set'}")
    print(f"  OLLAMA_BASE_URL:    {os.getenv('OLLAMA_BASE_URL', 'Not set (using default)')}")
    print(f"  SIARE_LOG_LEVEL:    {os.getenv('SIARE_LOG_LEVEL', 'Not set (using default)')}")

if __name__ == "__main__":
    run_diagnostics()
```

Run it with:
```bash
python diagnose.py
```

---

## Common Issues

### LLM Provider Issues

#### Ollama Connection Failed

**Symptoms:**
- `Connection refused` errors
- `requests.exceptions.ConnectionError`
- Timeout errors when using Ollama provider
- Error: "Ollama not running at http://localhost:11434"

**Root Causes:**
- Ollama server not running
- Wrong port configuration
- Firewall blocking local connections
- Ollama not installed

**Solutions:**

1. **Start Ollama server:**
   ```bash
   # Start Ollama in foreground
   ollama serve

   # Or start as background service (macOS)
   brew services start ollama
   ```

2. **Check if Ollama is running:**
   ```bash
   # Test the API endpoint
   curl http://localhost:11434/api/tags

   # Should return JSON with available models
   ```

3. **Verify port configuration:**
   ```bash
   # Check if Ollama is listening on correct port
   lsof -i :11434  # macOS/Linux

   # Or check process
   ps aux | grep ollama
   ```

4. **Pull required model:**
   ```bash
   # Pull the default model
   ollama pull llama3.2

   # List available models
   ollama list

   # Test model
   ollama run llama3.2 "Hello"
   ```

5. **Custom Ollama URL:**
   ```bash
   # If Ollama runs on different host/port
   export OLLAMA_BASE_URL="http://custom-host:11434"
   ```

6. **Check firewall:**
   ```bash
   # macOS
   sudo /usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate

   # Linux (ufw)
   sudo ufw status
   ```

#### OpenAI API Errors

**Symptoms:**
- `401 Unauthorized`
- `429 Rate Limited`
- `openai.APIError: Invalid API key`
- "OpenAI API key not provided"

**Solutions:**

1. **Verify API key is set:**
   ```bash
   echo $OPENAI_API_KEY

   # Should print: sk-...
   # If empty, set it:
   export OPENAI_API_KEY="sk-your-actual-key"
   ```

2. **Check API key validity:**
   ```bash
   # Test with curl
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY"

   # Should return list of models, not 401 error
   ```

3. **Rate limit errors (429):**
   ```bash
   # Check your OpenAI usage limits at:
   # https://platform.openai.com/account/limits

   # Reduce concurrency in config
   # Or upgrade your OpenAI tier
   ```

4. **Persistent environment variables:**
   ```bash
   # Add to ~/.bashrc or ~/.zshrc for persistence:
   echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.bashrc
   source ~/.bashrc
   ```

#### Anthropic API Errors

**Symptoms:**
- `anthropic.APIError: Invalid API key`
- "Anthropic API key not provided"

**Solutions:**

1. **Set API key:**
   ```bash
   export ANTHROPIC_API_KEY="sk-ant-your-key"

   # Verify
   echo $ANTHROPIC_API_KEY
   ```

2. **Check API key validity:**
   ```bash
   curl https://api.anthropic.com/v1/messages \
     -H "x-api-key: $ANTHROPIC_API_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -H "content-type: application/json" \
     -d '{"model":"claude-3-haiku-20240307","max_tokens":10,"messages":[{"role":"user","content":"Hi"}]}'
   ```

#### Model Not Found

**Symptoms:**
- "Model X not found"
- `openai.NotFoundError: model 'gpt-5' not found`
- "Ollama model 'llama3.2' not available"

**Solutions:**

1. **For Ollama models:**
   ```bash
   # Pull the missing model
   ollama pull llama3.2

   # Or use a different model you have
   ollama list

   # Update your config to use available model
   ```

2. **For OpenAI models:**
   ```bash
   # Check model name spelling - common mistakes:
   # "gpt-4o-mini" NOT "gpt4-mini"
   # "gpt-4-turbo" NOT "gpt-4-turbo-preview" (in code)

   # List available models
   curl https://api.openai.com/v1/models \
     -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[].id'
   ```

3. **Model access issues:**
   ```bash
   # Some models require special access (e.g., GPT-4)
   # Check your OpenAI account tier and permissions
   ```

---

### Execution Issues

#### DAG Validation Errors

**Symptoms:**
- "Invalid graph structure"
- "Cycle detected in DAG"
- "Role 'X' referenced in edge but not defined"
- `ValueError: Graph validation failed`

**Root Causes:**
- Circular dependencies in execution graph
- Edge references non-existent roles
- Invalid edge definitions
- Orphaned edges after role removal

**Solutions:**

1. **Visualize the graph:**
   ```python
   from siare.utils.graph_viz import visualize_graph
   from siare.core.models import ProcessConfig

   # Load your SOP
   sop = ProcessConfig.parse_file("my_sop.json")

   # Generate graph visualization
   visualize_graph(sop.graph, "graph.png")
   ```

2. **Check for circular dependencies:**
   ```python
   # Verify each edge
   for edge in sop.graph:
       print(f"{edge.from_} -> {edge.to}")

   # Look for cycles: A -> B -> C -> A
   ```

3. **Validate role references:**
   ```python
   # List all role IDs
   role_ids = {role.id for role in sop.roles}

   # Check all edges reference existing roles
   for edge in sop.graph:
       from_roles = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
       for role in from_roles:
           if role not in role_ids and role != "user_input":
               print(f"ERROR: Edge references unknown role: {role}")
   ```

4. **Clean up orphaned edges:**
   ```python
   from siare.services.director import DirectorService

   # Director will clean orphaned edges during mutations
   director = DirectorService(llm_provider=provider)

   # Or manually filter
   valid_edges = [
       edge for edge in sop.graph
       if all(r in role_ids or r == "user_input"
              for r in (edge.from_ if isinstance(edge.from_, list) else [edge.from_]))
   ]
   ```

#### Timeout Errors

**Symptoms:**
- `ExecutionTimeoutError: Operation timed out after X seconds`
- Execution hangs indefinitely
- Process killed by OS

**Root Causes:**
- Complex pipeline with too many steps
- Slow LLM provider (local models on weak hardware)
- Large context windows
- Network latency

**Solutions:**

1. **Increase timeout:**
   ```bash
   # Set timeout environment variable (seconds)
   export SIARE_TIMEOUT=300

   # Or in Python
   from siare.services.execution_engine import ExecutionEngine
   engine = ExecutionEngine(llm_provider=provider, timeout=300)
   ```

2. **Check LLM provider latency:**
   ```python
   import time
   from siare.services.llm_provider import LLMMessage, LLMProviderFactory

   provider = LLMProviderFactory.create("ollama", model="llama3.2")

   start = time.time()
   response = provider.complete(
       messages=[LLMMessage(role="user", content="Test")],
       model="llama3.2"
   )
   elapsed = time.time() - start

   print(f"LLM latency: {elapsed:.2f}s")
   # If > 30s, consider faster model or hardware upgrade
   ```

3. **Reduce pipeline complexity:**
   ```python
   # Break large pipeline into smaller sub-pipelines
   # Reduce number of roles
   # Simplify prompts
   # Reduce context size
   ```

4. **Use faster models:**
   ```bash
   # Ollama: Use smaller/faster models
   ollama pull llama3.2:7b  # Instead of :70b
   ollama pull phi3         # Very fast, smaller model

   # OpenAI: Use gpt-4o-mini instead of gpt-4
   ```

#### Conditional Execution Errors

**Symptoms:**
- `ValueError: Invalid condition syntax`
- "Condition evaluation failed"
- Roles executed when they shouldn't be

**Root Causes:**
- Invalid Python expression in condition
- Unsafe code in condition (blocked by safety checks)
- Role outputs not available in condition context

**Solutions:**

1. **Check condition syntax:**
   ```python
   # Valid condition examples:
   edge = GraphEdge(
       from_="retriever",
       to="answerer",
       condition="retriever.documents != []"  # Valid
   )

   # Invalid:
   edge = GraphEdge(
       from_="retriever",
       to="answerer",
       condition="import os; os.system('rm -rf /')"  # BLOCKED by safety
   )
   ```

2. **Available context in conditions:**
   ```python
   # Conditions have access to:
   # - All upstream role outputs as {role_id}.{output_key}
   # - Python built-ins: len(), str(), int(), etc.
   # - Comparison operators: ==, !=, <, >, <=, >=
   # - Logical operators: and, or, not

   # Example:
   condition = "len(retriever.documents) > 3 and answerer.confidence > 0.8"
   ```

3. **Debug condition evaluation:**
   ```python
   # Enable debug logging
   import logging
   logging.getLogger("siare.services.execution_engine").setLevel(logging.DEBUG)

   # Check execution trace
   trace = engine.execute(sop, genome, task)
   for node in trace.node_executions:
       if node.get("skipped"):
           print(f"Role {node['role_id']} skipped: {node.get('skip_reason')}")
   ```

---

### Evolution Issues

#### No Improvement After Multiple Iterations

**Symptoms:**
- Quality scores stagnant across generations
- All mutations rejected
- Pareto frontier not expanding

**Root Causes:**
- Too few iterations (need 10+ for meaningful evolution)
- Poor initial prompts (local optima)
- Inadequate or incorrect metrics
- Overly restrictive constraints
- Evaluation metrics not aligned with goals

**Solutions:**

1. **Increase iteration count:**
   ```python
   from siare.services.scheduler import EvolutionScheduler

   scheduler = EvolutionScheduler(
       gene_pool=gene_pool,
       director=director,
       # ...
   )

   job = scheduler.start_job(
       job_config={
           "max_generations": 20,  # Increase from default 10
           "population_size": 10,
           # ...
       }
   )
   ```

2. **Review base prompts:**
   ```python
   # Check initial prompt quality
   genome = PromptGenome.parse_file("genome.json")
   for role_id, prompt in genome.rolePrompts.items():
       print(f"\n{role_id}:")
       print(prompt.content[:200])  # Preview

   # Improve prompts manually before evolution:
   # - Add clear instructions
   # - Provide examples
   # - Specify output format
   ```

3. **Add diverse metrics:**
   ```python
   from siare.core.models import MetricConfig, MetricType

   metrics = [
       MetricConfig(
           name="accuracy",
           type=MetricType.LLM_JUDGE,
           weight=1.0,
           config={"judge_prompt": "..."}
       ),
       MetricConfig(
           name="completeness",
           type=MetricType.LLM_JUDGE,
           weight=0.8,
           config={"judge_prompt": "..."}
       ),
       MetricConfig(
           name="latency",
           type=MetricType.RUNTIME,
           weight=0.3,
           config={"metric": "total_duration"}
       ),
   ]
   ```

4. **Relax constraints:**
   ```python
   # Check if constraints are too restrictive
   from siare.core.models import EvolutionConstraints

   constraints = EvolutionConstraints(
       max_roles=10,        # Increase if too low
       max_prompt_length=4000,  # Increase if prompts truncated
       allowed_models=["gpt-4o-mini", "gpt-4o"],  # Add more models
       # ...
   )
   ```

5. **Enable debug logging:**
   ```bash
   export SIARE_LOG_LEVEL=DEBUG

   # Check mutation logs:
   # - Which mutations were attempted?
   # - Why were they rejected?
   # - What were the scores?
   ```

#### Constraint Violations

**Symptoms:**
- "Constraint X violated"
- `ValueError: Constraint check failed`
- Mutations rejected with constraint errors

**Root Causes:**
- Mutation violates defined constraints
- Constraints too strict
- Bug in constraint validation

**Solutions:**

1. **Review constraint definitions:**
   ```python
   from siare.core.models import EvolutionConstraints

   # List all active constraints
   constraints = EvolutionConstraints(
       max_roles=5,
       max_prompt_length=2000,
       allowed_models=["gpt-4o-mini"],
       require_retrieval=True,
       max_tool_calls_per_role=3,
   )

   # Check which constraint failed
   errors = director.validate_constraints(sop, constraints)
   for error in errors:
       print(f"Constraint violation: {error}")
   ```

2. **Adjust constraint bounds:**
   ```python
   # Increase limits if too restrictive
   constraints = EvolutionConstraints(
       max_roles=10,              # Was 5
       max_prompt_length=4000,    # Was 2000
       allowed_models=["gpt-4o-mini", "gpt-4o"],  # Added gpt-4o
   )
   ```

3. **Validate before mutation:**
   ```python
   # Always validate constraints BEFORE applying mutations
   errors = director.validate_constraints(sop, constraints)
   if errors:
       raise ValueError(f"Constraint violations: {errors}")

   # Then mutate
   mutated_sop = director.mutate_sop(...)
   ```

---

### Memory Issues

#### Out of Memory (OOM)

**Symptoms:**
- `MemoryError`
- Process killed by OS (killed: 9)
- System becomes unresponsive
- Swap usage very high

**Root Causes:**
- Large batch sizes
- Too many concurrent executions
- Memory leaks
- Large model embeddings in memory

**Solutions:**

1. **Reduce batch sizes:**
   ```python
   # In evolution config
   job_config = {
       "population_size": 5,  # Reduce from 10
       "max_concurrent": 2,   # Reduce from 5
   }
   ```

2. **Limit concurrent executions:**
   ```python
   import asyncio

   # Limit asyncio semaphore
   semaphore = asyncio.Semaphore(2)  # Max 2 concurrent
   ```

3. **Monitor memory usage:**
   ```python
   import psutil
   import os

   process = psutil.Process(os.getpid())
   print(f"Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

4. **Increase available memory:**
   ```bash
   # Docker: Increase memory limit
   docker run -m 8g ...

   # Or close other applications
   # Or upgrade RAM
   ```

5. **Use smaller models:**
   ```bash
   # Ollama: Use 7B instead of 70B models
   ollama pull llama3.2:7b  # ~4GB RAM

   # Instead of:
   # ollama pull llama3.1:70b  # ~40GB RAM
   ```

6. **Clear caches periodically:**
   ```python
   from siare.services.llm_cache import LLMCache

   cache = LLMCache()
   cache.clear()  # Free memory
   ```

---

### Import and Module Issues

#### ModuleNotFoundError

**Symptoms:**
- `ModuleNotFoundError: No module named 'siare'`
- `ImportError: cannot import name 'X' from 'siare.Y'`
- `ModuleNotFoundError: No module named 'openai'`

**Root Causes:**
- Dependencies not installed
- Wrong virtual environment
- Incorrect PYTHONPATH

**Solutions:**

1. **Verify installation:**
   ```bash
   # Check if siare is installed
   pip show siare

   # If not found, install dependencies
   pip install -r requirements.txt
   ```

2. **Check virtual environment:**
   ```bash
   # Verify venv is active
   which python
   # Should point to: /path/to/siare/venv/bin/python

   # If not, activate it
   source venv/bin/activate
   ```

3. **Verify Python version:**
   ```bash
   python --version
   # Must be 3.10 or higher

   # If wrong version, create venv with correct Python:
   python3.10 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

4. **Install missing packages:**
   ```bash
   # Common missing packages:
   pip install openai          # For OpenAI provider
   pip install anthropic       # For Anthropic provider
   pip install fastapi uvicorn # For API server (siare-cloud)
   ```

5. **Check PYTHONPATH:**
   ```bash
   # Add repo root to PYTHONPATH
   export PYTHONPATH="/path/to/siare:$PYTHONPATH"

   # Or install in development mode
   pip install -e .
   ```

#### Import Errors After Updates

**Symptoms:**
- Code worked before, now imports fail
- `AttributeError: module 'siare' has no attribute 'X'`

**Solutions:**

1. **Reinstall dependencies:**
   ```bash
   pip install -r requirements.txt --upgrade
   ```

2. **Clear Python cache:**
   ```bash
   # Remove bytecode cache
   find . -type d -name __pycache__ -exec rm -rf {} +
   find . -type f -name "*.pyc" -delete
   ```

3. **Check for breaking changes:**
   ```bash
   git log --oneline
   # Review recent commits for API changes
   ```

---

### API Server Issues 🔒

> **Note**: The REST API server is an enterprise feature available in siare-cloud. The open-source core provides CLI and Python library interfaces.

#### Server Won't Start (siare-cloud)

**Symptoms:**
- `uvicorn: command not found`
- `Address already in use`
- `ImportError` when starting server

**Solutions:**

1. **Install FastAPI dependencies:**
   ```bash
   pip install fastapi uvicorn websockets
   ```

2. **Port already in use:**
   ```bash
   # Check what's using port 8000
   lsof -i :8000

   # Kill the process
   kill -9 <PID>

   # Or use different port
   uvicorn siare_cloud.api.server:app --port 8001
   ```

3. **Check environment variables:**
   ```bash
   # Required for server
   export OPENAI_API_KEY="sk-..."
   # Or
   export OLLAMA_BASE_URL="http://localhost:11434"
   ```

#### WebSocket Connection Errors

**Symptoms:**
- WebSocket connection refused
- 403 Forbidden on WebSocket
- Connection drops immediately

**Solutions:**

1. **Check CORS settings:**
   ```python
   # In server.py, verify CORS middleware
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],  # Or specific origins
       allow_credentials=True,
       allow_methods=["*"],
       allow_headers=["*"],
   )
   ```

2. **Test WebSocket endpoint:**
   ```bash
   # Using websocat
   websocat ws://localhost:8000/v1/jobs/abc-123/stream
   ```

---

### Test Failures

#### Tests Fail After Code Changes

**Symptoms:**
- Previously passing tests now fail
- `AssertionError` in tests
- Mock objects not working as expected

**Solutions:**

1. **Run specific failing test:**
   ```bash
   # Run single test with verbose output
   pytest tests/unit/test_director.py::test_mutate_sop -v

   # With debug output
   pytest tests/unit/test_director.py::test_mutate_sop -v -s
   ```

2. **Check test isolation:**
   ```bash
   # Run test in isolation
   pytest tests/unit/test_director.py::test_mutate_sop --forked

   # Or clear state between tests
   pytest --cache-clear
   ```

3. **Update test fixtures:**
   ```python
   # Check if fixtures need updating after API changes
   # Review tests/conftest.py
   ```

4. **Verify mock setup:**
   ```python
   # Ensure mocks match new signatures
   from unittest.mock import Mock

   mock_provider = Mock(spec=LLMProvider)
   # Verify spec matches current LLMProvider interface
   ```

---

## Debug Mode

### Enable Debug Logging

Get detailed logs for troubleshooting:

```bash
# Set log level
export SIARE_LOG_LEVEL=DEBUG

# Or in Python
import logging
logging.basicConfig(level=logging.DEBUG)

# For specific module
logging.getLogger("siare.services.execution_engine").setLevel(logging.DEBUG)
```

### View Execution Traces

```python
from siare.services.execution_engine import ExecutionEngine

# Execute pipeline
trace = execution_engine.execute(sop, genome, task)

# Inspect trace
print(f"Status: {trace.status}")
print(f"Duration: {(trace.end_time - trace.start_time).total_seconds()}s")
print(f"Total cost: ${trace.total_cost:.4f}")

# View node executions
for node in trace.node_executions:
    print(f"\nRole: {node['role_id']}")
    print(f"  Duration: {node['duration_ms']:.2f}ms")
    print(f"  Cost: ${node.get('cost', 0):.4f}")
    print(f"  Inputs: {list(node['inputs'].keys())}")
    print(f"  Outputs: {list(node['outputs'].keys())}")
    if node.get('skipped'):
        print(f"  SKIPPED: {node.get('skip_reason')}")

# View errors
for error in trace.errors:
    print(f"\nError in {error['role_id']}: {error['error']}")
```

### Inspect Evolution State

```python
from siare.services.gene_pool import GenePool

gene_pool = GenePool()

# List all SOPs
sops = gene_pool.list_sops()
for sop_id in sops:
    genes = gene_pool.get_history(sop_id)
    print(f"\nSOP {sop_id}: {len(genes)} versions")

    # Show best version
    best = max(genes, key=lambda g: g.quality)
    print(f"  Best quality: {best.quality:.4f}")
    print(f"  Version: {best.version}")

# View Pareto frontier
frontier = gene_pool.compute_pareto_frontier(
    sop_id="my_sop",
    metrics=["accuracy", "latency"]
)
print(f"\nPareto frontier: {len(frontier)} solutions")
for gene in frontier:
    print(f"  {gene.version}: accuracy={gene.metrics['accuracy']:.3f}, "
          f"latency={gene.metrics['latency']:.3f}")
```

---

## Performance Optimization

### Slow Execution

**Symptoms:**
- Pipeline takes too long to complete
- High latency on every request

**Solutions:**

1. **Profile execution:**
   ```python
   import cProfile
   import pstats

   profiler = cProfile.Profile()
   profiler.enable()

   trace = engine.execute(sop, genome, task)

   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)  # Top 20 functions
   ```

2. **Use LLM caching:**
   ```python
   from siare.services.llm_cache import LLMCache

   cache = LLMCache(ttl=3600)  # 1 hour cache
   ```

3. **Optimize prompts:**
   ```python
   # Reduce prompt length
   # Use more efficient models (gpt-4o-mini vs gpt-4)
   # Cache common responses
   ```

4. **Parallelize independent roles:**
   ```python
   # Ensure DAG allows parallel execution
   # Roles with no dependencies can run concurrently
   ```

### High Costs

**Symptoms:**
- OpenAI bills too high
- Running out of API credits quickly

**Solutions:**

1. **Track costs:**
   ```python
   # Check execution trace costs
   total_cost = sum(node['cost'] for node in trace.node_executions)
   print(f"Execution cost: ${total_cost:.4f}")
   ```

2. **Switch to cheaper models:**
   ```python
   # Use gpt-4o-mini instead of gpt-4
   # Or use Ollama (free, local)

   sop.models = {
       "default": "gpt-4o-mini"  # Was "gpt-4"
   }
   ```

3. **Use Ollama for development:**
   ```bash
   # Free local models
   ollama pull llama3.2

   # Use in config
   provider = LLMProviderFactory.create("ollama", model="llama3.2")
   ```

4. **Enable caching:**
   ```python
   from siare.services.llm_cache import LLMCache

   # Cache identical requests
   cache = LLMCache(ttl=86400)  # 24 hour cache
   ```

---

## Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing GitHub issues**: https://github.com/synaptiai/siare/issues
3. **Run diagnostics**: Use the diagnostic script above
4. **Enable debug logging**: `export SIARE_LOG_LEVEL=DEBUG`
5. **Isolate the problem**: Create minimal reproducible example

### How to Report Issues

When creating a GitHub issue, include:

1. **Environment information:**
   ```bash
   python --version
   pip list | grep -E "(siare|openai|anthropic|pydantic|fastapi)"
   uname -a  # macOS/Linux
   ```

2. **Diagnostic output:**
   ```bash
   python diagnose.py > diagnostics.txt
   ```

3. **Error messages:**
   - Full stack trace
   - Error message
   - Relevant log output (with DEBUG enabled)

4. **Minimal reproducible example:**
   ```python
   # Smallest possible code that reproduces the issue
   from siare.services import DirectorService
   # ...
   ```

5. **What you've tried:**
   - Solutions from this guide that didn't work
   - Workarounds attempted

### Community Resources

- **GitHub Issues**: https://github.com/synaptiai/siare/issues
- **Documentation**: `docs/`
- **Examples**: `siare/demos/`
- **Architecture Guide**: `docs/architecture/SYSTEM_ARCHITECTURE.md`

---

## Appendix: Environment Variables

### Supported Environment Variables

```bash
# LLM Providers
OPENAI_API_KEY=sk-...               # OpenAI API key
ANTHROPIC_API_KEY=sk-ant-...        # Anthropic API key
OLLAMA_BASE_URL=http://localhost:11434  # Ollama server URL
OLLAMA_MODEL=llama3.2               # Default Ollama model

# Logging
SIARE_LOG_LEVEL=INFO                # DEBUG, INFO, WARNING, ERROR, CRITICAL

# Execution
SIARE_TIMEOUT=120                   # Default timeout in seconds
SIARE_MAX_RETRIES=3                 # Max retries for LLM calls

# API Server
SIARE_HOST=0.0.0.0                  # API server host
SIARE_PORT=8000                     # API server port
SIARE_WORKERS=4                     # Number of workers

# Storage
SIARE_DATA_DIR=/path/to/data        # Data directory for gene pool
SIARE_CACHE_DIR=/path/to/cache      # Cache directory

# Feature Flags
SIARE_ENABLE_CACHE=true             # Enable LLM caching
SIARE_ENABLE_SAFETY_CHECKS=true     # Enable safety validation
```

### Setting Environment Variables

**Temporary (current session only):**
```bash
export OPENAI_API_KEY="sk-..."
```

**Persistent (all sessions):**
```bash
# Add to ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="sk-..."' >> ~/.bashrc
source ~/.bashrc
```

**Using .env file:**
```bash
# Create .env in project root
cat > .env << EOF
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
SIARE_LOG_LEVEL=DEBUG
EOF

# Load automatically (if using python-dotenv)
# Already configured in siare_cloud.api.server (enterprise)
```

---

## Quick Reference

### Essential Commands

```bash
# Health checks
python diagnose.py
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8000/health      # API server

# Run tests
pytest                                  # All tests
pytest -v -s                           # Verbose with output
pytest --lf                            # Last failed

# Start services
ollama serve                                # Ollama server
uvicorn siare_cloud.api.server:app --reload # API server (enterprise)

# Environment
source venv/bin/activate               # Activate venv
pip install -r requirements.txt        # Install deps
export SIARE_LOG_LEVEL=DEBUG           # Debug mode
```

### Common Fixes

```bash
# Ollama not connecting
ollama serve
ollama pull llama3.2

# OpenAI 401
export OPENAI_API_KEY="sk-..."

# Port in use
lsof -i :8000
kill -9 <PID>

# Import errors
pip install -r requirements.txt
source venv/bin/activate

# Clear cache
find . -name __pycache__ -exec rm -rf {} +
pytest --cache-clear
```

---

## See Also

- [Quick Start](QUICKSTART.md) — Initial setup guide
- [Configuration](CONFIGURATION.md) — All configuration options
- [Deployment](DEPLOYMENT.md) — Production deployment guide
- [GitHub Issues](https://github.com/synaptiai/siare/issues) — Report bugs
