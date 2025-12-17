"""Execution Engine - Builds and runs SOP DAGs"""

import asyncio
import logging
import re
import signal
import threading
import uuid
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timezone
from types import FrameType
from typing import TYPE_CHECKING, Any, ClassVar, Optional

from siare.core.hooks import HookContext, HookRegistry, fire_execution_hook
from siare.core.models import ProcessConfig, PromptGenome, RoleConfig

if TYPE_CHECKING:
    from siare.core.models import FeedbackArtifact, FeedbackInjectionConfig
    from siare.services.prompt_evolution.feedback_injector import FeedbackInjector
from siare.services.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitBreakerRegistry,
)
from siare.services.llm_provider import LLMMessage, LLMProvider, LLMResponse
from siare.services.retry_handler import RetryExhausted, RetryHandler

logger = logging.getLogger(__name__)


class ExecutionTimeoutError(Exception):
    """Raised when an operation times out"""



@contextmanager
def timeout(seconds: int):
    """
    Context manager for timeout protection.

    Note: Signal-based timeouts only work in the main thread. When running
    in a worker thread (e.g., with ThreadPoolExecutor), this context manager
    becomes a no-op to avoid ValueError. The underlying LLM provider typically
    has its own timeout handling.

    Args:
        seconds: Timeout duration in seconds

    Raises:
        ExecutionTimeoutError: If operation exceeds timeout (main thread only)

    Example:
        with timeout(30):
            long_running_operation()
    """
    # Check if we're in the main thread - signals only work there
    if threading.current_thread() is not threading.main_thread():
        # In a worker thread, skip signal-based timeout
        # The LLM provider has its own timeout handling
        yield
        return

    def timeout_handler(signum: int, frame: FrameType | None) -> None:
        # signum and frame are required by signal.signal() callback signature
        raise ExecutionTimeoutError(f"Operation timed out after {seconds} seconds")

    # Set the signal handler
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore old handler and cancel alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


class ExecutionTrace:
    """Trace of an execution run"""

    def __init__(self, run_id: str, sop_id: str, sop_version: str):
        self.run_id = run_id
        self.sop_id = sop_id
        self.sop_version = sop_version
        self.start_time = datetime.now(timezone.utc)
        self.end_time: datetime | None = None

        # Execution trace
        self.node_executions: list[dict[str, Any]] = []
        self.tool_calls: list[dict[str, Any]] = []
        self.errors: list[dict[str, Any]] = []

        # Final outputs
        self.final_outputs: dict[str, Any] = {}
        self.status: str = "running"

        # Cost tracking
        self.total_cost: float = 0.0

    def add_node_execution(
        self,
        role_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
        duration_ms: float,
        model: str,
        prompt: str,
        cost: float = 0.0,
    ) -> None:
        """Record a node execution"""
        self.node_executions.append(
            {
                "role_id": role_id,
                "inputs": inputs,
                "outputs": outputs,
                "duration_ms": duration_ms,
                "model": model,
                "prompt": prompt,
                "cost": cost,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )
        # Accumulate total cost
        self.total_cost += cost

    def add_tool_call(
        self,
        role_id: str,
        tool_id: str,
        inputs: dict[str, Any],
        outputs: dict[str, Any],
    ) -> None:
        """Record a tool invocation"""
        self.tool_calls.append(
            {
                "role_id": role_id,
                "tool_id": tool_id,
                "inputs": inputs,
                "outputs": outputs,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def add_error(self, role_id: str, error: str) -> None:
        """Record an error"""
        self.errors.append(
            {
                "role_id": role_id,
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def finalize(self, status: str, final_outputs: dict[str, Any]) -> None:
        """Finalize execution"""
        self.end_time = datetime.now(timezone.utc)
        self.status = status
        self.final_outputs = final_outputs

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary"""
        duration_ms = 0
        if self.end_time:
            duration_ms = (self.end_time - self.start_time).total_seconds() * 1000

        return {
            "run_id": self.run_id,
            "sop_id": self.sop_id,
            "sop_version": self.sop_version,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": duration_ms,
            "status": self.status,
            "node_executions": self.node_executions,
            "tool_calls": self.tool_calls,
            "errors": self.errors,
            "final_outputs": self.final_outputs,
            "total_cost": self.total_cost,
        }


class ExecutionEngine:
    """
    Executes SOPs as DAGs

    For MVP: Simplified executor with mock LLM calls
    For Production: Integrate with actual LLM providers
    """

    # Default model fallback cascade
    DEFAULT_MODEL_FALLBACK: ClassVar[list[str]] = [
        "gpt-5",
        "gpt-3.5-turbo",
        "claude-3-sonnet-20240229",
    ]

    # Default timeout for role execution (5 minutes)
    DEFAULT_ROLE_TIMEOUT = 300

    # Default timeout for tool invocation (30 seconds)
    DEFAULT_TOOL_TIMEOUT = 30

    # Error type constants
    ERROR_TIMEOUT = "timeout"
    ERROR_LLM_FAILURE = "llm_failure"
    ERROR_UNEXPECTED = "unexpected"

    # Metadata key constants
    META_ERROR = "_error"
    META_DEGRADED = "_degraded"
    META_MODEL_USED = "_model_used"
    META_LLM_RESPONSE = "_llm_response"

    def __init__(
        self,
        llm_provider: LLMProvider | None = None,
        tool_adapters: dict[str, Any] | None = None,
        retry_handler: RetryHandler | None = None,
        circuit_breaker_registry: CircuitBreakerRegistry | None = None,
        role_timeout: int = DEFAULT_ROLE_TIMEOUT,
        tool_timeout: int = DEFAULT_TOOL_TIMEOUT,
        model_fallback_cascade: list[str] | None = None,
    ):
        """
        Initialize execution engine

        Args:
            llm_provider: LLM provider for role execution (None = mock mode)
            tool_adapters: Optional dictionary of tool_id -> adapter implementations
            retry_handler: Retry handler for transient failures (creates default if None)
            circuit_breaker_registry: Circuit breaker registry (creates default if None)
            role_timeout: Timeout for role execution in seconds (default: 300)
            tool_timeout: Timeout for tool invocation in seconds (default: 30)
            model_fallback_cascade: Model fallback order (default: gpt-5 → gpt-3.5-turbo → claude-3-sonnet)
        """
        self.llm_provider = llm_provider
        self.tool_adapters = tool_adapters or {}
        self.retry_handler = retry_handler or RetryHandler()
        self.role_timeout = role_timeout
        self.tool_timeout = tool_timeout
        self.model_fallback_cascade = model_fallback_cascade or self.DEFAULT_MODEL_FALLBACK

        # Initialize circuit breaker registry
        from siare.services.circuit_breaker import get_circuit_breaker_registry

        self.circuit_breaker_registry = circuit_breaker_registry or get_circuit_breaker_registry()

        # Create circuit breakers
        self.llm_circuit_breaker = self.circuit_breaker_registry.get_or_create(
            "execution_engine_llm",
            CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )
        self.tool_circuit_breaker = self.circuit_breaker_registry.get_or_create(
            "execution_engine_tools",
            CircuitBreaker.TOOL_CIRCUIT_CONFIG,
        )

    def _fire_hook(self, hook_name: str, ctx: HookContext, *args: Any, **kwargs: Any) -> Any:
        """Fire an execution hook from sync context.

        Safely runs async hooks from synchronous code. If no hooks are registered,
        returns immediately with zero overhead. Errors are logged but don't propagate.

        Args:
            hook_name: Name of the hook method (e.g., "on_execution_start").
            ctx: Hook context with correlation ID and metadata.
            *args: Arguments for the hook.
            **kwargs: Keyword arguments for the hook.

        Returns:
            Hook result, or None if no hooks registered or hook failed.
        """
        # Fast path: no hooks registered = no overhead
        if HookRegistry.get_execution_hooks() is None:
            return None

        try:
            # Try to get existing event loop (may be running in async context)
            try:
                loop = asyncio.get_running_loop()
                # We're in an async context - create task but don't await
                # This is fire-and-forget for hooks
                loop.create_task(fire_execution_hook(hook_name, ctx, *args, **kwargs))
                return None
            except RuntimeError:
                # No running loop - create new one for sync context
                return asyncio.run(fire_execution_hook(hook_name, ctx, *args, **kwargs))
        except Exception as e:
            logger.warning(f"Failed to fire hook {hook_name}: {e}")
            return None

    def execute(
        self,
        sop: ProcessConfig,
        prompt_genome: PromptGenome,
        task_input: dict[str, Any],
        run_id: str | None = None,
        feedback_injector: Optional["FeedbackInjector"] = None,
        feedback_artifacts: list["FeedbackArtifact"] | None = None,
        feedback_config: Optional["FeedbackInjectionConfig"] = None,
    ) -> ExecutionTrace:
        """
        Execute an SOP on a task

        Args:
            sop: ProcessConfig to execute
            prompt_genome: PromptGenome for role prompts
            task_input: Input data for the task
            run_id: Optional run ID (generated if not provided)
            feedback_injector: Optional injector for runtime feedback
            feedback_artifacts: Feedback to inject
            feedback_config: Injection configuration

        Returns:
            ExecutionTrace with results and trace
        """
        run_id = run_id or str(uuid.uuid4())
        trace = ExecutionTrace(run_id, sop.id, sop.version)

        # Create hook context for this execution
        hook_ctx = HookContext(
            correlation_id=run_id,
            metadata={"sop_id": sop.id, "sop_version": sop.version},
        )

        # Fire execution start hook
        self._fire_hook("on_execution_start", hook_ctx, sop, task_input)

        try:
            # Inject feedback into prompts if configured
            active_genome = self._maybe_inject_feedback(
                sop=sop,
                prompt_genome=prompt_genome,
                feedback_injector=feedback_injector,
                feedback_artifacts=feedback_artifacts,
                feedback_config=feedback_config,
            )

            # Build execution graph with conditions
            graph, edge_conditions = self._build_graph(sop)

            # Topological sort to get execution order
            execution_order = self._topological_sort(graph)

            # Execute nodes in order
            state = {"user_input": task_input}  # Initial state

            for role_id in execution_order:
                role_config = self._get_role_config(sop, role_id)

                # Check if this role should execute based on incoming edge conditions
                should_execute = self._should_execute_role(
                    role_id, edge_conditions, state
                )

                if not should_execute:
                    # Skip this role - conditions not met
                    # Add a trace entry indicating skip
                    trace.add_node_execution(
                        role_id=role_id,
                        inputs={},
                        outputs={"_skipped": True, "_reason": "Conditional edge evaluated to false"},
                        duration_ms=0,
                        model=role_config.model,
                        prompt="[SKIPPED - condition not met]",
                    )
                    # Add skipped marker to state
                    state[role_id] = {"_skipped": True}
                    continue

                # Collect inputs for this role
                role_inputs = self._collect_inputs(role_config, state)

                # Get prompt for this role
                prompt = self._get_prompt(active_genome, role_config.promptRef)

                # Execute role (mock for MVP)
                start_time = datetime.now(timezone.utc)
                role_outputs = self._execute_role(
                    role_config,
                    role_inputs,
                    prompt,
                    trace,
                )
                end_time = datetime.now(timezone.utc)
                duration_ms = (end_time - start_time).total_seconds() * 1000

                # Extract cost from LLM response if available
                llm_response = role_outputs.get(self.META_LLM_RESPONSE)
                role_cost = llm_response.cost if llm_response and hasattr(llm_response, "cost") else 0.0

                # Record execution
                trace.add_node_execution(
                    role_id=role_id,
                    inputs=role_inputs,
                    outputs=role_outputs,
                    duration_ms=duration_ms,
                    model=role_config.model,
                    prompt=prompt,
                    cost=role_cost,
                )

                # Fire role complete hook
                self._fire_hook(
                    "on_role_complete",
                    hook_ctx,
                    role_id,
                    role_inputs,
                    role_outputs,
                    duration_ms,
                )

                # Update state
                state[role_id] = role_outputs

            # Extract final outputs
            final_outputs = self._extract_final_outputs(sop, state)
            trace.finalize("completed", final_outputs)

            # Fire execution complete hook (success path)
            execution_duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000 if trace.end_time else 0.0
            self._fire_hook("on_execution_complete", hook_ctx, sop, trace, execution_duration_ms)

        except ValueError as e:
            trace.add_error("system", f"ValueError: {e!s}")
            trace.finalize("failed", {})
            # Fire execution complete hook (error path)
            execution_duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000 if trace.end_time else 0.0
            self._fire_hook("on_execution_complete", hook_ctx, sop, trace, execution_duration_ms)
        except KeyError as e:
            trace.add_error("system", f"KeyError: {e!s}")
            trace.finalize("failed", {})
            # Fire execution complete hook (error path)
            execution_duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000 if trace.end_time else 0.0
            self._fire_hook("on_execution_complete", hook_ctx, sop, trace, execution_duration_ms)
        except Exception as e:
            trace.add_error("system", str(e))
            trace.finalize("failed", {})
            # Fire execution complete hook (error path)
            execution_duration_ms = (trace.end_time - trace.start_time).total_seconds() * 1000 if trace.end_time else 0.0
            self._fire_hook("on_execution_complete", hook_ctx, sop, trace, execution_duration_ms)

        return trace

    def _should_execute_role(
        self,
        role_id: str,
        edge_conditions: dict[tuple[str, str], str | None],
        state: dict[str, Any],
    ) -> bool:
        """
        Determine if a role should execute based on incoming edge conditions

        A role executes if:
        1. It has no incoming edges (always executes)
        2. At least one incoming edge has no condition (always true)
        3. At least one incoming edge's condition evaluates to true

        Args:
            role_id: ID of the role to check
            edge_conditions: Dict of (from_node, to_node) -> condition
            state: Current execution state

        Returns:
            True if role should execute, False otherwise
        """
        # Find all incoming edges to this role
        incoming_edges = [
            (_from_node, _to_node, condition)
            for (_from_node, _to_node), condition in edge_conditions.items()
            if _to_node == role_id
        ]

        # If no incoming edges, always execute
        if not incoming_edges:
            return True

        # Check if any incoming edge allows execution
        # (OR logic: at least one path must be satisfied)
        for _from_node, _to_node, condition in incoming_edges:
            # If edge has no condition, it's always active
            if not condition or not condition.strip():
                return True

            # Evaluate the condition
            try:
                if self._evaluate_condition(condition, state):
                    # At least one path is satisfied
                    return True
            except ValueError:
                # Condition evaluation failed - treat as false but log error
                # This allows execution to continue with other paths
                # In a production system, this would be logged
                continue

        # No incoming edge condition was satisfied
        return False

    def _check_condition_safety(self, condition: str) -> None:
        """
        Check condition syntax and safety (shared validation logic)

        Args:
            condition: Condition expression string

        Raises:
            ValueError: If condition syntax is invalid or contains dangerous keywords
        """
        # Validate syntax with regex
        # Only allow safe operators and prevent code injection
        safe_pattern = r"^[\w\s\.\(\)><=!]+(?:and|or|is|not|in)?[\w\s\.\(\)><=!]*$"
        if not re.match(safe_pattern, condition, re.IGNORECASE):
            raise ValueError(
                f"Invalid condition syntax: '{condition}'. "
                "Only alphanumeric characters, comparison operators (==, !=, >, <, >=, <=), "
                "boolean operators (and, or, not), and 'is' checks are allowed."
            )

        # Prevent dangerous operations
        dangerous_keywords = ["import", "exec", "eval", "__", "lambda", "def", "class"]
        condition_lower = condition.lower()
        for keyword in dangerous_keywords:
            if keyword in condition_lower:
                raise ValueError(f"Forbidden keyword in condition: '{keyword}'")

    def _evaluate_condition(self, condition: str, state: dict[str, Any]) -> bool:
        """
        Evaluate a conditional expression against execution state

        Supported condition syntax:
        - Simple field checks: "field_name == value", "score > 0.8", "status != 'error'"
        - Boolean operators: "field1 and field2", "score > 0.8 or fallback == true"
        - None checks: "field is not None", "result is None"

        Args:
            condition: Condition expression string
            state: Current execution state (role outputs)

        Returns:
            Boolean result of condition evaluation

        Raises:
            ValueError: If condition syntax is invalid or references undefined fields
        """
        if not condition or not condition.strip():
            # Empty condition means always true
            return True

        # Build a safe evaluation context with state values
        # Only expose simple types and avoid code execution risks
        eval_context: dict[str, Any] = {}

        # Flatten state into evaluation context
        # Each role's outputs become available as variables
        for role_id, outputs in state.items():
            if isinstance(outputs, dict):
                # Add each output field directly to context
                for field_name, field_value in outputs.items():  # type: ignore[misc]
                    eval_context[field_name] = field_value
            else:
                # Add role output directly
                eval_context[role_id] = outputs

        # Add None for None checks
        eval_context["None"] = None

        # Validate condition syntax and safety
        self._check_condition_safety(condition)

        try:
            # Evaluate condition in restricted context
            # This is safe because:
            # 1. We validated syntax with regex
            # 2. We blocked dangerous keywords
            # 3. We only expose simple data types from state
            result = eval(condition, {"__builtins__": {}}, eval_context)  # noqa: S307

            # Ensure result is boolean
            if not isinstance(result, bool):
                raise ValueError(
                    f"Condition '{condition}' must evaluate to boolean, got {type(result).__name__}"
                )

            return result

        except NameError as e:
            # Field referenced in condition doesn't exist in state
            raise ValueError(
                f"Condition '{condition}' references undefined field: {e!s}"
            ) from e
        except SyntaxError as e:
            raise ValueError(f"Condition syntax error: {e!s}") from e
        except Exception as e:
            raise ValueError(f"Failed to evaluate condition '{condition}': {e!s}") from e

    def _validate_condition_syntax(self, condition: str) -> None:
        """
        Validate condition syntax without evaluating it

        This is used during SOP validation to catch syntax errors early,
        before execution begins.

        Args:
            condition: Condition expression string

        Raises:
            ValueError: If condition syntax is invalid
        """
        if not condition or not condition.strip():
            return  # Empty condition is valid

        # Validate condition syntax and safety
        self._check_condition_safety(condition)

        # Try to compile as Python expression to catch syntax errors
        try:
            compile(condition, "<condition>", "eval")
        except SyntaxError as e:
            raise ValueError(f"Condition syntax error: {e!s}") from e

    def _build_graph(self, sop: ProcessConfig) -> tuple[dict[str, list[str]], dict[tuple[str, str], str | None]]:
        """
        Build adjacency list representation of graph with edge conditions

        Returns:
            Tuple of:
            - Dict mapping role_id -> list of dependent role_ids (adjacency list)
            - Dict mapping (from_node, to_node) -> condition string (edge conditions)
        """
        graph: dict[str, list[str]] = defaultdict(list)
        conditions: dict[tuple[str, str], str | None] = {}

        # Add all roles as nodes
        for role in sop.roles:
            if role.id not in graph:
                graph[role.id] = []

        # Add edges and conditions
        for edge in sop.graph:
            from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
            for from_node in from_nodes:
                if from_node != "user_input":  # Skip special "user_input" node
                    graph[from_node].append(edge.to)

                # Store condition for this edge
                # Key is (from_node, to_node) tuple
                conditions[(from_node, edge.to)] = edge.condition

        return dict(graph), conditions

    def _topological_sort(self, graph: dict[str, list[str]]) -> list[str]:
        """
        Topological sort of DAG using Kahn's algorithm

        Args:
            graph: Adjacency list representation

        Returns:
            List of role IDs in execution order

        Raises:
            ValueError: If graph contains cycles
        """
        # Calculate in-degrees
        in_degree: dict[str, int] = dict.fromkeys(graph, 0)
        for node in graph:
            for neighbor in graph[node]:
                if neighbor not in in_degree:
                    in_degree[neighbor] = 0
                in_degree[neighbor] += 1

        # Initialize queue with nodes that have no dependencies
        queue = deque([node for node, degree in in_degree.items() if degree == 0])
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)

            # Reduce in-degree for neighbors
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        # Check for cycles
        if len(result) != len(in_degree):
            raise ValueError("Graph contains cycles - cannot execute")

        return result

    def _get_role_config(self, sop: ProcessConfig, role_id: str) -> RoleConfig:
        """Get RoleConfig by ID"""
        for role in sop.roles:
            if role.id == role_id:
                return role
        raise ValueError(f"Role {role_id} not found in SOP")

    def _collect_inputs(self, role_config: RoleConfig, state: dict[str, Any]) -> dict[str, Any]:
        """Collect inputs for a role from execution state"""
        if not role_config.inputs:
            return {}

        inputs: dict[str, Any] = {}

        for input_spec in role_config.inputs:
            # Handle both dict (legacy) and RoleInput (new) formats
            from_nodes: Any
            fields: Any
            if isinstance(input_spec, dict):
                from_nodes = input_spec.get("from")  # type: ignore[misc]
                fields = input_spec.get("fields")  # type: ignore[misc]
            else:
                # RoleInput object
                from_nodes = input_spec.from_
                fields = input_spec.fields

            # Normalize to list
            from_nodes_list: list[Any] = from_nodes if isinstance(from_nodes, list) else [from_nodes]  # type: ignore[misc]

            from_node: Any
            for from_node in from_nodes_list:
                if from_node in state:
                    source_data = state[from_node]

                    if fields:
                        # Extract specific fields
                        field: Any
                        for field in fields:  # type: ignore[misc]
                            if field in source_data:
                                inputs[field] = source_data[field]
                    # Include all data
                    elif isinstance(source_data, dict):
                        inputs.update(source_data)  # type: ignore[misc]
                    else:
                        inputs[from_node] = source_data

        return inputs

    def _get_prompt(self, prompt_genome: PromptGenome, prompt_ref: str) -> str:
        """Get prompt content from PromptGenome"""
        if prompt_ref not in prompt_genome.rolePrompts:
            raise ValueError(f"Prompt {prompt_ref} not found in PromptGenome")
        return prompt_genome.rolePrompts[prompt_ref].content

    def _maybe_inject_feedback(
        self,
        sop: ProcessConfig,
        prompt_genome: PromptGenome,
        feedback_injector: Optional["FeedbackInjector"],
        feedback_artifacts: list["FeedbackArtifact"] | None,
        feedback_config: Optional["FeedbackInjectionConfig"],
    ) -> PromptGenome:
        """Inject feedback into prompts if configured.

        Args:
            sop: SOP configuration with roles.
            prompt_genome: Original prompt genome.
            feedback_injector: Optional injector instance.
            feedback_artifacts: Optional feedback to inject.
            feedback_config: Optional injection configuration.

        Returns:
            Modified genome with feedback injected, or original if not configured.
        """
        # Check all required components are present and enabled
        should_inject = (
            feedback_injector is not None
            and feedback_artifacts is not None
            and feedback_config is not None
            and feedback_config.enabled
        )

        if not should_inject:
            return prompt_genome

        active_genome = prompt_genome
        for role in sop.roles:
            active_genome = feedback_injector.inject_feedback(  # type: ignore[union-attr]
                prompt_genome=active_genome,
                feedback_artifacts=feedback_artifacts,  # type: ignore[arg-type]
                role_id=role.id,
                prompt_ref=role.promptRef,
                config=feedback_config,
            )
        logger.info("Injected feedback into prompts for execution")
        return active_genome

    def _call_llm_with_fallback(
        self,
        messages: list[LLMMessage],
        preferred_model: str,
        temperature: float = 0.7,
        role_id: str = "unknown",
    ) -> tuple[str, str, LLMResponse]:
        """
        Call LLM with model fallback cascade

        Tries models in order from fallback cascade until one succeeds.
        Uses retry handler for transient failures on each model.

        Args:
            messages: Messages to send to LLM
            preferred_model: Preferred model from role config
            temperature: Temperature for generation
            role_id: Role ID for logging

        Returns:
            Tuple of (response_content, model_used, llm_response)

        Raises:
            RuntimeError: If all models in cascade fail
        """
        # Build model cascade: preferred model first, then fallback models
        models_to_try = [preferred_model]
        for fallback_model in self.model_fallback_cascade:
            if fallback_model != preferred_model and fallback_model not in models_to_try:
                models_to_try.append(fallback_model)

        last_error: Exception | None = None
        for model in models_to_try:
            try:
                logger.debug(f"Trying model {model} for role {role_id}")

                # Wrap with circuit breaker first, then retry handler
                # Note: self.llm_provider is guaranteed to be non-None here because
                # this function is only called from _execute_role when llm_provider is not None
                response = self.llm_circuit_breaker.call(
                    lambda: self.retry_handler.execute_with_retry(
                        self.llm_provider.complete,  # type: ignore[union-attr]
                        messages=messages,
                        model=model,
                        temperature=temperature,
                        retry_config=RetryHandler.LLM_RETRY_CONFIG,
                        component="ExecutionEngine",
                        operation=f"llm_call_{role_id}",
                    )
                )

                logger.info(f"Successfully called {model} for role {role_id}")
                return response.content, model, response

            except (RetryExhausted, CircuitBreakerOpenError) as e:
                last_error = e
                logger.warning(
                    f"Model {model} failed for role {role_id}: {e}. "
                    f"Trying next model in cascade..."
                )
                continue

            except Exception as e:
                last_error = e
                logger.warning(
                    f"Unexpected error with model {model} for role {role_id}: {e}. "
                    f"Trying next model in cascade..."
                )
                continue

        # All models failed
        logger.exception(f"All models failed for role {role_id}")
        raise RuntimeError(
            f"All models in cascade failed for role {role_id}. "
            f"Last error: {last_error}"
        )

    def _execute_role(
        self,
        role_config: RoleConfig,
        inputs: dict[str, Any],
        prompt: str,
        trace: ExecutionTrace,
    ) -> dict[str, Any]:
        """
        Execute a single role with error handling and timeout protection

        If LLM provider is available: Makes real LLM calls with model fallback
        If LLM provider is None: Returns mock outputs (backward compatibility)

        Features:
        - Model fallback cascade for resilience
        - Timeout protection for long-running operations
        - Graceful degradation on partial failures
        - Tool invocation with circuit breakers

        Args:
            role_config: Role configuration
            inputs: Input data for the role
            prompt: Formatted prompt
            trace: Execution trace for logging

        Returns:
            dict[str, Any]: Role outputs
        """
        outputs = {}

        # PRODUCTION MODE: Real LLM calls if provider available
        if self.llm_provider is not None:
            try:
                # Add timeout protection
                with timeout(self.role_timeout):
                    # Format prompt with inputs
                    formatted_prompt = self._format_prompt(prompt, inputs)

                    # Prepare messages
                    messages = [
                        LLMMessage(role="system", content="You are a helpful AI assistant."),
                        LLMMessage(role="user", content=formatted_prompt),
                    ]

                    # Call LLM with fallback cascade
                    response_content, model_used, llm_response = self._call_llm_with_fallback(
                        messages=messages,
                        preferred_model=role_config.model,
                        temperature=0.7,
                        role_id=role_config.id,
                    )

                    # Parse LLM response into structured outputs
                    outputs = self._parse_llm_response(
                        response_content,
                        role_config.outputs or [],
                    )

                    # Add metadata - store full LLMResponse for cost tracking
                    outputs[self.META_MODEL_USED] = model_used
                    outputs[self.META_LLM_RESPONSE] = llm_response  # Full LLMResponse object (has .cost)

                    logger.debug(
                        f"Successfully executed role {role_config.id} with model {model_used}"
                    )

            except ExecutionTimeoutError as e:
                # Timeout occurred - graceful degradation
                logger.warning(
                    f"Role {role_config.id} timed out after {self.role_timeout}s. "
                    f"Falling back to mock outputs."
                )
                trace.add_error(role_config.id, f"Timeout: {e}")
                outputs = self._create_error_outputs(role_config, inputs, self.ERROR_TIMEOUT)

            except RuntimeError as e:
                # All models in cascade failed - graceful degradation
                logger.warning(
                    f"All models failed for role {role_config.id}. "
                    f"Falling back to mock outputs. Error: {e}"
                )
                trace.add_error(role_config.id, f"LLM failure: {e}")
                outputs = self._create_error_outputs(role_config, inputs, self.ERROR_LLM_FAILURE)

            except Exception as e:
                # Unexpected error - graceful degradation
                logger.exception(f"Unexpected error executing role {role_config.id}")
                trace.add_error(role_config.id, f"Unexpected error: {e}")
                outputs = self._create_error_outputs(role_config, inputs, self.ERROR_UNEXPECTED)

        # NO LLM provider - fail loudly (CLAUDE.md rule: no silent fallbacks)
        else:
            raise RuntimeError(
                f"LLM provider required to execute role '{role_config.id}'. "
                "Cannot run SOP without an LLM provider. "
                "Ensure ExecutionEngine is initialized with a valid llm_provider."
            )

        # Invoke tools if configured (with error handling)
        if role_config.tools:
            for tool_ref in role_config.tools:
                try:
                    tool_output = self._invoke_tool(
                        tool_ref,
                        inputs,
                        trace,
                        role_config.id,
                        role_params=role_config.params,  # Pass role params (e.g., top_k, similarity_threshold)
                    )
                    trace.add_tool_call(
                        role_id=role_config.id,
                        tool_id=tool_ref,
                        inputs=inputs,
                        outputs=tool_output,
                    )
                    outputs[f"{tool_ref}_result"] = tool_output

                except Exception as e:
                    # Tool invocation failed - log but continue (graceful degradation)
                    logger.warning(
                        f"Tool {tool_ref} failed for role {role_config.id}: {e}. Continuing..."
                    )
                    trace.add_error(role_config.id, f"Tool {tool_ref} failed: {e}")
                    outputs[f"{tool_ref}_result"] = {
                        "error": str(e),
                        "status": "failed",
                    }

        return outputs

    def _format_prompt(self, prompt: str, inputs: dict[str, Any]) -> str:
        """
        Format prompt template with inputs

        Simple string substitution for now.
        In production, could use more sophisticated templating.
        """
        formatted = prompt
        for key, value in inputs.items():
            placeholder = f"{{{key}}}"
            if placeholder in formatted:
                formatted = formatted.replace(placeholder, str(value))
        return formatted

    def _parse_llm_response(self, response: str, expected_outputs: list[str]) -> dict[str, Any]:
        """
        Parse LLM response into structured outputs.

        Attempts to extract expected output fields using:
        1. JSON parsing (if response is valid JSON)
        2. Structured text parsing (field: value format)
        3. Fallback to raw response

        Args:
            response: Raw LLM response
            expected_outputs: List of expected output field names

        Returns:
            Parsed outputs dictionary
        """
        import json

        outputs: dict[str, Any] = {}

        # Try JSON parsing first
        try:
            # Handle markdown code blocks
            content = response.strip()
            if content.startswith("```"):
                lines = content.split("\n")
                # Find content between ``` markers
                start_idx = 1 if lines[0].startswith("```") else 0
                end_idx = len(lines)
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() == "```":
                        end_idx = i
                        break
                content = "\n".join(lines[start_idx:end_idx])
                # Remove language identifier if present
                if content.startswith("json"):
                    content = content[4:].strip()

            parsed = json.loads(content)
            if isinstance(parsed, dict):
                for field in expected_outputs:
                    if field in parsed:
                        outputs[field] = parsed[field]
                    # Try case-insensitive match
                    elif field.lower() in {k.lower(): k for k in parsed}:  # type: ignore[misc]
                        key: str = next(k for k in parsed if k.lower() == field.lower())  # type: ignore[misc]
                        outputs[field] = parsed[key]

                if outputs:
                    # Also keep full response for reference
                    outputs["response"] = response
                    return outputs

        except json.JSONDecodeError:
            pass

        # Try structured text parsing (field: value format)
        for field in expected_outputs:
            # Pattern: "field: value" or "field - value" on its own line
            pattern = rf"^{re.escape(field)}\s*[:\-]\s*(.+?)(?:\n|$)"
            match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
            if match:
                value = match.group(1).strip()
                # Try to parse as number if possible
                try:
                    if "." in value:
                        outputs[field] = float(value)
                    else:
                        outputs[field] = int(value)
                except ValueError:
                    outputs[field] = value

        if outputs:
            outputs["response"] = response
            return outputs

        # Fallback: full response as primary output
        outputs["response"] = response
        if expected_outputs:
            # Use first expected output as alias for full response
            outputs[expected_outputs[0]] = response

        return outputs

    def _create_error_outputs(
        self,
        role_config: RoleConfig,
        inputs: dict[str, Any],
        error_type: str,
    ) -> dict[str, Any]:
        """
        Create error outputs for graceful degradation (no fake data).

        Returns outputs with error markers but NO fake scores or content.
        This ensures evaluation metrics reflect the actual failure.

        Args:
            role_config: Role configuration
            inputs: Role inputs
            error_type: Error type constant (e.g., ERROR_TIMEOUT, ERROR_LLM_FAILURE)

        Returns:
            Outputs with error metadata (no fake data)
        """
        outputs: dict[str, Any] = {
            self.META_ERROR: error_type,
            self.META_DEGRADED: True,
        }

        # Mark each expected output as None to indicate failure
        # (NOT fake values - evaluation should handle None appropriately)
        if role_config.outputs:
            for output_field in role_config.outputs:
                outputs[output_field] = None

        return outputs

    def _handle_tool_error(
        self,
        tool_ref: str,
        role_id: str,
        error: Exception,
        error_type: str,
    ) -> None:
        """Handle tool invocation error with appropriate logging.

        Args:
            tool_ref: Tool identifier
            role_id: Role ID for context
            error: The exception that occurred
            error_type: Type of error for message selection

        Raises:
            RuntimeError: Always raises with formatted message
        """
        error_configs = {
            "timeout": {
                "log": f"Tool {tool_ref} timed out after {self.tool_timeout}s for role {role_id}",
                "msg": f"Tool {tool_ref} timed out: {error}",
            },
            "circuit_open": {
                "log": f"Circuit breaker open for tool {tool_ref} in role {role_id}. Tool temporarily unavailable.",
                "msg": f"Tool {tool_ref} circuit breaker open: {error}",
            },
            "retry_exhausted": {
                "log": f"All retry attempts exhausted for tool {tool_ref} in role {role_id}",
                "msg": f"Tool {tool_ref} failed after retries: {error}",
            },
            "unexpected": {
                "log": f"Unexpected error invoking tool {tool_ref} for role {role_id}",
                "msg": f"Tool {tool_ref} unexpected error: {error}",
            },
        }

        config = error_configs[error_type]
        if error_type == "unexpected":
            logger.exception(config["log"])
        else:
            logger.warning(config["log"])

        raise RuntimeError(config["msg"]) from error

    def _invoke_tool(
        self,
        tool_ref: str,
        inputs: dict[str, Any],
        trace: ExecutionTrace,
        role_id: str,
        role_params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Invoke a tool adapter with error handling and timeout protection.

        Features:
        - Circuit breaker for fault isolation
        - Retry handler for transient failures
        - Timeout protection
        - Graceful degradation
        - Role params merged into tool inputs (for evolvable RAG parameters)

        Args:
            tool_ref: Tool identifier
            inputs: Input data for tool
            trace: Execution trace for logging
            role_id: Role ID for context
            role_params: Optional role configuration parameters (e.g., top_k, similarity_threshold)
                        These are merged into inputs before tool invocation.

        Returns:
            Tool output dictionary

        Raises:
            KeyError: If tool is not registered
            RuntimeError: If tool invocation fails after all retries
        """
        if tool_ref not in self.tool_adapters:
            raise KeyError(
                f"Tool '{tool_ref}' not found in registered tool adapters. "
                f"Available tools: {list(self.tool_adapters.keys())}. "
                "Register the tool adapter before using it in a role configuration."
            )

        # Merge role params into inputs (role params take precedence for RAG config)
        # Filter out internal keys (starting with _) like _evolvable metadata
        merged_inputs = {**inputs}
        if role_params:
            filtered_params = {k: v for k, v in role_params.items() if not k.startswith("_")}
            merged_inputs.update(filtered_params)
            logger.debug(f"Merged role params {filtered_params} into tool inputs for {tool_ref}")

        try:
            with timeout(self.tool_timeout):
                result = self.tool_circuit_breaker.call(
                    lambda: self.retry_handler.execute_with_retry(
                        lambda: self.tool_adapters[tool_ref](merged_inputs),
                        retry_config=RetryHandler.TOOL_RETRY_CONFIG,
                        component="ExecutionEngine",
                        operation=f"tool_{tool_ref}",
                    )
                )
                logger.debug(f"Successfully invoked tool {tool_ref} for role {role_id}")
                return result

        except ExecutionTimeoutError as e:
            self._handle_tool_error(tool_ref, role_id, e, "timeout")
        except CircuitBreakerOpenError as e:
            self._handle_tool_error(tool_ref, role_id, e, "circuit_open")
        except RetryExhausted as e:
            self._handle_tool_error(tool_ref, role_id, e, "retry_exhausted")
        except Exception as e:
            self._handle_tool_error(tool_ref, role_id, e, "unexpected")

        # This line is unreachable but satisfies type checkers
        raise RuntimeError("Unreachable")

    def _extract_final_outputs(self, sop: ProcessConfig, state: dict[str, Any]) -> dict[str, Any]:
        """
        Extract final outputs from execution state

        For now, returns outputs from all terminal nodes
        """
        # Find terminal nodes (nodes with no outgoing edges)
        terminal_nodes = {role.id for role in sop.roles}

        for edge in sop.graph:
            from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
            for from_node in from_nodes:
                terminal_nodes.discard(from_node)

        # Collect outputs from terminal nodes
        final_outputs = {}
        for node_id in terminal_nodes:
            if node_id in state:
                final_outputs[node_id] = state[node_id]

        # If no terminal nodes found, return all role outputs
        if not final_outputs:
            final_outputs = {
                role.id: state.get(role.id, {}) for role in sop.roles if role.id in state
            }

        return final_outputs

    def validate_sop(self, sop: ProcessConfig) -> list[str]:
        """
        Validate SOP structure

        Returns:
            List of validation errors (empty if valid)
        """
        errors: list[str] = []

        # Check for duplicate role IDs
        role_ids = [role.id for role in sop.roles]
        if len(role_ids) != len(set(role_ids)):
            duplicates: list[str] = [rid for rid in role_ids if role_ids.count(rid) > 1]
            errors.append(f"Duplicate role IDs: {set(duplicates)}")

        # Check graph references valid roles
        valid_role_ids = set(role_ids) | {"user_input"}
        for edge in sop.graph:
            from_nodes = edge.from_ if isinstance(edge.from_, list) else [edge.from_]
            for from_node in from_nodes:
                if from_node not in valid_role_ids:
                    errors.append(f"Graph edge references unknown role: {from_node}")
            if edge.to not in role_ids:
                errors.append(f"Graph edge targets unknown role: {edge.to}")

        # Check for cycles
        try:
            graph, _ = self._build_graph(sop)
            self._topological_sort(graph)
        except ValueError as e:
            errors.append(str(e))

        # Validate edge conditions
        for edge in sop.graph:
            if edge.condition:
                try:
                    # Try to validate condition syntax
                    # We can't fully evaluate it without state, but we can check syntax
                    self._validate_condition_syntax(edge.condition)
                except ValueError as e:
                    errors.append(
                        f"Invalid condition on edge from {edge.from_} to {edge.to}: {e!s}"
                    )

        # Check tool references
        if sop.tools:
            role_tools: set[str] = set()
            for role in sop.roles:
                if role.tools:
                    role_tools.update(role.tools)

            # Warn about unused tools in config
            unused = set(sop.tools) - role_tools
            if unused:
                errors.append(f"Tools defined but not used by any role: {unused}")

        return errors
