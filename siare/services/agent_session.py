"""Multi-turn LLM agent session with tool dispatch.

Provides the conversational infrastructure for the AgenticDirector.
Each session maintains a message history and can dispatch tool calls
to registered VariationTools, feeding results back to the LLM.

Design Principles:
    1. Builds on existing LLMProvider abstraction — no new LLM clients.
    2. Budget-aware — every LLM call and tool invocation is tracked via
       InnerLoopBudget.
    3. Resilient — uses CircuitBreaker + RetryHandler from existing infra.
    4. Tool results are fed back as user messages for maximum compatibility
       with providers that don't support native tool-use responses.
"""

from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from siare.core.models import InnerLoopBudget, VariationToolSpec
from siare.services.circuit_breaker import CircuitBreaker
from siare.services.llm_provider import LLMMessage, LLMResponse
from siare.services.retry_handler import RetryHandler

if TYPE_CHECKING:
    from siare.services.llm_provider import LLMProvider

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolExecutor(Protocol):
    """Interface for executing a variation tool by name."""

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return a string result."""
        ...


class AgentSession:
    """Multi-turn LLM conversation with tool dispatch and budget tracking.

    Usage:
        session = AgentSession(
            llm_provider=provider,
            model="gpt-5",
            system_prompt="You are a mutation expert.",
            tools=[VariationToolSpec(name="dry_run", ...)],
            tool_executor=my_executor,
            budget=InnerLoopBudget(),
        )
        response = session.turn("Diagnose this SOP and propose a mutation.")
        # response contains the LLM's final text after any tool calls
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        model: str,
        system_prompt: str,
        tools: list[VariationToolSpec] | None = None,
        tool_executor: ToolExecutor | None = None,
        budget: InnerLoopBudget | None = None,
        retry_handler: RetryHandler | None = None,
        circuit_breaker: CircuitBreaker | None = None,
        max_tool_rounds: int = 5,
        temperature: float = 0.5,
        max_messages: int = 50,
    ) -> None:
        self.llm_provider = llm_provider
        self.model = model
        self.tools = tools or []
        self.tool_executor = tool_executor
        self.budget = budget or InnerLoopBudget()
        self.retry_handler = retry_handler or RetryHandler()
        self.circuit_breaker = circuit_breaker or CircuitBreaker(
            name="agent_session_llm",
            config=CircuitBreaker.LLM_CIRCUIT_CONFIG,
        )
        self.max_tool_rounds = max_tool_rounds
        self.temperature = temperature
        self.max_messages = max_messages

        self._messages: list[LLMMessage] = [
            LLMMessage(role="system", content=system_prompt),
        ]
        self._tool_calls_made: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[LLMMessage]:
        """Current conversation history (read-only view)."""
        return list(self._messages)

    @property
    def tool_calls_made(self) -> list[dict[str, Any]]:
        """Record of all tool calls made in this session."""
        return list(self._tool_calls_made)

    def _prune_messages(self) -> None:
        """Keep system prompt + last max_messages messages."""
        if len(self._messages) <= self.max_messages:
            return
        # Always keep the first message (system prompt)
        system_msg = self._messages[0]
        # Keep the last (max_messages - 1) messages
        self._messages = [system_msg] + self._messages[-(self.max_messages - 1):]

    def turn(self, user_message: str) -> str:
        """Execute one conversational turn.

        Sends the user message to the LLM, processes any tool calls
        in the response, feeds tool results back, and repeats until
        the LLM produces a final text response (no more tool calls).

        Args:
            user_message: The user prompt for this turn.

        Returns:
            The LLM's final text response after all tool calls resolve.

        Raises:
            RuntimeError: If the circuit breaker is open or retries exhausted.
        """
        self._prune_messages()
        self._messages.append(LLMMessage(role="user", content=user_message))

        for _round in range(self.max_tool_rounds):
            if self.budget.exhausted():
                logger.info("Agent session budget exhausted, returning last state")
                return self._last_assistant_content()

            response = self._call_llm()
            content = response.content

            self._messages.append(LLMMessage(role="assistant", content=content))

            tool_calls = self._extract_tool_calls(content)
            if not tool_calls or self.tool_executor is None:
                return content

            tool_results = self._execute_tools(tool_calls)
            results_text = self._format_tool_results(tool_results)
            self._messages.append(
                LLMMessage(role="user", content=results_text)
            )

        return self._last_assistant_content()

    def _call_llm(self) -> LLMResponse:
        """Make a single LLM call with resilience wrappers."""
        from siare.services.circuit_breaker import CircuitBreakerOpenError
        from siare.services.retry_handler import RetryExhausted

        try:
            response: LLMResponse = self.circuit_breaker.call(
                lambda: self.retry_handler.execute_with_retry(
                    self.llm_provider.complete,
                    messages=self._messages,
                    model=self.model,
                    temperature=self.temperature,
                    retry_config=RetryHandler.LLM_RETRY_CONFIG,
                    component="AgentSession",
                    operation="llm_turn",
                )
            )
        except CircuitBreakerOpenError as e:
            raise RuntimeError(
                "Agent session LLM circuit breaker is open"
            ) from e
        except RetryExhausted as e:
            raise RuntimeError(
                "Agent session LLM retries exhausted"
            ) from e

        self.budget.record_llm_call(cost=response.cost)
        return response

    def _extract_tool_calls(
        self, content: str
    ) -> list[dict[str, Any]]:
        """Extract tool calls from LLM response content.

        Supports a simple XML-like format that works across providers:
            <tool_call>
            {"name": "tool_name", "arguments": {...}}
            </tool_call>
        """
        pattern = r"<tool_call>\s*(.*?)\s*</tool_call>"
        matches = re.findall(pattern, content, re.DOTALL)
        calls: list[dict[str, Any]] = []
        for match in matches:
            try:
                parsed = json.loads(match)
                if isinstance(parsed, dict) and "name" in parsed:
                    calls.append(parsed)
            except json.JSONDecodeError:
                logger.warning("Failed to parse tool call: %s", match[:200])
        return calls

    def _execute_tools(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Execute tool calls and collect results."""
        results: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call.get("name", "unknown")
            arguments = call.get("arguments", {})
            try:
                if self.tool_executor is None:
                    raise RuntimeError("tool_executor is required")
                result = self.tool_executor.execute(name, arguments)
                self._tool_calls_made.append({
                    "name": name,
                    "arguments": arguments,
                    "result": result,
                    "error": None,
                })
                results.append({"name": name, "result": result})
            except Exception as e:
                error_msg = f"Tool '{name}' failed: {e}"
                logger.warning(error_msg)
                self._tool_calls_made.append({
                    "name": name,
                    "arguments": arguments,
                    "result": None,
                    "error": str(e),
                })
                results.append({"name": name, "result": error_msg})
        return results

    def _format_tool_results(
        self, results: list[dict[str, Any]]
    ) -> str:
        """Format tool results as a user message for the next LLM turn."""
        parts: list[str] = ["Tool results:"]
        for r in results:
            parts.append(f"\n[{r['name']}]: {r['result']}")
        return "\n".join(parts)

    def _last_assistant_content(self) -> str:
        """Return the last assistant message content, or empty string."""
        for msg in reversed(self._messages):
            if msg.role == "assistant":
                return msg.content
        return ""

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Return tool schemas for inclusion in the system prompt."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self.tools
        ]

    def inject_context(self, context: str) -> None:
        """Inject additional context into the conversation.

        Sent as a user message with a [Context] prefix. Used to inject
        supervisor directives or updated constraints mid-session.
        """
        self._messages.append(
            LLMMessage(role="user", content=f"[Context]: {context}")
        )
