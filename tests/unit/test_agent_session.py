"""Tests for AgentSession multi-turn LLM conversation with tool dispatch."""

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from siare.core.models import InnerLoopBudget, VariationToolSpec
from siare.services.agent_session import AgentSession, ToolExecutor
from siare.services.llm_provider import LLMMessage, LLMResponse


# ============================================================================
# Test Fixtures
# ============================================================================


def make_llm_response(content: str, cost: float = 0.01) -> LLMResponse:
    """Create a mock LLM response."""
    return LLMResponse(
        content=content,
        model="test-model",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        finish_reason="stop",
        cost=cost,
    )


class MockLLMProvider:
    """Mock LLM provider that returns pre-configured responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._call_idx = 0
        self.calls: list[dict[str, Any]] = []

    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
        })
        idx = min(self._call_idx, len(self._responses) - 1)
        self._call_idx += 1
        return make_llm_response(self._responses[idx])

    def get_model_name(self, model_ref: str) -> str:
        return model_ref


class MockToolExecutor:
    """Mock tool executor that returns pre-configured results."""

    def __init__(self, results: dict[str, str] | None = None) -> None:
        self._results = results or {}
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> str:
        self.calls.append((tool_name, arguments))
        if tool_name in self._results:
            return self._results[tool_name]
        return f"Result for {tool_name}"


# ============================================================================
# Basic Session Tests
# ============================================================================


class TestAgentSessionBasic:
    """Tests for basic session creation and single-turn conversations."""

    def test_creation(self):
        provider = MockLLMProvider(["Hello"])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="You are a test agent.",
        )
        assert len(session.messages) == 1
        assert session.messages[0].role == "system"
        assert session.tool_calls_made == []

    def test_single_turn_no_tools(self):
        provider = MockLLMProvider(["The SOP has weak retrieval."])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="You are a diagnostician.",
        )

        response = session.turn("Analyze this SOP.")
        assert response == "The SOP has weak retrieval."
        assert len(provider.calls) == 1
        assert len(session.messages) == 3  # system + user + assistant

    def test_multi_turn_conversation(self):
        provider = MockLLMProvider([
            "The retrieval is weak.",
            "I suggest adding a re-ranker role.",
        ])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="You are a diagnostician.",
        )

        r1 = session.turn("Analyze this SOP.")
        assert r1 == "The retrieval is weak."

        r2 = session.turn("What mutation would fix this?")
        assert r2 == "I suggest adding a re-ranker role."
        assert len(session.messages) == 5  # sys + u1 + a1 + u2 + a2

    def test_budget_tracking(self):
        provider = MockLLMProvider(["Response"])
        budget = InnerLoopBudget(maxLLMCalls=10)
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            budget=budget,
        )

        session.turn("Hello")
        assert budget.llmCallsUsed == 1

    def test_budget_exhaustion_stops_session(self):
        provider = MockLLMProvider(["Response"])
        budget = InnerLoopBudget(maxLLMCalls=1, llmCallsUsed=1)
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            budget=budget,
        )

        result = session.turn("Hello")
        assert result == ""  # No assistant message yet
        assert len(provider.calls) == 0  # LLM not called


# ============================================================================
# Tool Call Tests
# ============================================================================


class TestAgentSessionToolCalls:
    """Tests for tool call extraction and execution."""

    def test_tool_call_extraction_and_execution(self):
        tool_response = (
            'I will inspect the trace.\n'
            '<tool_call>\n'
            '{"name": "inspect_trace", "arguments": {"trace_id": "t-123"}}\n'
            '</tool_call>'
        )
        provider = MockLLMProvider([
            tool_response,
            "The trace shows retrieval took 5s. Propose PROMPT_CHANGE.",
        ])
        executor = MockToolExecutor({
            "inspect_trace": "Node 'retriever': 5000ms, 3 docs retrieved",
        })

        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tool_executor=executor,
        )

        result = session.turn("Diagnose the SOP.")
        assert "PROMPT_CHANGE" in result
        assert len(executor.calls) == 1
        assert executor.calls[0][0] == "inspect_trace"
        assert session.tool_calls_made[0]["name"] == "inspect_trace"
        assert session.tool_calls_made[0]["error"] is None

    def test_multiple_tool_calls_in_one_response(self):
        tool_response = (
            '<tool_call>\n'
            '{"name": "inspect_trace", "arguments": {"trace_id": "t-1"}}\n'
            '</tool_call>\n'
            '<tool_call>\n'
            '{"name": "query_gene_pool", "arguments": {"query_type": "top_performers"}}\n'
            '</tool_call>'
        )
        provider = MockLLMProvider([
            tool_response,
            "Analysis complete.",
        ])
        executor = MockToolExecutor({
            "inspect_trace": "Trace data",
            "query_gene_pool": "Top 3 SOPs listed",
        })

        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tool_executor=executor,
        )

        result = session.turn("Analyze everything.")
        assert result == "Analysis complete."
        assert len(executor.calls) == 2
        assert len(session.tool_calls_made) == 2

    def test_tool_error_handled_gracefully(self):
        tool_response = (
            '<tool_call>\n'
            '{"name": "failing_tool", "arguments": {}}\n'
            '</tool_call>'
        )
        provider = MockLLMProvider([
            tool_response,
            "Tool failed but I can continue.",
        ])

        class FailingExecutor:
            def execute(self, tool_name: str, arguments: dict) -> str:
                raise RuntimeError("Connection timeout")

        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tool_executor=FailingExecutor(),
        )

        result = session.turn("Try the tool.")
        assert "continue" in result
        assert session.tool_calls_made[0]["error"] == "Connection timeout"

    def test_no_tool_executor_skips_tool_calls(self):
        tool_response = (
            '<tool_call>\n'
            '{"name": "some_tool", "arguments": {}}\n'
            '</tool_call>'
        )
        provider = MockLLMProvider([tool_response])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tool_executor=None,
        )

        result = session.turn("Call a tool.")
        assert "<tool_call>" in result  # Raw response returned

    def test_max_tool_rounds_prevents_infinite_loop(self):
        always_tool = (
            '<tool_call>\n'
            '{"name": "loop_tool", "arguments": {}}\n'
            '</tool_call>'
        )
        provider = MockLLMProvider([always_tool] * 10)
        executor = MockToolExecutor({"loop_tool": "looping"})

        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tool_executor=executor,
            max_tool_rounds=3,
        )

        result = session.turn("Keep calling tools.")
        assert len(executor.calls) == 3  # Stopped after max rounds

    def test_malformed_tool_call_ignored(self):
        bad_response = (
            '<tool_call>\n'
            'this is not json\n'
            '</tool_call>\n'
            'But here is my actual analysis.'
        )
        provider = MockLLMProvider([bad_response])
        executor = MockToolExecutor()

        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tool_executor=executor,
        )

        result = session.turn("Analyze.")
        # No tool calls made (JSON parsing failed)
        assert len(executor.calls) == 0
        assert "actual analysis" in result


# ============================================================================
# Tool Schema Tests
# ============================================================================


class TestAgentSessionToolSchema:
    """Tests for tool schema management."""

    def test_get_tools_schema(self):
        tools = [
            VariationToolSpec(
                name="dry_run",
                description="Execute candidate SOP",
                parameters={"config": {"type": "object"}},
            ),
            VariationToolSpec(
                name="validate_mutation",
                description="Pre-validate mutation",
            ),
        ]
        provider = MockLLMProvider(["OK"])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tools=tools,
        )

        schema = session.get_tools_schema()
        assert len(schema) == 2
        assert schema[0]["name"] == "dry_run"
        assert schema[1]["name"] == "validate_mutation"


# ============================================================================
# Context Injection Tests
# ============================================================================


class TestAgentSessionResilience:
    """Tests for LLM call failure modes."""

    def test_circuit_breaker_open_raises_runtime_error(self):
        from siare.services.circuit_breaker import CircuitBreakerOpenError

        provider = MockLLMProvider(["never reached"])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
        )
        session.circuit_breaker.call = lambda fn: (_ for _ in ()).throw(
            CircuitBreakerOpenError("agent_session_llm")
        )

        with pytest.raises(RuntimeError, match="circuit breaker is open"):
            session.turn("Hello")

    def test_retry_exhausted_raises_runtime_error(self):
        from siare.services.retry_handler import RetryExhausted

        provider = MockLLMProvider(["never reached"])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
        )
        original_call = session.circuit_breaker.call
        session.circuit_breaker.call = lambda fn: (_ for _ in ()).throw(
            RetryExhausted("All retries exhausted")
        )

        with pytest.raises(RuntimeError, match="retries exhausted"):
            session.turn("Hello")

    def test_tool_call_missing_name_key_ignored(self):
        bad_response = (
            '<tool_call>\n'
            '{"arguments": {"trace_id": "t-1"}}\n'
            '</tool_call>\n'
            'Continuing with analysis.'
        )
        provider = MockLLMProvider([bad_response])
        executor = MockToolExecutor()
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            tool_executor=executor,
        )
        result = session.turn("Analyze.")
        assert len(executor.calls) == 0
        assert "Continuing" in result


# ============================================================================
# Message Pruning Tests
# ============================================================================


class TestAgentSessionMessagePruning:
    """Tests for sliding window message history pruning."""

    def test_default_max_messages(self):
        """Default max_messages is 50."""
        provider = MockLLMProvider(["OK"])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
        )
        assert session.max_messages == 50

    def test_custom_max_messages(self):
        """Create session with custom max_messages and verify it's respected."""
        provider = MockLLMProvider(["OK"] * 20)
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            max_messages=10,
        )
        assert session.max_messages == 10

        # Send enough turns to exceed max_messages=10
        # Each turn adds 2 messages (user + assistant), starting with 1 (system)
        for i in range(8):
            session.turn(f"Message {i}")

        # 1 system + 16 user/assistant = 17 messages total before pruning
        # But pruning happens at the start of each turn, so it gets trimmed
        # After final pruning: system + 9 most recent = 10
        assert len(session.messages) <= 10 + 2  # max_messages + last turn pair

    def test_messages_pruned_when_exceeding_max(self):
        """Set max_messages=5, send enough turns to exceed, verify pruning."""
        provider = MockLLMProvider(["OK"] * 20)
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="System prompt here",
            max_messages=5,
        )

        # Do 4 turns: each adds user + assistant = 8 msgs + 1 system = 9 total
        for i in range(4):
            session.turn(f"Turn {i}")

        # After 4th turn, _prune_messages ran at start of turn 4:
        # Before prune on turn 4: 1 system + 6 (3 turns * 2) + 1 user = 8
        #   -> pruned to 5: system + last 4
        # Then turn 4 adds user + assistant = 7
        # The key invariant: messages never grow unbounded
        messages = session.messages
        # System prompt is always first
        assert messages[0].role == "system"
        assert messages[0].content == "System prompt here"
        # Total messages bounded: max_messages + 2 (the turn that just happened)
        assert len(messages) <= 5 + 2

    def test_system_prompt_always_preserved(self):
        """After pruning, first message is always the system prompt."""
        provider = MockLLMProvider(["OK"] * 50)
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="I am the system prompt",
            max_messages=3,
        )

        # Do many turns to trigger pruning multiple times
        for i in range(10):
            session.turn(f"Turn {i}")

        messages = session.messages
        assert messages[0].role == "system"
        assert messages[0].content == "I am the system prompt"

    def test_no_pruning_when_under_limit(self):
        """Messages are not pruned when count is within max_messages."""
        provider = MockLLMProvider(["OK", "OK"])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
            max_messages=50,
        )

        session.turn("Hello")
        session.turn("World")
        # 1 system + 2 user + 2 assistant = 5, well under 50
        assert len(session.messages) == 5

    def test_pruning_preserves_most_recent_messages(self):
        """After pruning, the most recent messages are preserved."""
        provider = MockLLMProvider(["OK"] * 20)
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="System",
            max_messages=5,
        )

        # Do several turns
        for i in range(6):
            session.turn(f"Turn {i}")

        messages = session.messages
        # The last assistant message should be from the most recent turn
        last_assistant = [m for m in messages if m.role == "assistant"][-1]
        assert last_assistant.content == "OK"
        # The last user message content should be the most recent turn input
        user_msgs = [m for m in messages if m.role == "user"]
        assert user_msgs[-1].content == "Turn 5"


class TestAgentSessionContextInjection:
    """Tests for mid-session context injection."""

    def test_inject_context(self):
        provider = MockLLMProvider(["OK", "Understood"])
        session = AgentSession(
            llm_provider=provider,
            model="test-model",
            system_prompt="Test",
        )

        session.turn("Hello")
        session.inject_context("Focus on topology mutations now.")

        messages = session.messages
        context_msg = messages[-1]
        assert context_msg.role == "user"
        assert "Focus on topology mutations" in context_msg.content
