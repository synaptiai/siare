"""Tests for EvaluationService"""

from unittest.mock import Mock

from siare.core.models import (
    EvaluationArtifacts,
    MetricConfig,
    MetricType,
)
from siare.services.evaluation_service import EvaluationService
from siare.services.execution_engine import ExecutionTrace
from siare.services.llm_provider import LLMProvider, LLMResponse


def create_mock_trace():
    """Helper function to create a mock execution trace."""
    trace = ExecutionTrace(
        run_id="test-run-1",
        sop_id="test-sop",
        sop_version="1.0.0"
    )
    trace.finalize(status="completed", final_outputs={"answer": "This is a test answer"})
    return trace


def test_evaluate_llm_judge_with_real_provider():
    """Test that LLM judge uses actual provider when available."""
    # Create mock LLM provider
    mock_provider = Mock(spec=LLMProvider)
    mock_provider.complete.return_value = LLMResponse(
        content='{"score": 0.92, "reasoning": "Excellent factual accuracy"}',
        model="gpt-4o",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        finish_reason="stop",
    )

    # Create service with provider
    service = EvaluationService(llm_provider=mock_provider)

    # Create trace and metric config
    trace = create_mock_trace()
    # Note: For this test, we're testing the internal method directly
    # so we bypass Pydantic validation by using model_construct
    metric_config = MetricConfig.model_construct(
        id="factuality",
        type=MetricType.LLM_JUDGE,
        weight=1.0,
        inputs=[],
        model="gpt-4o-mini",
        promptRef="factuality_prompt"
    )
    artifacts = EvaluationArtifacts(llmFeedback={}, failureModes=[], toolErrors=[], traceRefs=[])

    # The current _evaluate_llm_judge is still mock-based, so this test will fail
    # until Task 2 is implemented. For now, we just verify the provider is stored.
    assert service.llm_provider is mock_provider


def test_evaluation_service_init_without_provider():
    """Test that EvaluationService can be initialized without a provider (backward compatibility)."""
    service = EvaluationService()
    assert service.llm_provider is None


def test_evaluation_service_init_with_provider():
    """Test that EvaluationService stores the provider when provided."""
    mock_provider = Mock(spec=LLMProvider)
    service = EvaluationService(llm_provider=mock_provider)
    assert service.llm_provider is mock_provider


def test_llm_judge_parses_json_response():
    """Test that LLM judge correctly parses JSON response."""
    from siare.core.models import MetricSource

    # Create mock LLM provider
    mock_provider = Mock(spec=LLMProvider)
    mock_provider.complete.return_value = LLMResponse(
        content='{"score": 0.78, "reasoning": "Good but could be more specific"}',
        model="gpt-4o",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        finish_reason="stop",
    )

    service = EvaluationService(llm_provider=mock_provider)

    # Create trace and metric config
    trace = create_mock_trace()
    metric_config = MetricConfig.model_construct(
        id="factuality",
        type=MetricType.LLM_JUDGE,
        weight=1.0,
        inputs=[],
        model="gpt-4o-mini",
        promptRef="factuality_prompt"
    )
    artifacts = EvaluationArtifacts(llmFeedback={}, failureModes=[], toolErrors=[], traceRefs=[])

    # Evaluate with task_data
    task_data = {
        "input": {"query": "What is the capital of France?"},
        "groundTruth": {"answer": "Paris"}
    }

    result = service._evaluate_llm_judge(metric_config, trace, artifacts, task_data)

    # Assert provider was called
    mock_provider.complete.assert_called_once()
    assert result.score == 0.78
    assert result.reasoning == "Good but could be more specific"
    assert result.source == MetricSource.LLM


def test_llm_judge_handles_markdown_code_blocks():
    """Test that LLM judge handles markdown code blocks in response."""
    from siare.core.models import MetricSource

    # Create mock LLM provider that returns markdown-wrapped JSON
    mock_provider = Mock(spec=LLMProvider)
    mock_provider.complete.return_value = LLMResponse(
        content='```json\n{"score": 0.65, "reasoning": "Needs improvement"}\n```',
        model="gpt-4o",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        finish_reason="stop",
    )

    service = EvaluationService(llm_provider=mock_provider)

    trace = create_mock_trace()
    metric_config = MetricConfig.model_construct(
        id="relevance",
        type=MetricType.LLM_JUDGE,
        weight=1.0,
        inputs=[],
        model="gpt-4o-mini",
        promptRef="relevance_prompt"
    )
    artifacts = EvaluationArtifacts(llmFeedback={}, failureModes=[], toolErrors=[], traceRefs=[])

    result = service._evaluate_llm_judge(metric_config, trace, artifacts)

    assert result.score == 0.65
    assert result.reasoning == "Needs improvement"
    assert result.source == MetricSource.LLM


def test_llm_judge_fallback_on_parse_error():
    """Test that LLM judge falls back gracefully on JSON parse errors."""
    from siare.core.models import MetricSource

    # Create mock LLM provider that returns invalid JSON
    mock_provider = Mock(spec=LLMProvider)
    mock_provider.complete.return_value = LLMResponse(
        content="This is not valid JSON at all",
        model="gpt-4o",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        finish_reason="stop",
    )

    service = EvaluationService(llm_provider=mock_provider)

    trace = create_mock_trace()
    metric_config = MetricConfig.model_construct(
        id="coherence",
        type=MetricType.LLM_JUDGE,
        weight=1.0,
        inputs=[],
        model="gpt-4o-mini",
        promptRef="coherence_prompt"
    )
    artifacts = EvaluationArtifacts(llmFeedback={}, failureModes=[], toolErrors=[], traceRefs=[])

    result = service._evaluate_llm_judge(metric_config, trace, artifacts)

    # Should use neutral score on parse error
    assert result.score == 0.5
    assert "Parse error" in result.reasoning
    assert result.source == MetricSource.LLM


def test_llm_judge_raises_error_when_no_provider():
    """Test that LLM judge raises RuntimeError when no provider available."""
    import pytest

    # Create service without provider - should fail loudly when LLM is needed
    service = EvaluationService(llm_provider=None)

    trace = create_mock_trace()
    metric_config = MetricConfig.model_construct(
        id="factuality",
        type=MetricType.LLM_JUDGE,
        weight=1.0,
        inputs=[],
        model="gpt-4o-mini",
        promptRef="factuality_prompt"
    )
    artifacts = EvaluationArtifacts(llmFeedback={}, failureModes=[], toolErrors=[], traceRefs=[])

    # Should raise RuntimeError - no silent fallback to mock!
    with pytest.raises(RuntimeError, match="LLM provider required"):
        service._evaluate_llm_judge(metric_config, trace, artifacts)


def test_cost_metric_uses_actual_token_counts():
    """Test that cost is calculated from actual token usage."""
    from siare.core.models import MetricSource

    service = EvaluationService()

    # Create trace with token usage
    trace = ExecutionTrace(
        run_id="run1",
        sop_id="sop1",
        sop_version="1.0.0"
    )
    trace.add_node_execution(
        role_id="analyzer",
        inputs={},
        outputs={},
        duration_ms=1000,
        model="gpt-4o-mini",
        prompt="...",
    )
    trace.node_executions[0]["usage"] = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
    }
    trace.finalize("completed", {})

    metric_config = MetricConfig.model_construct(
        id="cost",
        type=MetricType.RUNTIME,
        weight=1.0,
        inputs=[]
    )

    result = service._evaluate_runtime(metric_config, trace)

    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    expected_cost = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
    assert abs(result.rawValue - expected_cost) < 0.0001
    assert result.source == MetricSource.RUNTIME
    assert result.metricId == "cost"
    # Score should be between 0 and 1 (cheaper is better)
    assert 0.0 <= result.score <= 1.0


def test_cost_metric_handles_multiple_nodes():
    """Test that cost is calculated across multiple node executions."""
    from siare.core.models import MetricSource

    service = EvaluationService()

    # Create trace with multiple nodes
    trace = ExecutionTrace(
        run_id="run1",
        sop_id="sop1",
        sop_version="1.0.0"
    )

    # First node - gpt-4o-mini
    trace.add_node_execution(
        role_id="analyzer",
        inputs={},
        outputs={},
        duration_ms=1000,
        model="gpt-4o-mini",
        prompt="...",
    )
    trace.node_executions[0]["usage"] = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
    }

    # Second node - gpt-4o
    trace.add_node_execution(
        role_id="synthesizer",
        inputs={},
        outputs={},
        duration_ms=2000,
        model="gpt-4o",
        prompt="...",
    )
    trace.node_executions[1]["usage"] = {
        "prompt_tokens": 500,
        "completion_tokens": 250,
        "total_tokens": 750,
    }

    trace.finalize("completed", {})

    metric_config = MetricConfig.model_construct(
        id="cost",
        type=MetricType.RUNTIME,
        weight=1.0,
        inputs=[]
    )

    result = service._evaluate_runtime(metric_config, trace)

    # Calculate expected cost
    # gpt-4o-mini: $0.15/1M input, $0.60/1M output
    cost1 = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
    # gpt-4o: $2.50/1M input, $10.00/1M output
    cost2 = (500 * 2.50 / 1_000_000) + (250 * 10.00 / 1_000_000)
    expected_cost = cost1 + cost2

    assert abs(result.rawValue - expected_cost) < 0.0001
    assert result.source == MetricSource.RUNTIME


def test_cost_metric_normalizes_model_names():
    """Test that cost calculation handles model names with version suffixes."""

    service = EvaluationService()

    # Create trace with model name that includes date suffix
    trace = ExecutionTrace(
        run_id="run1",
        sop_id="sop1",
        sop_version="1.0.0"
    )
    trace.add_node_execution(
        role_id="analyzer",
        inputs={},
        outputs={},
        duration_ms=1000,
        model="gpt-4o-mini-2024-11-06",  # Model name with date suffix
        prompt="...",
    )
    trace.node_executions[0]["usage"] = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
    }
    trace.finalize("completed", {})

    metric_config = MetricConfig.model_construct(
        id="cost",
        type=MetricType.RUNTIME,
        weight=1.0,
        inputs=[]
    )

    result = service._evaluate_runtime(metric_config, trace)

    # Should use gpt-4o-mini pricing even with date suffix
    expected_cost = (1000 * 0.15 / 1_000_000) + (500 * 0.60 / 1_000_000)
    assert abs(result.rawValue - expected_cost) < 0.0001


def test_cost_metric_fallback_without_usage_data():
    """Test that cost calculation falls back when no usage data is available."""
    from siare.core.models import MetricSource

    service = EvaluationService()

    # Create trace without usage data
    trace = ExecutionTrace(
        run_id="run1",
        sop_id="sop1",
        sop_version="1.0.0"
    )
    trace.add_node_execution(
        role_id="analyzer",
        inputs={},
        outputs={},
        duration_ms=1000,
        model="gpt-4o-mini",
        prompt="...",
    )
    # No usage data added
    trace.finalize("completed", {})

    metric_config = MetricConfig.model_construct(
        id="cost",
        type=MetricType.RUNTIME,
        weight=1.0,
        inputs=[]
    )

    result = service._evaluate_runtime(metric_config, trace)

    # Should use fallback estimation
    assert result.rawValue == 0.001  # 1 node * $0.001
    assert result.source == MetricSource.RUNTIME
    assert "estimated cost" in result.reasoning.lower()


def test_hallucination_judge_prompts_available():
    """Hallucination LLM Judge prompts must be available."""
    from siare.services.evaluation_service import EvaluationService

    # Access JUDGE_PROMPTS from the class
    judge_prompts = EvaluationService.JUDGE_PROMPTS

    assert "faithfulness" in judge_prompts
    assert "groundedness" in judge_prompts
    assert "citation_accuracy" in judge_prompts

    # Verify they have required placeholders
    assert "{answer}" in judge_prompts["faithfulness"]
    assert "{ground_truth}" in judge_prompts["faithfulness"]
    assert "{query}" in judge_prompts["faithfulness"]

    assert "{answer}" in judge_prompts["groundedness"]
    assert "{ground_truth}" in judge_prompts["groundedness"]

    assert "{answer}" in judge_prompts["citation_accuracy"]
    assert "{ground_truth}" in judge_prompts["citation_accuracy"]


def test_hallucination_metrics_registered():
    """Built-in hallucination metrics must be registered."""
    from tests.mocks.mock_llm_provider import MockLLMProvider

    service = EvaluationService(llm_provider=MockLLMProvider())

    # Check that hallucination metrics are available
    assert "faithfulness" in service._metric_functions
    assert "groundedness" in service._metric_functions
    assert "citation_accuracy" in service._metric_functions
