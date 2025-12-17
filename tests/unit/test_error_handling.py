"""Tests for error handling framework"""

import time
from unittest.mock import Mock

import pytest

from siare.core.models import (
    CircuitBreakerConfig,
    CircuitState,
    ErrorCategory,
    ErrorSeverity,
    RetryConfig,
)
from siare.services.circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from siare.services.error_classifier import ErrorClassifier
from siare.services.retry_handler import RetryExhausted, RetryHandler


# ============================================================================
# ErrorClassifier Tests
# ============================================================================


def test_error_classifier_rate_limit():
    """Test classification of rate limit errors"""
    classifier = ErrorClassifier()

    exc = RuntimeError("rate limit exceeded")
    context = classifier.classify(exc, "llm_service", "complete")

    assert context.category == ErrorCategory.TRANSIENT
    assert context.severity == ErrorSeverity.MEDIUM
    assert context.retryable is True
    assert context.component == "llm_service"
    assert context.operation == "complete"


def test_error_classifier_timeout():
    """Test classification of timeout errors"""
    classifier = ErrorClassifier()

    exc = TimeoutError("Request timed out after 30s")
    context = classifier.classify(exc, "api", "fetch_data")

    assert context.category == ErrorCategory.TRANSIENT
    assert context.severity == ErrorSeverity.MEDIUM
    assert context.retryable is True


def test_error_classifier_authentication():
    """Test classification of auth errors"""
    classifier = ErrorClassifier()

    exc = RuntimeError("401 Unauthorized")
    context = classifier.classify(exc, "api", "authenticate")

    assert context.category == ErrorCategory.PERMANENT
    assert context.severity == ErrorSeverity.HIGH
    assert context.retryable is False


def test_error_classifier_validation():
    """Test classification of validation errors"""
    classifier = ErrorClassifier()

    exc = ValueError("Invalid input format")
    context = classifier.classify(exc, "validator", "check_input")

    assert context.category == ErrorCategory.PERMANENT
    assert context.severity == ErrorSeverity.MEDIUM
    assert context.retryable is False


def test_error_classifier_database_transient():
    """Test classification of transient database errors"""
    classifier = ErrorClassifier()

    exc = RuntimeError("database connection refused")
    context = classifier.classify(exc, "database", "query")

    assert context.category == ErrorCategory.TRANSIENT
    assert context.severity == ErrorSeverity.HIGH
    assert context.retryable is True


def test_error_classifier_database_permanent():
    """Test classification of permanent database errors"""
    classifier = ErrorClassifier()

    exc = RuntimeError("duplicate key constraint violation")
    context = classifier.classify(exc, "database", "insert")

    assert context.category == ErrorCategory.PERMANENT
    assert context.severity == ErrorSeverity.HIGH
    assert context.retryable is False


def test_error_classifier_critical():
    """Test classification of critical errors"""
    classifier = ErrorClassifier()

    exc = RuntimeError("data corruption detected")
    context = classifier.classify(exc, "storage", "verify")

    assert context.category == ErrorCategory.CRITICAL
    assert context.severity == ErrorSeverity.CRITICAL
    assert context.retryable is False


def test_error_classifier_degraded():
    """Test classification of degraded errors"""
    classifier = ErrorClassifier()

    exc = RuntimeError("partial results returned")
    context = classifier.classify(exc, "evaluator", "run_metrics")

    assert context.category == ErrorCategory.DEGRADED
    assert context.severity == ErrorSeverity.MEDIUM
    assert context.retryable is True


def test_error_classifier_unknown_defaults_to_transient():
    """Test that unknown errors default to TRANSIENT"""
    classifier = ErrorClassifier()

    exc = RuntimeError("Something unexpected happened")
    context = classifier.classify(exc, "unknown", "unknown_op")

    assert context.category == ErrorCategory.TRANSIENT
    assert context.severity == ErrorSeverity.MEDIUM
    assert context.retryable is True


def test_error_classifier_from_message():
    """Test classification from message string only"""
    classifier = ErrorClassifier()

    context = classifier.classify_from_message(
        error_message="429 too many requests",
        component="api",
        operation="call",
        exception_type="HTTPError",
    )

    assert context.category == ErrorCategory.TRANSIENT
    assert context.severity == ErrorSeverity.MEDIUM
    assert context.retryable is True
    assert context.stackTrace is None  # No stack trace when classifying from message


# ============================================================================
# RetryHandler Tests
# ============================================================================


def test_retry_handler_success_first_attempt():
    """Test successful execution on first attempt"""
    retry_handler = RetryHandler()

    mock_fn = Mock(return_value="success")

    result = retry_handler.execute_with_retry(
        mock_fn,
        retry_config=RetryConfig(max_attempts=3),
        component="test",
        operation="test_op",
    )

    assert result == "success"
    assert mock_fn.call_count == 1


def test_retry_handler_success_after_retries():
    """Test successful execution after transient failures"""
    retry_handler = RetryHandler()

    # Fail twice, then succeed
    mock_fn = Mock(side_effect=[
        TimeoutError("timeout"),
        TimeoutError("timeout"),
        "success",
    ])

    result = retry_handler.execute_with_retry(
        mock_fn,
        retry_config=RetryConfig(max_attempts=3, base_delay=0.01, jitter=False),
        component="test",
        operation="test_op",
    )

    assert result == "success"
    assert mock_fn.call_count == 3


def test_retry_handler_permanent_error_fails_fast():
    """Test that permanent errors fail immediately without retry"""
    retry_handler = RetryHandler()

    mock_fn = Mock(side_effect=ValueError("Invalid input"))

    with pytest.raises(ValueError, match="Invalid input"):
        retry_handler.execute_with_retry(
            mock_fn,
            retry_config=RetryConfig(max_attempts=5),
            component="test",
            operation="test_op",
        )

    # Should only try once (permanent errors don't retry)
    assert mock_fn.call_count == 1


def test_retry_handler_exhausted_retries():
    """Test that RetryExhausted is raised after all attempts fail"""
    retry_handler = RetryHandler()

    mock_fn = Mock(side_effect=TimeoutError("always timeout"))

    with pytest.raises(RetryExhausted, match="Failed after 3 attempts"):
        retry_handler.execute_with_retry(
            mock_fn,
            retry_config=RetryConfig(max_attempts=3, base_delay=0.01, jitter=False),
            component="test",
            operation="test_op",
        )

    assert mock_fn.call_count == 3


def test_retry_handler_exponential_backoff():
    """Test exponential backoff delays"""
    retry_handler = RetryHandler()

    config = RetryConfig(
        max_attempts=4,
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=10.0,
        jitter=False,
    )

    # Test delay calculations
    assert retry_handler._calculate_delay(1, config) == 1.0  # 1.0 * 2^0
    assert retry_handler._calculate_delay(2, config) == 2.0  # 1.0 * 2^1
    assert retry_handler._calculate_delay(3, config) == 4.0  # 1.0 * 2^2
    assert retry_handler._calculate_delay(4, config) == 8.0  # 1.0 * 2^3


def test_retry_handler_max_delay_cap():
    """Test that delay is capped at max_delay"""
    retry_handler = RetryHandler()

    config = RetryConfig(
        max_attempts=10,
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=5.0,
        jitter=False,
    )

    # Delay would be 1.0 * 2^9 = 512.0, but should be capped at 5.0
    assert retry_handler._calculate_delay(10, config) == 5.0


def test_retry_handler_jitter():
    """Test that jitter adds randomness"""
    retry_handler = RetryHandler()

    config = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        exponential_base=2.0,
        max_delay=10.0,
        jitter=True,
    )

    # With jitter, delays should vary
    delays = [retry_handler._calculate_delay(2, config) for _ in range(10)]

    # All delays should be close to 2.0 but not exactly the same
    assert all(1.5 < d < 2.5 for d in delays)  # ±25% jitter
    assert len(set(delays)) > 1  # Should have variation


def test_retry_handler_should_retry():
    """Test should_retry classification"""
    retry_handler = RetryHandler()

    # Transient errors should retry
    assert retry_handler.should_retry(TimeoutError("timeout"), "test", "op") is True

    # Permanent errors should not retry
    assert retry_handler.should_retry(ValueError("invalid"), "test", "op") is False


def test_retry_handler_with_args_and_kwargs():
    """Test retry with function arguments"""
    retry_handler = RetryHandler()

    mock_fn = Mock(return_value="result")

    result = retry_handler.execute_with_retry(
        mock_fn,
        "arg1",
        "arg2",
        kwarg1="value1",
        retry_config=RetryConfig(max_attempts=2),
        component="test",
        operation="test_op",
    )

    assert result == "result"
    mock_fn.assert_called_once_with("arg1", "arg2", kwarg1="value1")


# ============================================================================
# CircuitBreaker Tests
# ============================================================================


def test_circuit_breaker_closed_state():
    """Test circuit breaker in CLOSED state allows calls"""
    config = CircuitBreakerConfig(failure_threshold=3, timeout=60, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    assert breaker.get_state() == CircuitState.CLOSED

    mock_fn = Mock(return_value="success")
    result = breaker.call(mock_fn)

    assert result == "success"
    assert mock_fn.call_count == 1


def test_circuit_breaker_opens_after_threshold():
    """Test circuit breaker opens after failure threshold"""
    config = CircuitBreakerConfig(failure_threshold=3, timeout=60, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    mock_fn = Mock(side_effect=RuntimeError("fail"))

    # First 3 failures should open the circuit
    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(mock_fn)

    assert breaker.get_state() == CircuitState.OPEN


def test_circuit_breaker_rejects_when_open():
    """Test circuit breaker rejects calls when OPEN"""
    config = CircuitBreakerConfig(failure_threshold=2, timeout=1, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    # Cause failures to open circuit
    mock_fn = Mock(side_effect=RuntimeError("fail"))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(mock_fn)

    assert breaker.get_state() == CircuitState.OPEN

    # Next call should be rejected immediately
    with pytest.raises(CircuitBreakerOpenError, match="is OPEN"):
        breaker.call(mock_fn)

    # mock_fn should only have been called twice (not on the rejected call)
    assert mock_fn.call_count == 2


def test_circuit_breaker_half_open_transition():
    """Test circuit breaker transitions to HALF_OPEN after timeout"""
    config = CircuitBreakerConfig(failure_threshold=2, timeout=1, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    # Open the circuit
    mock_fn = Mock(side_effect=[RuntimeError("fail"), RuntimeError("fail")])
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(mock_fn)

    assert breaker.get_state() == CircuitState.OPEN

    # Wait for timeout
    time.sleep(1.05)

    # Next call should transition to HALF_OPEN
    mock_fn = Mock(return_value="success")
    result = breaker.call(mock_fn)

    assert result == "success"
    assert breaker.get_state() == CircuitState.HALF_OPEN


def test_circuit_breaker_half_open_to_closed():
    """Test circuit breaker closes after successful test calls"""
    config = CircuitBreakerConfig(failure_threshold=2, timeout=1, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    # Open the circuit
    fail_fn = Mock(side_effect=RuntimeError("fail"))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(fail_fn)

    assert breaker.get_state() == CircuitState.OPEN

    # Wait for timeout and make successful calls
    time.sleep(1.05)

    success_fn = Mock(return_value="success")

    # First successful call transitions to HALF_OPEN
    breaker.call(success_fn)
    assert breaker.get_state() == CircuitState.HALF_OPEN

    # Second successful call should close the circuit
    breaker.call(success_fn)
    assert breaker.get_state() == CircuitState.CLOSED


def test_circuit_breaker_half_open_to_open():
    """Test circuit breaker reopens if test call fails"""
    config = CircuitBreakerConfig(failure_threshold=2, timeout=1, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    # Open the circuit
    fail_fn = Mock(side_effect=RuntimeError("fail"))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(fail_fn)

    assert breaker.get_state() == CircuitState.OPEN

    # Wait for timeout
    time.sleep(1.05)

    # Make a successful call (transitions to HALF_OPEN)
    success_fn = Mock(return_value="success")
    breaker.call(success_fn)
    assert breaker.get_state() == CircuitState.HALF_OPEN

    # Failing call should reopen the circuit
    fail_again_fn = Mock(side_effect=RuntimeError("fail again"))
    with pytest.raises(RuntimeError):
        breaker.call(fail_again_fn)

    assert breaker.get_state() == CircuitState.OPEN


def test_circuit_breaker_gradual_recovery_in_closed():
    """Test that successful calls in CLOSED state reduce failure count"""
    config = CircuitBreakerConfig(failure_threshold=5, timeout=60, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    # Cause some failures (but not enough to open)
    fail_fn = Mock(side_effect=RuntimeError("fail"))
    for _ in range(3):
        with pytest.raises(RuntimeError):
            breaker.call(fail_fn)

    assert breaker.failure_count == 3
    assert breaker.get_state() == CircuitState.CLOSED

    # Successful calls should reduce failure count
    success_fn = Mock(return_value="success")
    breaker.call(success_fn)
    assert breaker.failure_count == 2  # Reduced by 1

    breaker.call(success_fn)
    assert breaker.failure_count == 1  # Reduced by 1 again


def test_circuit_breaker_get_stats():
    """Test get_stats returns correct information"""
    config = CircuitBreakerConfig(failure_threshold=3, timeout=60, half_open_max_calls=2)
    breaker = CircuitBreaker("my_circuit", config)

    stats = breaker.get_stats()

    assert stats["name"] == "my_circuit"
    assert stats["state"] == "closed"
    assert stats["failure_count"] == 0
    assert stats["failure_threshold"] == 3
    assert stats["time_until_reset"] is None  # Not open yet


def test_circuit_breaker_manual_reset():
    """Test manual reset of circuit breaker"""
    config = CircuitBreakerConfig(failure_threshold=2, timeout=60, half_open_max_calls=2)
    breaker = CircuitBreaker("test_circuit", config)

    # Open the circuit
    fail_fn = Mock(side_effect=RuntimeError("fail"))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(fail_fn)

    assert breaker.get_state() == CircuitState.OPEN

    # Manually reset
    breaker.reset()

    assert breaker.get_state() == CircuitState.CLOSED
    assert breaker.failure_count == 0


# ============================================================================
# Integration Tests
# ============================================================================


def test_retry_with_circuit_breaker_integration():
    """Test RetryHandler and CircuitBreaker working together"""
    retry_handler = RetryHandler()
    breaker = CircuitBreaker(
        "integration_test",
        CircuitBreakerConfig(failure_threshold=3, timeout=1, half_open_max_calls=2),
    )

    # Function that fails then succeeds
    call_count = [0]

    def flaky_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise TimeoutError("timeout")
        return "success"

    # Wrap with both retry and circuit breaker
    def protected_call():
        return breaker.call(flaky_function)

    result = retry_handler.execute_with_retry(
        protected_call,
        retry_config=RetryConfig(max_attempts=5, base_delay=0.01, jitter=False),
        component="integration",
        operation="test",
    )

    assert result == "success"
    assert call_count[0] == 3  # Failed twice, succeeded on third
    assert breaker.get_state() == CircuitState.CLOSED  # Should still be closed


def test_error_classifier_with_retry_handler():
    """Test ErrorClassifier integrated with RetryHandler"""
    classifier = ErrorClassifier()
    retry_handler = RetryHandler(error_classifier=classifier)

    # Permanent error should not retry
    mock_fn = Mock(side_effect=ValueError("invalid input"))

    with pytest.raises(ValueError):
        retry_handler.execute_with_retry(
            mock_fn,
            retry_config=RetryConfig(max_attempts=5),
            component="test",
            operation="classify_test",
        )

    assert mock_fn.call_count == 1  # No retries for permanent error
