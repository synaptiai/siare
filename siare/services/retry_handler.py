"""Retry handler with exponential backoff for transient failures"""

import logging
import random
import time
from collections.abc import Callable
from typing import Any, TypeVar

from siare.core.models import RetryConfig
from siare.services.error_classifier import ErrorClassifier

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryExhausted(Exception):
    """Raised when all retry attempts are exhausted"""



class RetryHandler:
    """
    Handles retry logic with exponential backoff and jitter

    Features:
    - Exponential backoff: delay = base_delay * (exponential_base ^ attempt)
    - Jitter: Random variation (±25%) to prevent thundering herd
    - Max delay cap to prevent indefinite waits
    - Automatic classification of errors
    - Skip retries for PERMANENT errors
    """

    # Predefined retry configurations for common use cases
    LLM_RETRY_CONFIG = RetryConfig(
        max_attempts=5,
        base_delay=2.0,
        max_delay=60.0,
        exponential_base=2.0,
        jitter=True,
    )

    DATABASE_RETRY_CONFIG = RetryConfig(
        max_attempts=3,
        base_delay=0.5,
        max_delay=10.0,
        exponential_base=2.0,
        jitter=True,
    )

    TOOL_RETRY_CONFIG = RetryConfig(
        max_attempts=3,
        base_delay=1.0,
        max_delay=30.0,
        exponential_base=2.0,
        jitter=True,
    )

    CONFIG_RETRY_CONFIG = RetryConfig(
        max_attempts=2,
        base_delay=0.5,
        max_delay=5.0,
        exponential_base=2.0,
        jitter=False,
    )

    def __init__(self, error_classifier: ErrorClassifier | None = None):
        """
        Initialize retry handler

        Args:
            error_classifier: Optional error classifier for categorizing exceptions
        """
        self.error_classifier = error_classifier or ErrorClassifier()

    def execute_with_retry(
        self,
        fn: Callable[..., T],
        *args: Any,
        retry_config: RetryConfig = DATABASE_RETRY_CONFIG,
        component: str = "unknown",
        operation: str = "unknown",
        **kwargs: Any,
    ) -> T:
        """
        Execute function with retry logic

        Args:
            fn: Function to execute
            *args: Positional arguments for fn
            retry_config: Retry configuration
            component: Component name for logging
            operation: Operation name for logging
            **kwargs: Keyword arguments for fn

        Returns:
            Result from fn

        Raises:
            RetryExhausted: If all retry attempts fail
            Exception: If error is PERMANENT (fail fast)
        """
        last_exception: Exception | None = None

        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                result = fn(*args, **kwargs)

                # Log successful recovery if this wasn't the first attempt
                if attempt > 1:
                    logger.info(
                        f"[{component}.{operation}] Succeeded on attempt {attempt}/{retry_config.max_attempts}"
                    )

                return result

            except Exception as e:
                last_exception = e

                # Classify the error
                error_context = self.error_classifier.classify(
                    exception=e,
                    component=component,
                    operation=operation,
                    metadata={"attempt": attempt, "max_attempts": retry_config.max_attempts},
                )

                # If error is not retryable (PERMANENT or CRITICAL), fail fast
                if not error_context.retryable:
                    logger.error(
                        f"[{component}.{operation}] Non-retryable error ({error_context.category.value}): {e}"
                    )
                    raise

                # If this was the last attempt, raise RetryExhausted
                if attempt >= retry_config.max_attempts:
                    logger.error(
                        f"[{component}.{operation}] All {retry_config.max_attempts} retry attempts exhausted: {e}"
                    )
                    raise RetryExhausted(
                        f"Failed after {retry_config.max_attempts} attempts: {e}"
                    ) from last_exception

                # Calculate delay and wait
                delay = self._calculate_delay(attempt, retry_config)
                logger.warning(
                    f"[{component}.{operation}] Attempt {attempt}/{retry_config.max_attempts} failed "
                    f"({error_context.category.value}): {e}. Retrying in {delay:.2f}s..."
                )

                time.sleep(delay)

        # Should never reach here, but raise just in case
        if last_exception:
            raise RetryExhausted(
                f"Failed after {retry_config.max_attempts} attempts: {last_exception}"
            ) from last_exception
        raise RetryExhausted(f"Failed after {retry_config.max_attempts} attempts")

    async def execute_with_retry_async(
        self,
        fn: Callable[..., Any],
        *args: Any,
        retry_config: RetryConfig = DATABASE_RETRY_CONFIG,
        component: str = "unknown",
        operation: str = "unknown",
        **kwargs: Any,
    ) -> Any:
        """
        Execute async function with retry logic

        Args:
            fn: Async function to execute
            *args: Positional arguments for fn
            retry_config: Retry configuration
            component: Component name for logging
            operation: Operation name for logging
            **kwargs: Keyword arguments for fn

        Returns:
            Result from fn

        Raises:
            RetryExhausted: If all retry attempts fail
            Exception: If error is PERMANENT (fail fast)
        """
        import asyncio

        last_exception: Exception | None = None

        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                result = await fn(*args, **kwargs)

                if attempt > 1:
                    logger.info(
                        f"[{component}.{operation}] Async succeeded on attempt {attempt}/{retry_config.max_attempts}"
                    )

                return result

            except Exception as e:
                last_exception = e

                # Classify the error
                error_context = self.error_classifier.classify(
                    exception=e,
                    component=component,
                    operation=operation,
                    metadata={"attempt": attempt, "max_attempts": retry_config.max_attempts, "async": True},
                )

                # If error is not retryable, fail fast
                if not error_context.retryable:
                    logger.error(
                        f"[{component}.{operation}] Non-retryable async error ({error_context.category.value}): {e}"
                    )
                    raise

                # If this was the last attempt, raise RetryExhausted
                if attempt >= retry_config.max_attempts:
                    logger.error(
                        f"[{component}.{operation}] All {retry_config.max_attempts} async retry attempts exhausted: {e}"
                    )
                    raise RetryExhausted(
                        f"Async failed after {retry_config.max_attempts} attempts: {e}"
                    ) from last_exception

                # Calculate delay and wait
                delay = self._calculate_delay(attempt, retry_config)
                logger.warning(
                    f"[{component}.{operation}] Async attempt {attempt}/{retry_config.max_attempts} failed "
                    f"({error_context.category.value}): {e}. Retrying in {delay:.2f}s..."
                )

                await asyncio.sleep(delay)

        # Should never reach here
        if last_exception:
            raise RetryExhausted(
                f"Async failed after {retry_config.max_attempts} attempts: {last_exception}"
            ) from last_exception
        raise RetryExhausted(f"Async failed after {retry_config.max_attempts} attempts")

    def _calculate_delay(self, attempt: int, config: RetryConfig) -> float:
        """
        Calculate delay for exponential backoff with optional jitter

        Formula: delay = base_delay * (exponential_base ^ (attempt - 1))
        Capped at max_delay
        Jitter adds ±25% random variation

        Args:
            attempt: Current attempt number (1-indexed)
            config: Retry configuration

        Returns:
            Delay in seconds
        """
        # Exponential backoff: delay = base * (base^(attempt-1))
        delay = config.base_delay * (config.exponential_base ** (attempt - 1))

        # Cap at max delay
        delay = min(delay, config.max_delay)

        # Add jitter if configured (±25%)
        if config.jitter:
            jitter_amount = delay * 0.25
            delay += random.uniform(-jitter_amount, jitter_amount)

            # Ensure delay is never negative
            delay = max(0.1, delay)

        return delay

    def should_retry(self, exception: Exception, component: str, operation: str) -> bool:
        """
        Check if an exception should be retried

        Args:
            exception: Exception to check
            component: Component name
            operation: Operation name

        Returns:
            True if should retry, False otherwise
        """
        error_context = self.error_classifier.classify(
            exception=exception,
            component=component,
            operation=operation,
        )

        return error_context.retryable
