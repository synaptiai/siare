"""Circuit breaker pattern for preventing cascading failures"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

from siare.core.models import CircuitBreakerConfig, CircuitState

logger = logging.getLogger(__name__)

T = TypeVar("T")


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open and rejects a request"""



class CircuitBreaker:
    """
    Circuit breaker pattern implementation

    States:
    - CLOSED: Normal operation, all requests pass through
    - OPEN: Too many failures, reject all requests immediately
    - HALF_OPEN: Testing recovery, allow limited requests

    State Transitions:
    - CLOSED → OPEN: When failure_count >= failure_threshold
    - OPEN → HALF_OPEN: When time_since_last_failure >= timeout
    - HALF_OPEN → CLOSED: When success_count >= half_open_max_calls
    - HALF_OPEN → OPEN: When any failure occurs during test
    """

    # Predefined configurations for common use cases
    LLM_CIRCUIT_CONFIG = CircuitBreakerConfig(
        failure_threshold=10,
        timeout=300,  # 5 minutes
        half_open_max_calls=5,
    )

    DATABASE_CIRCUIT_CONFIG = CircuitBreakerConfig(
        failure_threshold=5,
        timeout=60,  # 1 minute
        half_open_max_calls=3,
    )

    TOOL_CIRCUIT_CONFIG = CircuitBreakerConfig(
        failure_threshold=5,
        timeout=120,  # 2 minutes
        half_open_max_calls=3,
    )

    def __init__(self, name: str, config: CircuitBreakerConfig):
        """
        Initialize circuit breaker

        Args:
            name: Name of the circuit (for logging)
            config: Circuit breaker configuration
        """
        self.name = name
        self.config = config

        # State management
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.success_count: int = 0
        self.last_failure_time: float | None = None

        # Thread safety
        self._lock = threading.Lock()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={config.failure_threshold}, timeout={config.timeout}s"
        )

    def call(
        self,
        fn: Callable[..., T],
        *args: Any,
        **kwargs: Any,
    ) -> T:
        """
        Execute function through circuit breaker

        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from fn

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from fn
        """
        with self._lock:
            # Check if circuit should transition from OPEN to HALF_OPEN
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(f"Circuit breaker '{self.name}' transitioning OPEN → HALF_OPEN (testing recovery)")
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    # Circuit still open, reject immediately
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN "
                        f"(will retry in {self._time_until_reset():.1f}s)"
                    )

            # Check if we're in HALF_OPEN and already have enough test calls
            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.half_open_max_calls:
                    # Already have enough successful test calls, close circuit
                    logger.info(
                        f"Circuit breaker '{self.name}' transitioning HALF_OPEN → CLOSED "
                        f"({self.success_count} successful test calls)"
                    )
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

        # Execute the function
        try:
            result = fn(*args, **kwargs)

            # Record success
            with self._lock:
                self._record_success()

            return result

        except Exception:
            # Record failure
            with self._lock:
                self._record_failure()

            raise

    async def call_async(
        self,
        fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Execute async function through circuit breaker

        Args:
            fn: Async function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments

        Returns:
            Result from fn

        Raises:
            CircuitBreakerOpenError: If circuit is open
            Exception: Any exception from fn
        """
        with self._lock:
            if self.state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    logger.info(
                        f"Circuit breaker '{self.name}' transitioning OPEN → HALF_OPEN (async, testing recovery)"
                    )
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError(
                        f"Circuit breaker '{self.name}' is OPEN (async, will retry in {self._time_until_reset():.1f}s)"
                    )

            if self.state == CircuitState.HALF_OPEN:
                if self.success_count >= self.config.half_open_max_calls:
                    logger.info(
                        f"Circuit breaker '{self.name}' transitioning HALF_OPEN → CLOSED (async, {self.success_count} successful)"
                    )
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0

        # Execute the function
        try:
            result = await fn(*args, **kwargs)

            with self._lock:
                self._record_success()

            return result

        except Exception:
            with self._lock:
                self._record_failure()

            raise

    def _record_success(self) -> None:
        """Record successful call (must hold lock)"""
        if self.state == CircuitState.HALF_OPEN:
            # In HALF_OPEN, increment success count
            self.success_count += 1
            logger.debug(
                f"Circuit breaker '{self.name}' test call succeeded "
                f"({self.success_count}/{self.config.half_open_max_calls})"
            )

            # If we have enough successful calls, close the circuit
            if self.success_count >= self.config.half_open_max_calls:
                logger.info(
                    f"Circuit breaker '{self.name}' transitioning HALF_OPEN → CLOSED "
                    f"({self.success_count} successful test calls)"
                )
                self.state = CircuitState.CLOSED
                self.failure_count = 0

        elif self.state == CircuitState.CLOSED:
            # In CLOSED, gradually reduce failure count on success
            if self.failure_count > 0:
                self.failure_count -= 1
                logger.debug(
                    f"Circuit breaker '{self.name}' success, failure count reduced to {self.failure_count}"
                )

    def _record_failure(self) -> None:
        """Record failed call (must hold lock)"""
        self.last_failure_time = time.time()

        if self.state == CircuitState.HALF_OPEN:
            # In HALF_OPEN, any failure opens the circuit again
            logger.warning(
                f"Circuit breaker '{self.name}' transitioning HALF_OPEN → OPEN (test call failed)"
            )
            self.state = CircuitState.OPEN
            self.failure_count = self.config.failure_threshold  # Reset to threshold

        elif self.state == CircuitState.CLOSED:
            # In CLOSED, increment failure count
            self.failure_count += 1
            logger.debug(
                f"Circuit breaker '{self.name}' failure recorded "
                f"({self.failure_count}/{self.config.failure_threshold})"
            )

            # Check if we should open the circuit
            if self.failure_count >= self.config.failure_threshold:
                logger.error(
                    f"Circuit breaker '{self.name}' transitioning CLOSED → OPEN "
                    f"({self.failure_count} failures >= threshold {self.config.failure_threshold})"
                )
                self.state = CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset (must hold lock)"""
        if self.last_failure_time is None:
            return False

        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.config.timeout

    def _time_until_reset(self) -> float:
        """Calculate time until reset attempt (must hold lock)"""
        if self.last_failure_time is None:
            return 0.0

        time_since_failure = time.time() - self.last_failure_time
        return max(0.0, self.config.timeout - time_since_failure)

    def get_state(self) -> CircuitState:
        """
        Get current circuit state (thread-safe)

        Returns:
            Current circuit state
        """
        with self._lock:
            return self.state

    def get_stats(self) -> dict[str, Any]:
        """
        Get circuit breaker statistics (thread-safe)

        Returns:
            Dictionary with current statistics
        """
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "failure_threshold": self.config.failure_threshold,
                "time_until_reset": self._time_until_reset() if self.state == CircuitState.OPEN else None,
            }

    def reset(self) -> None:
        """
        Manually reset circuit breaker to CLOSED state (thread-safe)

        Use with caution - this should typically only be used for
        administrative purposes or testing
        """
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None

            logger.warning(
                f"Circuit breaker '{self.name}' manually reset from {old_state.value} to CLOSED"
            )


class CircuitBreakerRegistry:
    """
    Registry for managing multiple circuit breakers

    Provides centralized access to circuit breakers with
    automatic creation and configuration
    """

    def __init__(self):
        """Initialize empty registry"""
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = threading.Lock()

    def get_or_create(
        self,
        name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """
        Get existing circuit breaker or create new one

        Args:
            name: Circuit breaker name
            config: Configuration (uses default if not provided)

        Returns:
            CircuitBreaker instance
        """
        with self._lock:
            if name not in self._breakers:
                if config is None:
                    config = CircuitBreaker.DATABASE_CIRCUIT_CONFIG

                self._breakers[name] = CircuitBreaker(name, config)

            return self._breakers[name]

    def get(self, name: str) -> CircuitBreaker | None:
        """
        Get existing circuit breaker

        Args:
            name: Circuit breaker name

        Returns:
            CircuitBreaker instance or None if not found
        """
        with self._lock:
            return self._breakers.get(name)

    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """
        Get statistics for all circuit breakers

        Returns:
            Dictionary mapping circuit names to their stats
        """
        with self._lock:
            return {name: breaker.get_stats() for name, breaker in self._breakers.items()}

    def reset_all(self) -> None:
        """Reset all circuit breakers to CLOSED state"""
        with self._lock:
            for breaker in self._breakers.values():
                breaker.reset()


# Global registry instance
_global_registry: CircuitBreakerRegistry | None = None


def get_circuit_breaker_registry() -> CircuitBreakerRegistry:
    """
    Get global circuit breaker registry (singleton)

    Returns:
        Global CircuitBreakerRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = CircuitBreakerRegistry()
    return _global_registry
