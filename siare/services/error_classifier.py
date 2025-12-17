"""Error classification service for categorizing and analyzing errors"""

import logging
import traceback
from typing import Any, ClassVar, Optional

from siare.core.models import ErrorCategory, ErrorContext, ErrorSeverity


logger = logging.getLogger(__name__)


class ErrorClassifier:
    """
    Classifies errors into categories and severity levels for appropriate handling

    Analyzes exception types, error messages, and context to determine:
    - Error category (TRANSIENT, PERMANENT, DEGRADED, CRITICAL)
    - Severity level (LOW, MEDIUM, HIGH, CRITICAL)
    - Whether error is retryable
    """

    # Keyword patterns for error type detection
    RATE_LIMIT_KEYWORDS: ClassVar[set[str]] = {
        "rate limit",
        "429",
        "quota exceeded",
        "too many requests",
        "ratelimit",
    }
    TIMEOUT_KEYWORDS: ClassVar[set[str]] = {
        "timeout",
        "timed out",
        "deadline exceeded",
        "connection timeout",
    }
    AUTH_KEYWORDS: ClassVar[set[str]] = {
        "401",
        "403",
        "unauthorized",
        "forbidden",
        "authentication",
        "permission denied",
    }
    VALIDATION_KEYWORDS: ClassVar[set[str]] = {
        "validation",
        "invalid",
        "malformed",
        "schema",
        "bad request",
        "400",
    }
    DATABASE_KEYWORDS: ClassVar[set[str]] = {
        "database",
        "connection refused",
        "deadlock",
        "duplicate key",
        "constraint violation",
        "sqlite",
        "postgresql",
    }
    CRITICAL_KEYWORDS: ClassVar[set[str]] = {
        "corruption",
        "data loss",
        "integrity",
        "fatal",
        "panic",
        "abort",
    }

    def classify(
        self,
        exception: Exception,
        component: str,
        operation: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ErrorContext:
        """
        Classify an exception into an ErrorContext

        Args:
            exception: The exception to classify
            component: Name of the component that raised the error
            operation: Name of the operation that failed
            metadata: Optional additional context

        Returns:
            ErrorContext with classification and details
        """
        error_message = str(exception)
        error_message_lower = error_message.lower()
        exception_type = type(exception).__name__

        # Classify by exception type and message content
        category = self._determine_category(exception, error_message_lower, exception_type)
        severity = self._determine_severity(exception, error_message_lower, exception_type, category)
        retryable = self._is_retryable(category, severity)

        # Get stack trace
        stack_trace = "".join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        return ErrorContext(
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            errorMessage=error_message,
            stackTrace=stack_trace,
            metadata=metadata or {},
            retryable=retryable,
        )

    def _determine_category(self, exception: Exception, error_message: str, exception_type: str) -> ErrorCategory:
        """Determine error category based on exception and message"""

        # Check for critical errors first
        if any(keyword in error_message for keyword in self.CRITICAL_KEYWORDS):
            return ErrorCategory.CRITICAL

        # Check for specific exception types
        if exception_type in ("ValueError", "TypeError", "KeyError", "AttributeError"):
            # Programming errors - permanent
            return ErrorCategory.PERMANENT

        if exception_type in ("TimeoutError", "asyncio.TimeoutError"):
            return ErrorCategory.TRANSIENT

        # Check error message keywords
        if any(keyword in error_message for keyword in self.RATE_LIMIT_KEYWORDS):
            return ErrorCategory.TRANSIENT

        if any(keyword in error_message for keyword in self.TIMEOUT_KEYWORDS):
            return ErrorCategory.TRANSIENT

        if any(keyword in error_message for keyword in self.AUTH_KEYWORDS):
            return ErrorCategory.PERMANENT

        if any(keyword in error_message for keyword in self.VALIDATION_KEYWORDS):
            return ErrorCategory.PERMANENT

        if any(keyword in error_message for keyword in self.DATABASE_KEYWORDS):
            # Database errors are often transient (connection issues)
            # but can be permanent (constraint violations)
            if "constraint" in error_message or "duplicate" in error_message:
                return ErrorCategory.PERMANENT
            return ErrorCategory.TRANSIENT

        # Check for partial success indicators
        # Use word boundaries to avoid false matches (e.g., "Something" shouldn't match)
        if "partial" in error_message or " some " in error_message:
            return ErrorCategory.DEGRADED

        # Default to transient for unknown errors
        logger.warning(f"Unknown error type {exception_type}, defaulting to TRANSIENT: {error_message[:100]}")
        return ErrorCategory.TRANSIENT

    def _determine_severity(
        self,
        exception: Exception,
        error_message: str,
        exception_type: str,
        category: ErrorCategory,
    ) -> ErrorSeverity:
        """Determine error severity based on category and context"""

        # CRITICAL category always gets CRITICAL severity
        if category == ErrorCategory.CRITICAL:
            return ErrorSeverity.CRITICAL

        # Check for critical keywords
        if any(keyword in error_message for keyword in self.CRITICAL_KEYWORDS):
            return ErrorSeverity.CRITICAL

        # PERMANENT errors are usually HIGH severity
        if category == ErrorCategory.PERMANENT:
            # Auth errors are HIGH (block user action)
            if any(keyword in error_message for keyword in self.AUTH_KEYWORDS):
                return ErrorSeverity.HIGH
            # Validation errors are MEDIUM (user can fix)
            if any(keyword in error_message for keyword in self.VALIDATION_KEYWORDS):
                return ErrorSeverity.MEDIUM
            # Other permanent errors are HIGH
            return ErrorSeverity.HIGH

        # DEGRADED errors are MEDIUM (partial success)
        if category == ErrorCategory.DEGRADED:
            return ErrorSeverity.MEDIUM

        # TRANSIENT errors severity depends on impact
        if category == ErrorCategory.TRANSIENT:
            # Rate limits and timeouts are MEDIUM (annoying but recoverable)
            if any(keyword in error_message for keyword in self.RATE_LIMIT_KEYWORDS):
                return ErrorSeverity.MEDIUM
            if any(keyword in error_message for keyword in self.TIMEOUT_KEYWORDS):
                return ErrorSeverity.MEDIUM
            # Database connection issues are HIGH (system unavailable)
            if any(keyword in error_message for keyword in self.DATABASE_KEYWORDS):
                return ErrorSeverity.HIGH
            # Unknown transient errors default to MEDIUM (safer assumption)
            return ErrorSeverity.MEDIUM

        # Default to MEDIUM
        return ErrorSeverity.MEDIUM

    def _is_retryable(self, category: ErrorCategory, severity: ErrorSeverity) -> bool:
        """Determine if error should be retried"""

        # Never retry PERMANENT or CRITICAL errors
        if category in (ErrorCategory.PERMANENT, ErrorCategory.CRITICAL):
            return False

        # Always retry TRANSIENT errors
        if category == ErrorCategory.TRANSIENT:
            return True

        # Retry DEGRADED errors (might succeed fully on retry)
        if category == ErrorCategory.DEGRADED:
            return True

        # Default to not retryable
        return False

    def classify_from_message(
        self,
        error_message: str,
        component: str,
        operation: str,
        exception_type: str = "Exception",
        metadata: Optional[dict[str, Any]] = None,
    ) -> ErrorContext:
        """
        Classify error from message string (when exception object not available)

        Args:
            error_message: Error message string
            component: Name of the component
            operation: Name of the operation
            exception_type: Type name of exception
            metadata: Optional additional context

        Returns:
            ErrorContext with classification
        """
        error_message_lower = error_message.lower()

        category = self._determine_category_from_message(error_message_lower, exception_type)
        severity = self._determine_severity_from_message(error_message_lower, exception_type, category)
        retryable = self._is_retryable(category, severity)

        return ErrorContext(
            category=category,
            severity=severity,
            component=component,
            operation=operation,
            errorMessage=error_message,
            stackTrace=None,
            metadata=metadata or {},
            retryable=retryable,
        )

    def _determine_category_from_message(self, error_message: str, exception_type: str) -> ErrorCategory:
        """Determine category from message only (used when exception not available)"""

        if any(keyword in error_message for keyword in self.CRITICAL_KEYWORDS):
            return ErrorCategory.CRITICAL

        if any(keyword in error_message for keyword in self.RATE_LIMIT_KEYWORDS):
            return ErrorCategory.TRANSIENT

        if any(keyword in error_message for keyword in self.TIMEOUT_KEYWORDS):
            return ErrorCategory.TRANSIENT

        if any(keyword in error_message for keyword in self.AUTH_KEYWORDS):
            return ErrorCategory.PERMANENT

        if any(keyword in error_message for keyword in self.VALIDATION_KEYWORDS):
            return ErrorCategory.PERMANENT

        if any(keyword in error_message for keyword in self.DATABASE_KEYWORDS):
            if "constraint" in error_message or "duplicate" in error_message:
                return ErrorCategory.PERMANENT
            return ErrorCategory.TRANSIENT

        # Check for partial success indicators
        if "partial" in error_message or " some " in error_message:
            return ErrorCategory.DEGRADED

        return ErrorCategory.TRANSIENT

    def _determine_severity_from_message(
        self,
        error_message: str,
        exception_type: str,
        category: ErrorCategory,
    ) -> ErrorSeverity:
        """Determine severity from message only"""

        if category == ErrorCategory.CRITICAL:
            return ErrorSeverity.CRITICAL

        if any(keyword in error_message for keyword in self.CRITICAL_KEYWORDS):
            return ErrorSeverity.CRITICAL

        if category == ErrorCategory.PERMANENT:
            if any(keyword in error_message for keyword in self.AUTH_KEYWORDS):
                return ErrorSeverity.HIGH
            if any(keyword in error_message for keyword in self.VALIDATION_KEYWORDS):
                return ErrorSeverity.MEDIUM
            return ErrorSeverity.HIGH

        if category == ErrorCategory.DEGRADED:
            return ErrorSeverity.MEDIUM

        if category == ErrorCategory.TRANSIENT:
            if any(keyword in error_message for keyword in self.RATE_LIMIT_KEYWORDS):
                return ErrorSeverity.MEDIUM
            if any(keyword in error_message for keyword in self.TIMEOUT_KEYWORDS):
                return ErrorSeverity.MEDIUM
            if any(keyword in error_message for keyword in self.DATABASE_KEYWORDS):
                return ErrorSeverity.HIGH
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM
