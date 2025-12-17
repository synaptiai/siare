"""File I/O utilities with fault tolerance"""

import contextlib
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any

from siare.services.retry_handler import RetryHandler

logger = logging.getLogger(__name__)


def atomic_write_json(
    file_path: Path,
    data: dict[str, Any],
    retry_handler: RetryHandler,
    component: str,
) -> None:
    """
    Atomically write JSON data to file with retry handling.

    Uses temp file + rename pattern for atomic write to prevent partial writes.
    This ensures that either the entire file is written successfully or the
    original file remains unchanged.

    Args:
        file_path: Target file path
        data: Data to write as JSON
        retry_handler: Retry handler for transient failures
        component: Component name for logging context

    Raises:
        RuntimeError: If write fails after all retries

    Example:
        >>> atomic_write_json(
        ...     Path("data.json"),
        ...     {"key": "value"},
        ...     retry_handler,
        ...     "MyComponent"
        ... )
    """

    def _write_to_temp_and_rename():
        """Inner function for retry wrapping"""
        # Write to temp file first (in same directory for atomic rename)
        temp_fd, temp_path = tempfile.mkstemp(
            dir=file_path.parent,
            prefix=f".{file_path.name}.",
            suffix=".tmp",
        )

        try:
            # Write JSON to temp file
            # Note: temp_fd is a file descriptor (int), not a path
            with open(temp_fd, "w") as f:
                json.dump(data, f, indent=2)

            # Atomic rename (POSIX guarantees atomicity)
            shutil.move(temp_path, file_path)

        except Exception:
            # Cleanup temp file on failure (best effort)
            # Suppress exceptions during cleanup to preserve original error
            with contextlib.suppress(Exception):
                Path(temp_path).unlink(missing_ok=True)
            raise

    # Wrap with retry handler for transient failures
    try:
        retry_handler.execute_with_retry(
            _write_to_temp_and_rename,
            retry_config=RetryHandler.CONFIG_RETRY_CONFIG,
            component=component,
            operation=f"atomic_write_{file_path.name}",
        )
        logger.debug(f"Successfully wrote {file_path}")

    except Exception as e:
        logger.exception(f"Failed to write {file_path} after all retries")
        raise RuntimeError(f"Failed to write {file_path}: {e}") from e
