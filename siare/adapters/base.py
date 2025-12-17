"""Base Tool Adapter - Plugin system for tools"""

from abc import ABC, abstractmethod
from typing import Any, Optional


class ToolAdapter(ABC):
    """
    Abstract base class for tool adapters

    Tool adapters provide standardized interfaces for external systems
    (vector stores, SQL databases, web search, etc.)
    """

    def __init__(self, config: dict[str, Any]):
        """
        Initialize adapter with configuration

        Args:
            config: Configuration dictionary specific to the adapter
        """
        self.config = config
        self.is_initialized = False

    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the adapter (connect to services, load resources, etc.)

        Should set self.is_initialized = True on success
        """

    @abstractmethod
    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Execute the tool with given inputs

        Args:
            inputs: Input parameters for the tool

        Returns:
            Dictionary with tool outputs
        """

    @abstractmethod
    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """
        Validate input parameters

        Args:
            inputs: Input parameters to validate

        Returns:
            List of validation error messages (empty if valid)
        """

    def cleanup(self) -> None:
        """
        Cleanup resources (connections, temporary files, etc.)

        Optional - override if needed
        """

    def get_schema(self) -> dict[str, Any]:
        """
        Get JSON schema for tool inputs/outputs

        Optional - override to provide schema for validation

        Returns:
            JSON schema dict
        """
        return {
            "inputs": {},
            "outputs": {},
        }

    def __enter__(self):
        """Context manager entry"""
        if not self.is_initialized:
            self.initialize()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        """Context manager exit"""
        self.cleanup()


class ToolRegistry:
    """
    Registry for tool adapter plugins

    Allows dynamic registration and creation of tool adapters
    """

    def __init__(self):
        """Initialize registry"""
        self._adapters: dict[str, type[ToolAdapter]] = {}

    def register(self, tool_type: str, adapter_class: type[ToolAdapter]) -> None:
        """
        Register a tool adapter class

        Args:
            tool_type: Unique identifier for the tool type
            adapter_class: ToolAdapter subclass
        """
        if not issubclass(adapter_class, ToolAdapter):  # type: ignore[arg-type]
            raise TypeError(f"{adapter_class} must be a subclass of ToolAdapter")

        self._adapters[tool_type] = adapter_class

    def unregister(self, tool_type: str) -> None:
        """Unregister a tool adapter"""
        if tool_type in self._adapters:
            del self._adapters[tool_type]

    def create(self, tool_type: str, config: dict[str, Any]) -> ToolAdapter:
        """
        Create an instance of a tool adapter

        Args:
            tool_type: Type of tool to create
            config: Configuration for the adapter

        Returns:
            ToolAdapter instance

        Raises:
            ValueError: If tool_type not registered
        """
        if tool_type not in self._adapters:
            raise ValueError(
                f"Tool type '{tool_type}' not registered. "
                f"Available types: {list(self._adapters.keys())}"
            )

        adapter_class = self._adapters[tool_type]
        return adapter_class(config)

    def list_adapters(self) -> list[str]:
        """List all registered adapter types"""
        return list(self._adapters.keys())

    def is_registered(self, tool_type: str) -> bool:
        """Check if a tool type is registered"""
        return tool_type in self._adapters


# Global registry instance
_global_registry = ToolRegistry()


def register_adapter(tool_type: str):
    """
    Decorator for registering tool adapters

    Usage:
        @register_adapter("vector_search")
        class VectorSearchAdapter(ToolAdapter):
            ...
    """

    def decorator(adapter_class: type[ToolAdapter]):
        _global_registry.register(tool_type, adapter_class)
        return adapter_class

    return decorator


def get_registry() -> ToolRegistry:
    """Get global tool registry"""
    return _global_registry
