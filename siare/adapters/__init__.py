"""SIARE Adapters - Tool adapters for external services."""

from siare.adapters.base import ToolAdapter, ToolRegistry, get_registry
from siare.adapters.vector_search import VectorSearchAdapter
from siare.adapters.web_search import WebSearchAdapter
from siare.adapters.wikipedia_search import WikipediaSearchAdapter

__all__ = [
    "ToolAdapter",
    "ToolRegistry",
    "VectorSearchAdapter",
    "WebSearchAdapter",
    "WikipediaSearchAdapter",
    "get_registry",
]
