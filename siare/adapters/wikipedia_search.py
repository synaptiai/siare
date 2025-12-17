"""Wikipedia Search Tool Adapter.

Uses Wikipedia's REST API for search and content retrieval.
No API key required, more reliable than DuckDuckGo for knowledge-based queries.
"""

import logging
from typing import Any

import requests

from siare.adapters.base import ToolAdapter, register_adapter

logger = logging.getLogger(__name__)


@register_adapter("wikipedia_search")
class WikipediaSearchAdapter(ToolAdapter):
    """Wikipedia search adapter using the MediaWiki API.

    Features:
    - No API key required
    - Reliable rate limits (200 requests/second for anonymous users)
    - Good for factual/encyclopedic queries
    - Supports search and content extraction
    """

    BASE_URL = "https://en.wikipedia.org/w/api.php"

    def __init__(self, config: dict[str, Any]):
        """Initialize Wikipedia search adapter.

        Config keys:
            - max_results: Maximum search results (default: 5)
            - timeout: Request timeout in seconds (default: 30)
            - extract_chars: Max characters to extract per article (default: 1000)
        """
        super().__init__(config)
        self.max_results = config.get("max_results", 5)
        self.timeout = config.get("timeout", 30)
        self.extract_chars = config.get("extract_chars", 1000)
        self.session: requests.Session | None = None

    def initialize(self) -> None:
        """Initialize HTTP session."""
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "SIARE/1.0 (Autonomous Agent Research; Contact: research@example.com)"
        })
        self.is_initialized = True

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Execute Wikipedia search.

        Inputs:
            - query: Search query string
            - max_results: Max results (overrides default)

        Returns:
            - results: List of {title, url, snippet, pageid}
            - query: Search query
            - result_count: Number of results
        """
        if not self.is_initialized:
            self.initialize()

        query = inputs.get("query", "").strip()
        max_results = inputs.get("max_results", self.max_results)

        if not query:
            return {"error": "No query provided", "results": [], "result_count": 0}

        try:
            # Search Wikipedia
            search_results = self._search(query, max_results)

            if not search_results:
                return {
                    "results": [],
                    "query": query,
                    "result_count": 0,
                    "status": "success",
                }

            # Get extracts for each result
            page_ids = [str(r["pageid"]) for r in search_results]
            extracts = self._get_extracts(page_ids)

            # Combine search results with extracts
            results = []
            for sr in search_results:
                page_id = str(sr["pageid"])
                extract = extracts.get(page_id, {})
                results.append({
                    "title": sr.get("title", ""),
                    "url": f"https://en.wikipedia.org/wiki/{sr.get('title', '').replace(' ', '_')}",
                    "snippet": extract.get("extract", sr.get("snippet", "")),
                    "pageid": page_id,
                })

            return {
                "results": results,
                "query": query,
                "result_count": len(results),
                "status": "success",
            }

        except requests.RequestException as e:
            logger.exception(f"Wikipedia API error: {e}")
            return {
                "error": f"API error: {e}",
                "results": [],
                "result_count": 0,
                "status": "error",
            }

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Search Wikipedia for articles."""
        assert self.session is not None

        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "srlimit": limit,
            "format": "json",
        }

        response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("query", {}).get("search", [])

    def _get_extracts(self, page_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Get text extracts for Wikipedia pages."""
        if not page_ids:
            return {}

        assert self.session is not None

        params = {
            "action": "query",
            "pageids": "|".join(page_ids),
            "prop": "extracts",
            "exintro": True,
            "explaintext": True,
            "exchars": self.extract_chars,
            "format": "json",
        }

        response = self.session.get(self.BASE_URL, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        pages = data.get("query", {}).get("pages", {})

        return {pid: page for pid, page in pages.items()}

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """Validate search inputs."""
        errors: list[str] = []

        if "query" not in inputs or not inputs["query"]:
            errors.append("Missing required field: query")

        return errors

    def cleanup(self) -> None:
        """Close HTTP session."""
        if self.session:
            self.session.close()

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema."""
        return {
            "inputs": {
                "query": {"type": "string", "required": True, "description": "Wikipedia search query"},
                "max_results": {
                    "type": "integer",
                    "default": 5,
                    "description": "Maximum number of results",
                },
            },
            "outputs": {
                "results": {
                    "type": "array",
                    "description": "Search results with title, url, snippet, pageid",
                },
                "result_count": {"type": "integer"},
                "query": {"type": "string"},
            },
        }
