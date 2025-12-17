"""Web Search Tool Adapter"""

import logging
from collections import deque
from time import sleep, time
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from siare.adapters.base import ToolAdapter, register_adapter


logger = logging.getLogger(__name__)


@register_adapter("web_search")
class WebSearchAdapter(ToolAdapter):
    """
    Web search adapter

    Supports multiple search providers:
    - Google Custom Search
    - Bing Search
    - DuckDuckGo
    - Serper (Google Search API)
    - Brave Search
    """

    # Rate limiting window (seconds)
    RATE_LIMIT_WINDOW_SECONDS = 60

    def __init__(self, config: dict[str, Any]):
        """
        Initialize web search adapter

        Config keys:
            - provider: "google", "bing", "duckduckgo", "serper", "brave"
            - api_key: API key for the provider
            - max_results: Maximum results to return (default: 10)
            - timeout: Request timeout in seconds (default: 10)
            - rate_limit: Maximum requests per minute (default: 10)
            - provider_config: Provider-specific configuration
        """
        super().__init__(config)

        self.provider = config.get("provider", "duckduckgo")
        self.api_key = config.get("api_key")
        self.max_results = config.get("max_results", 10)
        self.timeout = config.get("timeout", 10)
        self.rate_limit = config.get("rate_limit", 10)
        self.provider_config = config.get("provider_config", {})

        self.session: Optional[requests.Session] = None
        self.request_times: deque[float] = deque(maxlen=self.rate_limit)

    def initialize(self) -> None:
        """Initialize web search provider"""

        # Create requests session
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "SIARE/1.0 (Autonomous Agent Research)"})

        # Validate provider-specific requirements
        if self.provider in ["google", "bing", "serper", "brave"] and not self.api_key:
            raise ValueError(f"{self.provider} requires an api_key")

        self.is_initialized = True

    def execute(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """
        Execute web search

        Inputs:
            - query: Search query string
            - max_results: Max results (overrides default)
            - filter_date: Filter by date range ("day", "week", "month", "year")
            - safe_search: Enable safe search (default: True)

        Returns:
            - results: List of {title, url, snippet, date}
            - query: Search query
            - result_count: Number of results
            - provider: Search provider used
        """
        if not self.is_initialized:
            self.initialize()

        # Enforce rate limiting
        self._check_rate_limit()

        query = inputs.get("query", "").strip()
        max_results = inputs.get("max_results", self.max_results)
        filter_date = inputs.get("filter_date")
        safe_search = inputs.get("safe_search", True)

        if not query:
            return {"error": "No query provided", "results": [], "result_count": 0}

        # Search based on provider
        try:
            if self.provider == "google":
                results = self._search_google(query, max_results, filter_date, safe_search)

            elif self.provider == "bing":
                results = self._search_bing(query, max_results, filter_date, safe_search)

            elif self.provider == "duckduckgo":
                results = self._search_duckduckgo(query, max_results)

            elif self.provider == "serper":
                results = self._search_serper(query, max_results, filter_date)

            elif self.provider == "brave":
                results = self._search_brave(query, max_results, safe_search)

            else:
                return {
                    "error": f"Unsupported provider: {self.provider}",
                    "results": [],
                    "result_count": 0,
                }

            return {
                "results": results,
                "query": query,
                "result_count": len(results),
                "provider": self.provider,
                "status": "success",
            }

        except requests.RequestException as e:
            # Network-related errors (timeout, connection, HTTP errors)
            import logging

            logging.exception(f"Web search network error for provider {self.provider}: {e!s}")
            return {
                "error": f"Network error: {e!s}",
                "results": [],
                "result_count": 0,
                "status": "error",
            }

        except ValueError as e:
            # Configuration or validation errors
            import logging

            logging.exception(f"Web search configuration error for provider {self.provider}: {e!s}")
            return {
                "error": f"Configuration error: {e!s}",
                "results": [],
                "result_count": 0,
                "status": "error",
            }

        except Exception as e:
            # Catch-all for unexpected errors
            import logging

            logging.exception(f"Unexpected web search error for provider {self.provider}: {e!s}")
            return {
                "error": f"Unexpected error: {e!s}",
                "results": [],
                "result_count": 0,
                "status": "error",
            }

    def validate_inputs(self, inputs: dict[str, Any]) -> list[str]:
        """Validate search inputs"""
        errors: list[str] = []

        if "query" not in inputs or not inputs["query"]:
            errors.append("Missing required field: query")

        max_results = inputs.get("max_results", self.max_results)
        if not isinstance(max_results, int) or max_results <= 0:
            errors.append("max_results must be a positive integer")

        return errors

    def _check_rate_limit(self) -> None:
        """Enforce rate limiting"""
        now: float = time()

        # Remove requests older than rate limit window
        while self.request_times and (now - self.request_times[0]) > self.RATE_LIMIT_WINDOW_SECONDS:
            self.request_times.popleft()

        # If at limit, wait
        if len(self.request_times) >= self.rate_limit:
            sleep_time: float = self.RATE_LIMIT_WINDOW_SECONDS - (now - self.request_times[0])
            if sleep_time > 0:
                sleep(sleep_time)

        self.request_times.append(now)

    def _validate_url(self, url: str) -> bool:
        """Validate URL format"""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception as e:  # noqa: BLE001 - Catch-all with logging for URL validation
            logger.debug(f"URL validation failed: {e}")
            return False

    def _search_google(
        self, query: str, max_results: int, filter_date: Optional[str], safe_search: bool
    ) -> list[dict[str, Any]]:
        """Search using Google Custom Search API"""

        # Requires Google Custom Search JSON API
        # https://developers.google.com/custom-search/v1/overview

        cx = self.provider_config.get("search_engine_id")
        if not cx:
            raise ValueError("Google search requires search_engine_id in provider_config")

        url = "https://www.googleapis.com/customsearch/v1"

        params = {
            "key": self.api_key,
            "cx": cx,
            "q": query,
            "num": min(max_results, 10),  # Google limits to 10 per request
        }

        if safe_search:
            params["safe"] = "active"

        if filter_date:
            date_restrict_map = {"day": "d1", "week": "w1", "month": "m1", "year": "y1"}
            if filter_date in date_restrict_map:
                params["dateRestrict"] = date_restrict_map[filter_date]

        assert self.session is not None, "Session not initialized"
        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        items = data.get("items", [])

        results: list[dict[str, Any]] = []
        for item in items:
            url = item.get("link", "")
            results.append(
                {
                    "title": item.get("title"),
                    "url": url if self._validate_url(url) else None,
                    "snippet": item.get("snippet"),
                    "date": None,  # Google CSE doesn't always provide dates
                }
            )

        return results

    def _search_bing(
        self, query: str, max_results: int, filter_date: Optional[str], safe_search: bool
    ) -> list[dict[str, Any]]:
        """Search using Bing Search API"""

        url = "https://api.bing.microsoft.com/v7.0/search"

        headers = {"Ocp-Apim-Subscription-Key": self.api_key}

        params = {
            "q": query,
            "count": min(max_results, 50),
            "safeSearch": "Strict" if safe_search else "Off",
        }

        if filter_date:
            # Bing doesn't support simple date filters via API
            # Could add query modification like "query date:>2023-01-01"
            pass

        assert self.session is not None, "Session not initialized"
        response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        web_pages = data.get("webPages", {}).get("value", [])

        results: list[dict[str, Any]] = []
        for page in web_pages:
            url = page.get("url", "")
            results.append(
                {
                    "title": page.get("name"),
                    "url": url if self._validate_url(url) else None,
                    "snippet": page.get("snippet"),
                    "date": page.get("dateLastCrawled"),
                }
            )

        return results

    def _search_duckduckgo(self, query: str, max_results: int) -> list[dict[str, Any]]:
        """Search using DuckDuckGo (no API key required)"""

        try:
            from duckduckgo_search import DDGS  # type: ignore[import-untyped]
            from duckduckgo_search.exceptions import DuckDuckGoSearchException  # type: ignore[import-untyped]

            # Retry with exponential backoff for rate limits
            max_retries = 3
            base_delay = 5.0  # seconds

            for attempt in range(max_retries):
                try:
                    # Add delay between requests to avoid rate limiting
                    if attempt > 0:
                        delay = base_delay * (2 ** attempt)
                        logger.info(f"DuckDuckGo rate limit, waiting {delay}s (attempt {attempt + 1}/{max_retries})")
                        sleep(delay)

                    with DDGS() as ddgs:
                        results_raw = list(ddgs.text(query, max_results=max_results))  # type: ignore[attr-defined]

                    results: list[dict[str, Any]] = []
                    for item in results_raw:
                        url = item.get("href", "")
                        results.append(
                            {
                                "title": item.get("title"),
                                "url": url if self._validate_url(url) else None,
                                "snippet": item.get("body"),
                                "date": None,
                            }
                        )

                    return results

                except DuckDuckGoSearchException as e:
                    if "Ratelimit" in str(e) and attempt < max_retries - 1:
                        continue  # Retry
                    raise  # Re-raise on final attempt or non-rate-limit errors

            return []  # Should not reach here

        except ImportError:
            raise ImportError("DuckDuckGo search requires: pip install duckduckgo-search")

    def _search_serper(
        self, query: str, max_results: int, filter_date: Optional[str]
    ) -> list[dict[str, Any]]:
        """Search using Serper.dev (Google Search API)"""

        url = "https://google.serper.dev/search"

        headers = {"X-API-KEY": self.api_key, "Content-Type": "application/json"}

        payload = {"q": query, "num": max_results}

        if filter_date:
            # Serper supports tbs parameter for date filtering
            date_map = {"day": "qdr:d", "week": "qdr:w", "month": "qdr:m", "year": "qdr:y"}
            if filter_date in date_map:
                payload["tbs"] = date_map[filter_date]

        assert self.session is not None, "Session not initialized"
        response = self.session.post(url, json=payload, headers=headers, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        organic = data.get("organic", [])

        results: list[dict[str, Any]] = []
        for item in organic:
            url = item.get("link", "")
            results.append(
                {
                    "title": item.get("title"),
                    "url": url if self._validate_url(url) else None,
                    "snippet": item.get("snippet"),
                    "date": item.get("date"),
                }
            )

        return results

    def _search_brave(
        self, query: str, max_results: int, safe_search: bool
    ) -> list[dict[str, Any]]:
        """Search using Brave Search API"""

        url = "https://api.search.brave.com/res/v1/web/search"

        headers = {"X-Subscription-Token": self.api_key}

        params = {
            "q": query,
            "count": max_results,
            "safesearch": "strict" if safe_search else "off",
        }

        assert self.session is not None, "Session not initialized"
        response = self.session.get(url, headers=headers, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        web_results = data.get("web", {}).get("results", [])

        results: list[dict[str, Any]] = []
        for item in web_results:
            url = item.get("url", "")
            results.append(
                {
                    "title": item.get("title"),
                    "url": url if self._validate_url(url) else None,
                    "snippet": item.get("description"),
                    "date": item.get("age"),
                }
            )

        return results

    def cleanup(self) -> None:
        """Close requests session"""
        if self.session:
            self.session.close()

    def get_schema(self) -> dict[str, Any]:
        """Get tool schema"""
        return {
            "inputs": {
                "query": {"type": "string", "required": True, "description": "Web search query"},
                "max_results": {
                    "type": "integer",
                    "default": 10,
                    "description": "Maximum number of results",
                },
                "filter_date": {
                    "type": "string",
                    "enum": ["day", "week", "month", "year"],
                    "description": "Filter results by date range",
                },
                "safe_search": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable safe search",
                },
            },
            "outputs": {
                "results": {
                    "type": "array",
                    "description": "Search results with title, url, snippet, date",
                },
                "result_count": {"type": "integer"},
                "query": {"type": "string"},
                "provider": {"type": "string"},
            },
        }
