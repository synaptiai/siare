"""LLM Response Caching"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from siare.services.llm_provider import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class LLMCache:
    """
    Cache for LLM responses

    Reduces API costs and latency by caching identical requests
    """

    def __init__(
        self,
        cache_dir: str = "./cache/llm",
        ttl_seconds: int = 86400,  # 24 hours
        max_cache_size_mb: int = 1000,
    ):
        """
        Initialize LLM cache

        Args:
            cache_dir: Directory for cache storage
            ttl_seconds: Time-to-live for cache entries
            max_cache_size_mb: Maximum cache size in MB
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.ttl_seconds = ttl_seconds
        self.max_cache_size_mb = max_cache_size_mb

        # In-memory cache for fast access
        self._memory_cache: dict[str, dict[str, Any]] = {}

        # Statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "saves": 0,
            "evictions": 0,
        }

    def _compute_cache_key(
        self, messages: list[LLMMessage], model: str, temperature: float, **kwargs: Any
    ) -> str:
        """Compute cache key from request parameters"""

        # Create deterministic representation
        request_data: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "model": model,
            "temperature": temperature,
            "kwargs": dict(sorted(kwargs.items())),
        }

        # Hash to get cache key
        request_json = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_json.encode()).hexdigest()

    def get(
        self, messages: list[LLMMessage], model: str, temperature: float, **kwargs: Any
    ) -> LLMResponse | None:
        """Get cached response if available"""

        cache_key = self._compute_cache_key(messages, model, temperature, **kwargs)

        # Check memory cache first
        if cache_key in self._memory_cache:
            entry = self._memory_cache[cache_key]

            # Check TTL
            if self._is_expired(entry):
                del self._memory_cache[cache_key]
                self.stats["evictions"] += 1
                return None

            self.stats["hits"] += 1
            return self._deserialize_response(entry["response"])

        # Check disk cache
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file) as f:
                    entry = json.load(f)

                # Check TTL
                if self._is_expired(entry):
                    cache_file.unlink()
                    self.stats["evictions"] += 1
                    return None

                # Load into memory cache
                self._memory_cache[cache_key] = entry

                self.stats["hits"] += 1
                return self._deserialize_response(entry["response"])

            except Exception:
                return None

        self.stats["misses"] += 1
        return None

    def put(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float,
        response: LLMResponse,
        **kwargs: Any,
    ) -> None:
        """Cache a response"""

        cache_key = self._compute_cache_key(messages, model, temperature, **kwargs)

        entry: dict[str, Any] = {
            "cache_key": cache_key,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "response": self._serialize_response(response),
        }

        # Store in memory
        self._memory_cache[cache_key] = entry

        # Store on disk
        cache_file = self.cache_dir / f"{cache_key}.json"

        with open(cache_file, "w") as f:
            json.dump(entry, f, indent=2)

        self.stats["saves"] += 1

        # Check cache size
        self._evict_if_needed()

    def _is_expired(self, entry: dict[str, Any]) -> bool:
        """Check if cache entry is expired"""

        timestamp = datetime.fromisoformat(entry["timestamp"])
        age = datetime.now(timezone.utc) - timestamp

        return age.total_seconds() > self.ttl_seconds

    def _serialize_response(self, response: LLMResponse) -> dict[str, Any]:
        """Serialize LLMResponse to dict"""

        return {
            "content": response.content,
            "model": response.model,
            "usage": response.usage,
            "finish_reason": response.finish_reason,
        }

    def _deserialize_response(self, data: dict[str, Any]) -> LLMResponse:
        """Deserialize dict to LLMResponse"""

        return LLMResponse(
            content=data["content"],
            model=data["model"],
            usage=data["usage"],
            finish_reason=data["finish_reason"],
            raw_response=None,
        )

    def _evict_if_needed(self) -> None:
        """Evict old entries if cache is too large"""

        # Get cache size
        cache_size_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))

        cache_size_mb = cache_size_bytes / (1024 * 1024)

        if cache_size_mb > self.max_cache_size_mb:
            # Get all cache files sorted by modification time
            cache_files = sorted(self.cache_dir.glob("*.json"), key=lambda f: f.stat().st_mtime)

            # Remove oldest 20%
            num_to_remove = max(1, len(cache_files) // 5)

            for cache_file in cache_files[:num_to_remove]:
                cache_key = cache_file.stem

                # Remove from memory cache
                if cache_key in self._memory_cache:
                    del self._memory_cache[cache_key]

                # Remove file
                cache_file.unlink()

                self.stats["evictions"] += 1

    def clear(self) -> None:
        """Clear all cache"""

        # Clear memory
        self._memory_cache.clear()

        # Clear disk
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()

        logger.info("Cache cleared")

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics"""

        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        cache_size_bytes = sum(f.stat().st_size for f in self.cache_dir.glob("*.json"))

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "cache_size_mb": cache_size_bytes / (1024 * 1024),
            "cache_entries": len(list(self.cache_dir.glob("*.json"))),
        }


class CachedLLMProvider(LLMProvider):
    """
    LLM provider wrapper with caching

    Wraps any LLM provider to add caching capability
    """

    def __init__(
        self,
        provider: LLMProvider,
        cache: LLMCache | None = None,
        enable_cache: bool = True,
    ):
        """
        Initialize cached provider

        Args:
            provider: Underlying LLM provider
            cache: Optional cache instance (creates new if None)
            enable_cache: Whether to enable caching
        """
        self.provider = provider
        self.cache = cache or LLMCache()
        self.enable_cache = enable_cache

    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion with caching"""

        # Try cache first
        if self.enable_cache:
            cached_response = self.cache.get(messages, model, temperature, **kwargs)

            if cached_response:
                return cached_response

        # Call underlying provider
        response = self.provider.complete(messages, model, temperature, max_tokens, **kwargs)

        # Cache response
        if self.enable_cache:
            self.cache.put(messages, model, temperature, response, **kwargs)

        return response

    def get_model_name(self, model_ref: str) -> str:
        """Delegate to underlying provider"""
        return self.provider.get_model_name(model_ref)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics"""
        return self.cache.get_stats()

    def clear_cache(self) -> None:
        """Clear cache"""
        self.cache.clear()
