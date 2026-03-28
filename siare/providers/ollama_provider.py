"""Ollama LLM provider for local model inference."""

from __future__ import annotations

import logging
from typing import Any

import requests

from siare.services.llm_provider import LLMMessage, LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class OllamaProvider(LLMProvider):
    """LLM provider for Ollama local models.

    Usage:
        provider = OllamaProvider(model="llama3.2:1b")
        response = provider.complete(
            messages=[LLMMessage(role="user", content="Hello")],
            model="llama3.2:1b",
        )
    """

    def __init__(
        self,
        model: str = "llama3.2:3b",
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.default_model = model
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.timeout = timeout

    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion via Ollama API."""
        ollama_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]

        payload: dict[str, Any] = {
            "model": model or self.default_model,
            "messages": ollama_messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if max_tokens is not None:
            payload["options"]["num_predict"] = max_tokens

        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
            data = resp.json()
        except (requests.RequestException, ValueError) as e:
            raise RuntimeError(
                f"Ollama API call failed for model "
                f"{model or self.default_model}"
            ) from e

        content = data.get("message", {}).get("content", "")
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)

        return LLMResponse(
            content=content,
            model=model or self.default_model,
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            finish_reason=data.get("done_reason", "stop"),
            raw_response=data,
            cost=0.0,  # Local models are free
        )

    def get_model_name(self, model_ref: str) -> str:
        """Map model reference to actual model name."""
        return model_ref
