"""LLM Provider - Unified interface for OpenAI and Anthropic APIs"""

import asyncio
import logging
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal, cast

from siare.core.hooks import HookContext, HookRegistry, fire_llm_hook

logger = logging.getLogger(__name__)

# Optional dependency handling - initialize as None to avoid "possibly unbound" errors
openai: Any = None
anthropic: Any = None
openai_available = False
anthropic_available = False

try:
    import openai as _openai
    openai = _openai
    openai_available = True
except ImportError:
    pass

try:
    import anthropic as _anthropic
    anthropic = _anthropic
    anthropic_available = True
except ImportError:
    pass


@dataclass
class LLMMessage:
    """Unified message format"""

    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class LLMResponse:
    """Unified response format"""

    content: str
    model: str
    usage: dict[str, int]  # {prompt_tokens, completion_tokens, total_tokens}
    finish_reason: str
    raw_response: Any | None = None
    cost: float = 0.0  # Estimated cost in USD


# Model pricing per 1M tokens (input, output) - updated November 2024
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI models
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "gpt-4": (30.00, 60.00),
    "gpt-3.5-turbo": (0.50, 1.50),
    # Anthropic models
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    "claude-3-sonnet-20240229": (3.00, 15.00),
    "claude-3-haiku-20240307": (0.25, 1.25),
    # Ollama models (local, zero cost)
    "llama3.2": (0.0, 0.0),
    "llama3.1": (0.0, 0.0),
    "llama3": (0.0, 0.0),
    "llama2": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "mixtral": (0.0, 0.0),
    "qwen2.5": (0.0, 0.0),
    "qwen2": (0.0, 0.0),
    "gemma2": (0.0, 0.0),
    "phi3": (0.0, 0.0),
    "codellama": (0.0, 0.0),
    "deepseek": (0.0, 0.0),
    "deepseek-coder": (0.0, 0.0),
}


def calculate_cost(model: str, usage: dict[str, int]) -> float:
    """
    Calculate cost based on model and token usage.

    Args:
        model: Model identifier
        usage: Token usage dict with prompt_tokens and completion_tokens

    Returns:
        Estimated cost in USD
    """
    # Normalize model name to find pricing
    model_lower = model.lower()

    # Find matching pricing
    pricing = None
    for model_key, price in MODEL_PRICING.items():
        if model_key in model_lower or model_lower in model_key:
            pricing = price
            break

    if pricing is None:
        # Default to gpt-4o-mini pricing if model not found
        pricing = MODEL_PRICING["gpt-4o-mini"]

    input_price, output_price = pricing
    prompt_tokens = usage.get("prompt_tokens", 0)
    completion_tokens = usage.get("completion_tokens", 0)

    # Calculate cost (prices are per 1M tokens)
    cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1_000_000
    return cost


def _fire_llm_hook(hook_name: str, ctx: HookContext, *args: Any, **kwargs: Any) -> Any:
    """Fire an LLM hook from sync context.

    Safely runs async hooks from synchronous code. If no hooks are registered,
    returns immediately with zero overhead. Errors are logged but don't propagate.

    Args:
        hook_name: Name of the hook method (e.g., "on_llm_call_complete").
        ctx: Hook context with correlation ID and metadata.
        *args: Arguments for the hook.
        **kwargs: Keyword arguments for the hook.

    Returns:
        Hook result, or None if no hooks registered or hook failed.
    """
    # Fast path: no hooks registered = no overhead
    if HookRegistry.get_llm_hooks() is None:
        return None

    try:
        # Try to get existing event loop (may be running in async context)
        try:
            loop = asyncio.get_running_loop()
            # We're in an async context - create task but don't await
            loop.create_task(fire_llm_hook(hook_name, ctx, *args, **kwargs))
            return None
        except RuntimeError:
            # No running loop - create new one for sync context
            return asyncio.run(fire_llm_hook(hook_name, ctx, *args, **kwargs))
    except Exception as e:
        logger.warning(f"Failed to fire hook {hook_name}: {e}")
        return None


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate completion

        Args:
            messages: List of messages
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            LLMResponse
        """

    @abstractmethod
    def get_model_name(self, model_ref: str) -> str:
        """Map model reference to actual model name"""


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""

    def __init__(self, api_key: str | None = None):
        """
        Initialize OpenAI provider

        Args:
            api_key: API key (uses OPENAI_API_KEY env var if not provided)
        """
        if not openai_available:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")

        # Set reasonable timeout to avoid hanging connections
        # Default is no timeout which can cause indefinite hangs
        self.client = openai.OpenAI(
            api_key=self.api_key,
            timeout=120.0,  # 2 minute timeout
        )  # type: ignore[attr-defined]

        # Model mapping - map internal model names to actual OpenAI models
        # GPT-5 family now available: gpt-5, gpt-5-mini, gpt-5-nano
        self.model_map = {
            "gpt-5": "gpt-5",
            "gpt-5-mini": "gpt-5-mini",
            "gpt-5-nano": "gpt-5-nano",
            "gpt-4o": "gpt-4o",
            "gpt-4o-mini": "gpt-4o-mini",
            "gpt-4-turbo": "gpt-4-turbo-preview",
            "gpt-4": "gpt-4",
            "gpt-3.5-turbo": "gpt-3.5-turbo",
        }

    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using OpenAI API"""

        # Create hook context for this LLM call
        hook_ctx = HookContext(
            correlation_id=str(uuid.uuid4()),
            metadata={"provider": "openai"},
        )

        # Get actual model name
        model_name = self.get_model_name(model)

        # Fire LLM request hook
        openai_msgs = [{"role": msg.role, "content": msg.content} for msg in messages]
        _fire_llm_hook("on_llm_request", hook_ctx, provider="openai", model=model_name, messages=openai_msgs)

        # Convert to OpenAI format
        openai_messages: list[dict[str, Any]] = [
            {"role": msg.role, "content": msg.content} for msg in messages
        ]

        # Build API call parameters - only include max_tokens if specified
        api_params: dict[str, Any] = {
            "model": model_name,
            "messages": openai_messages,
            **kwargs,
        }

        # GPT-5 models don't support custom temperature - only default (1) is allowed
        if not model_name.startswith("gpt-5"):
            api_params["temperature"] = temperature

        if max_tokens is not None:
            api_params["max_tokens"] = max_tokens

        # Call API with comprehensive exception handling
        try:
            response: Any = self.client.chat.completions.create(**api_params)  # type: ignore[misc]

            # Extract response
            choice: Any = response.choices[0]  # type: ignore[misc]
            content: str = str(choice.message.content or "")  # type: ignore[misc]

            usage = {
                "prompt_tokens": int(response.usage.prompt_tokens),  # type: ignore[misc]
                "completion_tokens": int(response.usage.completion_tokens),  # type: ignore[misc]
                "total_tokens": int(response.usage.total_tokens),  # type: ignore[misc]
            }
            cost = calculate_cost(str(response.model), usage)  # type: ignore[misc]
            llm_response = LLMResponse(
                content=content,
                model=str(response.model),  # type: ignore[misc]
                usage=usage,
                finish_reason=str(choice.finish_reason),  # type: ignore[misc]
                raw_response=response,
                cost=cost,
            )

            # Fire LLM response hook
            _fire_llm_hook(
                "on_llm_response", hook_ctx,
                provider="openai",
                model=llm_response.model,
                tokens_in=usage["prompt_tokens"],
                tokens_out=usage["completion_tokens"],
                latency_ms=0.0,  # Not tracked at this level
            )

            return llm_response

        except openai.RateLimitError as e:  # type: ignore[attr-defined]
            raise RuntimeError(f"OpenAI rate limit exceeded: {e!s}") from e

        except openai.APIError as e:  # type: ignore[attr-defined]
            raise RuntimeError(f"OpenAI API error: {e!s}") from e

        except Exception as e:
            raise RuntimeError(f"Unexpected error during LLM completion: {e!s}") from e

    def get_model_name(self, model_ref: str) -> str:
        """Map model reference to OpenAI model name"""
        return self.model_map.get(model_ref, model_ref)

    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass  # Ignore errors during cleanup

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        self.cleanup()


class AnthropicProvider(LLMProvider):
    """Anthropic API provider"""

    def __init__(self, api_key: str | None = None):
        """
        Initialize Anthropic provider

        Args:
            api_key: API key (uses ANTHROPIC_API_KEY env var if not provided)
        """
        if not anthropic_available:
            raise ImportError(
                "Anthropic package not installed. Install with: pip install anthropic"
            )

        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")

        self.client = anthropic.Anthropic(api_key=self.api_key)  # type: ignore[attr-defined]

        # Model mapping
        self.model_map = {
            "claude-3-opus": "claude-3-opus-20240229",
            "claude-3-sonnet": "claude-3-sonnet-20240229",
            "claude-3-haiku": "claude-3-haiku-20240307",
            "claude-3.5-sonnet": "claude-3-5-sonnet-20241022",
            "claude-3.5-haiku": "claude-3-5-haiku-20241022",
        }

    def complete(
        self,
        messages: list[LLMMessage],
        model: str,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate completion using Anthropic API"""

        # Create hook context for this LLM call
        hook_ctx = HookContext(
            correlation_id=str(uuid.uuid4()),
            metadata={"provider": "anthropic"},
        )

        # Get actual model name
        model_name = self.get_model_name(model)

        # Fire LLM request hook
        anthropic_msgs = [{"role": msg.role, "content": msg.content} for msg in messages]
        _fire_llm_hook("on_llm_request", hook_ctx, provider="anthropic", model=model_name, messages=anthropic_msgs)

        # Separate system message
        system_message: str | None = None
        anthropic_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == "system":
                system_message = msg.content
            else:
                anthropic_messages.append({"role": msg.role, "content": msg.content})

        # Default max_tokens for Anthropic
        if max_tokens is None:
            max_tokens = 4096

        # Call API
        response: Any = self.client.messages.create(  # type: ignore[misc]
            model=model_name,
            messages=anthropic_messages,  # type: ignore[arg-type]
            system=system_message if system_message is not None else cast("Any", None),  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )

        # Extract response
        content = ""
        for block in response.content:  # type: ignore[misc]
            block_any: Any = block
            if hasattr(block_any, "text"):
                content += str(block_any.text)  # type: ignore[misc]

        usage = {
            "prompt_tokens": int(response.usage.input_tokens),  # type: ignore[misc]
            "completion_tokens": int(response.usage.output_tokens),  # type: ignore[misc]
            "total_tokens": int(response.usage.input_tokens + response.usage.output_tokens),  # type: ignore[misc]
        }
        cost = calculate_cost(str(response.model), usage)  # type: ignore[misc]
        llm_response = LLMResponse(
            content=content,
            model=str(response.model),  # type: ignore[misc]
            usage=usage,
            finish_reason=str(response.stop_reason),  # type: ignore[misc]
            raw_response=response,
            cost=cost,
        )

        # Fire LLM response hook
        _fire_llm_hook(
            "on_llm_response", hook_ctx,
            provider="anthropic",
            model=llm_response.model,
            tokens_in=usage["prompt_tokens"],
            tokens_out=usage["completion_tokens"],
            latency_ms=0.0,  # Not tracked at this level
        )

        return llm_response

    def get_model_name(self, model_ref: str) -> str:
        """Map model reference to Anthropic model name"""
        return self.model_map.get(model_ref, model_ref)

    def cleanup(self) -> None:
        """Clean up resources"""
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except Exception:
                pass  # Ignore errors during cleanup

    def __del__(self) -> None:
        """Destructor to ensure cleanup"""
        self.cleanup()


class LLMProviderFactory:
    """Factory for creating LLM providers - production providers only"""

    @staticmethod
    def create(
        provider_type: Literal["openai", "anthropic", "ollama", "mock"],
        api_key: str | None = None,
        **kwargs: Any,
    ) -> LLMProvider:
        """
        Create LLM provider

        Args:
            provider_type: Type of provider ("openai", "anthropic", "ollama", or "mock")
            api_key: API key (optional, can use env vars)
            **kwargs: Additional provider-specific arguments
                For ollama: model (str), base_url (str), timeout (float)

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider_type is unknown
        """
        if provider_type == "openai":
            return OpenAIProvider(api_key=api_key)
        if provider_type == "anthropic":
            return AnthropicProvider(api_key=api_key)
        if provider_type == "ollama":
            # Use importlib to avoid circular dependency
            # ollama_provider imports LLMProvider base class from this module
            # Using importlib.import_module breaks the bidirectional reference
            import importlib
            ollama_module = importlib.import_module("siare.providers.ollama_provider")
            ollama_provider_cls = ollama_module.OllamaProvider
            return ollama_provider_cls(
                model=kwargs.get("model", "llama3.2:7b"),
                base_url=kwargs.get("base_url"),
                timeout=kwargs.get("timeout", 120.0),
            )
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Supported types: 'openai', 'anthropic', 'ollama'"
        )

    @staticmethod
    def create_from_config(config: dict[str, Any]) -> LLMProvider:
        """
        Create provider from configuration dict

        Args:
            config: Configuration dict with keys:
                - type: "openai" or "anthropic" (REQUIRED)
                - api_key: Optional API key
                - Additional provider-specific settings

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If type is not specified or unknown
        """
        provider_type = config.get("type")
        if not provider_type:
            raise ValueError(
                "LLM provider 'type' is required in config. "
                "Supported types: 'openai', 'anthropic'"
            )
        api_key = config.get("api_key")

        return LLMProviderFactory.create(
            provider_type=provider_type,
            api_key=api_key,
            **{k: v for k, v in config.items() if k not in ["type", "api_key"]},
        )
