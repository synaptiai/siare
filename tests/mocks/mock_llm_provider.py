"""Mock LLM Provider - FOR TESTING ONLY

This module provides a mock LLM provider for unit tests.
It should NEVER be used in production code.
"""

from typing import Optional

from siare.services.llm_provider import LLMMessage, LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing - returns configurable responses"""

    # Default JSON response with common fields for testing
    DEFAULT_RESPONSE = '{"score": 0.85, "critique": "Mock critique", "result": "Mock result", "category": "A", "plan": "Mock plan", "answer": "Mock answer"}'

    def __init__(self, mock_response: str | None = None, response_sequence: list[str] | None = None):
        """
        Initialize mock provider.

        Args:
            mock_response: The response to return from complete().
                          Defaults to a JSON object with common test fields.
            response_sequence: Optional list of responses to return in sequence.
                             If provided, each call returns the next response.
                             Falls back to mock_response when exhausted.
        """
        self.mock_response = mock_response if mock_response is not None else self.DEFAULT_RESPONSE
        self.response_sequence = response_sequence or []
        self._sequence_index = 0
        self.call_count = 0
        self.last_messages: list[LLMMessage] = []

    def complete(
        self,
        messages: list[LLMMessage],
        model: str = "mock-model",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> LLMResponse:
        """Return mock response and track calls"""
        self.call_count += 1
        self.last_messages = messages

        # Use sequence if available, otherwise use single response
        if self._sequence_index < len(self.response_sequence):
            content = self.response_sequence[self._sequence_index]
            self._sequence_index += 1
        else:
            content = self.mock_response

        return LLMResponse(
            content=content,
            model=model,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            finish_reason="stop",
            cost=0.0001,  # Mock cost for testing
        )

    def get_model_name(self, model_ref: str) -> str:
        """Return model name unchanged"""
        return model_ref

    def set_response(self, response: str) -> None:
        """Update the mock response"""
        self.mock_response = response

    def reset(self) -> None:
        """Reset call tracking"""
        self.call_count = 0
        self.last_messages = []
