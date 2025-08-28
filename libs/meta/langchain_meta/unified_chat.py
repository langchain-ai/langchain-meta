# Copyright (c) Meta Platforms, Inc. and affiliates
"""Unified chat interface for different Llama implementations."""

from typing import Any, Union


class ChatLlamaOpenAI:
    """OpenAI-compatible ChatLlama implementation."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the OpenAI-compatible implementation."""
        self.kwargs = kwargs

    def invoke(self, messages: Any) -> Any:
        """Invoke chat completion."""
        return f"OpenAI response: {messages}"


class ChatLlamaNative:
    """Native ChatLlama implementation."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the native implementation."""
        self.kwargs = kwargs

    def invoke(self, messages: Any) -> Any:
        """Invoke chat completion."""
        return f"Native response: {messages}"


class ChatLlama:
    """Unified ChatLlama factory class."""

    # Fix for line 41: Option A - Add type ignore for __new__ method
    # (factory class pattern)
    def __new__(  # type: ignore[misc]
        cls, implementation: str = "openai", **kwargs: Any
    ) -> Union[ChatLlamaOpenAI, ChatLlamaNative]:
        """Create appropriate ChatLlama implementation.

        This is a factory class pattern where __new__ returns instances
        of different classes. The type ignore is needed because mypy
        expects __new__ to return cls instance.
        """
        if implementation == "openai":
            return ChatLlamaOpenAI(**kwargs)
        elif implementation == "native":
            return ChatLlamaNative(**kwargs)
        else:
            raise ValueError(f"Unknown implementation: {implementation}")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize ChatLlama (this won't be called due to __new__)."""
        pass


def create_chat_llama(
    implementation: str = "openai", **kwargs: Any
) -> Union[ChatLlamaOpenAI, ChatLlamaNative]:
    """Factory function to create ChatLlama instances."""
    return ChatLlama(implementation=implementation, **kwargs)  # type: ignore
