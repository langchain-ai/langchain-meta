"""Unified ChatLlama class.

Supports both OpenAI-based and native Llama API backends.
"""

from typing import Any, Union

from .chat_models import ChatLlama as ChatLlamaOpenAI
from .chat_models_v2_ported import ChatMetaLlama as ChatLlamaNative


class ChatLlama:
    """Unified ChatLlama class.

    Supports both OpenAI-based and native Llama API backends.

    This class provides a compatibility layer that allows users to choose between:
    - OpenAI-compatible backend (default, existing behavior)
    - Native Llama API backend (enhanced features, better tool calling)

    Args:
        use_native_client: If True, uses the native Llama API client.
                          If False (default), uses OpenAI-compatible client.
        **kwargs: Arguments passed to the underlying implementation.

    Examples:
        # Use existing OpenAI-based implementation (default)
        llm = ChatLlama(model="Llama-3.3-8B-Instruct")

        # Use enhanced native implementation
        llm = ChatLlama(model="Llama-3.3-8B-Instruct", use_native_client=True)

        # Native implementation with enhanced features
        llm_native = ChatLlama(
            model_name="Llama-3.3-8B-Instruct",
            use_native_client=True,
            temperature=0.1
        )
    """
    def __new__(
        cls, use_native_client: bool = False, **kwargs: Any
    ) -> Union[ChatLlamaOpenAI, ChatLlamaNative]:
        """Factory method that returns the appropriate implementation.

        Args:
            use_native_client: Whether to use the native Llama API client
            **kwargs: Arguments for the underlying chat model

        Returns:
            Either ChatLlamaOpenAI or ChatLlamaNative instance
        """
        if use_native_client:
            # Remove the use_native_client parameter before passing to ChatLlamaNative
            kwargs.pop("use_native_client", None)

            # Handle parameter mapping for native client
            # OpenAI-style 'model' -> native 'model_name'
            if "model" in kwargs and "model_name" not in kwargs:
                kwargs["model_name"] = kwargs.pop("model")

            return ChatLlamaNative(**kwargs)
        else:
            # Remove the use_native_client parameter before passing to ChatLlamaOpenAI
            kwargs.pop("use_native_client", None)

            # Ensure 'model' parameter is present for OpenAI-based implementation
            if "model_name" in kwargs and "model" not in kwargs:
                kwargs["model"] = kwargs.pop("model_name")

            if "model" not in kwargs:
                raise ValueError("OpenAI-based ChatLlama requires 'model' parameter")

            return ChatLlamaOpenAI(**kwargs)


# For backward compatibility, also export the specific implementations
__all__ = ["ChatLlama", "ChatLlamaOpenAI", "ChatLlamaNative"]
