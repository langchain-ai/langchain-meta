# Copyright (c) Meta Platforms, Inc. and affiliates
"""Ported chat models with v2 API compatibility."""

from collections.abc import Iterator
from typing import Any, Callable, Optional, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.tools import BaseTool
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self


class SyncChatMetaLlamaMixin:
    """Mixin for synchronous chat functionality."""

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[
            list[Union[dict[Any, Any], type[BaseModel], Callable[..., Any], BaseTool]]
        ] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response."""
        # Implementation would go here
        return iter([])


class ChatMetaLlama(BaseChatOpenAI, SyncChatMetaLlamaMixin):
    """Chat model for Meta Llama with v2 API ported functionality."""

    # Fix for line 173: Change model_name to str instead of Optional[str]
    # to satisfy base class
    model_name: str = Field(default="llama-default", alias="model")
    """Model name to use."""

    api_key: Optional[str] = Field(default=None)
    """API key for authentication."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the chat model."""
        super().__init__(**kwargs)

    @model_validator(mode="after")
    # Fix for line 368: Add explicit return type annotation
    def validate_model_name(self) -> Self:
        """Validate model name configuration."""
        if not self.model_name:
            self.model_name = "llama-default"
        return self

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get LangSmith parameters."""
        return {
            "ls_provider": "meta",
            "ls_model_name": self.model_name,
            "ls_model_type": "chat",
        }

    # Fix for line 560: Update _get_invocation_params signature to match BaseChatModel
    def _get_invocation_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> dict[Any, Any]:
        """Get parameters for model invocation."""
        params: dict[Any, Any] = {}
        if self.model_name:
            params["model"] = self.model_name
        # ignore stop parameter if not used, but preserve signature compatibility
        return params

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Generate chat response."""
        # Implementation would go here
        pass

    # Fix for line 714: Update _stream signature to include tools parameter
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[
            list[Union[dict[Any, Any], type[BaseModel], Callable[..., Any], BaseTool]]
        ] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat response."""
        # Pass tools parameter to maintain signature consistency
        return SyncChatMetaLlamaMixin._stream(
            self, messages, stop=stop, run_manager=run_manager, tools=tools, **kwargs
        )

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "meta-llama-v2"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Get identifying parameters."""
        return {"model_name": self.model_name}


class EnhancedChatMetaLlama(ChatMetaLlama):
    """Enhanced version with additional functionality."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the enhanced chat model."""
        super().__init__(**kwargs)

    def _create_enhanced_params(self) -> dict[str, Any]:
        """Create enhanced parameters."""
        base_params = self._get_invocation_params()
        enhanced_params = {**base_params, "enhanced": True, "version": "v2"}
        return enhanced_params
