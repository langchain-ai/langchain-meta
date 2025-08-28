# Copyright (c) Meta Platforms, Inc. and affiliates
"""Async chat functionality for Meta Llama models."""

from collections.abc import AsyncIterator, Collection
from typing import Any, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_openai.chat_models.base import BaseChatOpenAI


class AsyncChatMetaLlama(BaseChatOpenAI):
    """Async chat model for Meta Llama."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the async chat model."""
        super().__init__(**kwargs)

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat response asynchronously."""
        # This is where the problematic code would be according to the problem statement

        # Initialize generation_info with explicit typing to fix mypy errors
        # Fix: explicit type annotation allows mixed value types
        generation_info: dict[str, Any] = {}

        # Set some initial string values
        generation_info["finish_reason"] = "stop"
        generation_info["model"] = "llama-test"

        # ... imagine more code here around line 275 where generation_info
        # is first initialized

        # Simulate processing that leads to the problematic lines
        response_data = await self._simulate_api_call(messages, **kwargs)

        # Process response and build generation info
        generation_info = self._process_response(response_data, generation_info)

        # Create chat result
        generations = [
            ChatGeneration(
                message=response_data["message"], generation_info=generation_info
            )
        ]
        return ChatResult(generations=generations)

    def _process_response(
        self,
        response_data: dict[str, Any],
        generation_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Process API response and update generation info."""
        # Simulate the problematic lines mentioned in the problem statement

        # Line ~415: Incompatible types in assignment
        # (expression has type "dict[str, int]", target has type "str")
        generation_info["usage_metadata"] = {
            "input_tokens": 10,
            "output_tokens": 5,
        }

        # Line ~440: Incompatible types in assignment
        # (expression has type "dict[str, int]", target has type "str")
        generation_info["token_usage"] = {
            "completion_tokens": 5,
            "prompt_tokens": 10,
            "total_tokens": 15,
        }

        # Line ~456: Incompatible types in assignment
        # (expression has type "dict[str, object]", target has type "str")
        generation_info["response_metadata"] = {
            "model": "llama",
            "usage": {"tokens": 15},
        }

        # Line ~457: Incompatible types in assignment
        # (expression has type "dict[str, object]", target has type "str")
        generation_info["llm_output"] = {
            "token_usage": {"total": 15},
            "model_name": "llama",
        }

        # Line ~461: Incompatible types in assignment
        # (expression has type "float", target has type "str")
        generation_info["duration"] = 1.5

        # Some additional processing that could lead to line ~479
        tool_calls = response_data.get("tool_calls", [])
        if tool_calls:
            # Line ~479: Incompatible types in assignment
            # (expression has type "Collection[str] | Any | None",
            #  target has type "str")
            generation_info["tool_calls"] = self._process_tool_calls(tool_calls)

        return generation_info

    def _process_tool_calls(
        self, tool_calls: list[dict[str, Any]]
    ) -> Collection[str] | Any | None:
        """Process tool calls and return collection."""
        if not tool_calls:
            return None
        return [call.get("name", "") for call in tool_calls]

    async def _simulate_api_call(
        self, messages: list[BaseMessage], **kwargs: Any
    ) -> dict[str, Any]:
        """Simulate an API call for testing purposes."""
        from langchain_core.messages import AIMessage

        return {
            "message": AIMessage(content="Test response"),
            "tool_calls": [],
            "usage": {
                "completion_tokens": 5,
                "prompt_tokens": 10,
                "total_tokens": 15,
            },
        }

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream chat response asynchronously."""
        # Similar patterns for streaming with potential type issues
        llm_output_data: dict[str, Any] = {}  # Fix: explicit type annotation
        llm_output_data["model"] = "llama"

        # Simulate streaming chunks
        for i in range(3):
            # More type annotation fixes
            callback_options: dict[str, Any] = {}  # Fix: explicit type annotation
            callback_options["chunk_index"] = i
            callback_options["metadata"] = {"stream": True, "chunk": i}

            chunk = ChatGenerationChunk(
                message=self._create_chunk_message(f"chunk {i}"),
                generation_info=callback_options,
            )
            yield chunk

    def _create_chunk_message(self, content: str) -> BaseMessageChunk:
        """Create a chunk message."""
        from langchain_core.messages import AIMessageChunk

        return AIMessageChunk(content=content)
