"""Asynchronous chat implementation for Meta Llama API.

This module provides the AsyncChatMetaLlamaMixin class that implements
asynchronous chat functionality for the native Meta Llama API using
the llama-api-client package.
"""

import json
import logging
import re
import uuid
from collections.abc import AsyncIterator
from datetime import datetime
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
    cast,
)

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,  # Used by _detect_supervisor_request if that were moved,
    ToolCallChunk,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    Generation,
    LLMResult,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from llama_api_client import AsyncLlamaAPIClient
from llama_api_client.types.chat import (
    completion_create_params,
)
from llama_api_client.types.create_chat_completion_response import (
    CreateChatCompletionResponse,
)
from pydantic import BaseModel

# Assuming chat_models.py is in langchain_meta.chat_models
# Adjust the import path if necessary based on your project structure.
from .serialization import (
    _lc_tool_to_llama_tool_param,
    _parse_textual_tool_args,
)  # Changed from ..chat_models

logger = logging.getLogger(__name__)


class AsyncChatMetaLlamaMixin:
    """Mixin class to hold asynchronous methods for ChatMetaLlama."""

    # Type hints for attributes/methods from ChatMetaLlama main class
    # that are used by these async methods via \`self\`.
    _async_client: Optional[AsyncLlamaAPIClient]
    model_name: str

    # These methods are expected to be part of the main ChatMetaLlama class
    def _ensure_client_initialized(self) -> None:
        raise NotImplementedError  # pragma: no cover

    def _prepare_api_params(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[Any]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def _count_tokens(self, messages: list[BaseMessage]) -> int:
        raise NotImplementedError  # pragma: no cover

    def _get_invocation_params(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError  # pragma: no cover

    def _detect_supervisor_request(self, messages: list[BaseMessage]) -> bool:
        raise NotImplementedError  # pragma: no cover

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        tools: Optional[list[Union[dict, type[BaseModel], Callable, BaseTool]]] = None,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any", "required"]]
        ] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously generates a chat response using AsyncLlamaAPIClient."""
        self._ensure_client_initialized()
        if not self._async_client:
            raise ValueError(
                "Async client not initialized. Call `async_init_clients` first."
            )
        async_client_to_use = self._async_client

        prompt_tokens = 0
        completion_tokens = 0
        start_time = datetime.now()

        # Prepare callback options
        callback_options = {}

        # Check for structured output format in run_metadata
        if "run_metadata" in kwargs and isinstance(kwargs["run_metadata"], dict):
            run_metadata = kwargs["run_metadata"]
            if "ls_structured_output_format" in run_metadata:
                callback_options["ls_structured_output_format"] = run_metadata[
                    "ls_structured_output_format"
                ]

        # Also check directly in kwargs (for backward compatibility)
        if "ls_structured_output_format" in kwargs:
            callback_options["ls_structured_output_format"] = kwargs[
                "ls_structured_output_format"
            ]

        # Start the run if we have a manager
        if run_manager and hasattr(run_manager, "on_llm_start"):
            try:
                # Pass options to callback
                on_llm_start_fn = getattr(run_manager, "on_llm_start")
                await on_llm_start_fn(
                    {"name": self.__class__.__name__},
                    messages,
                    invocation_params=self._get_invocation_params(**kwargs),
                    options=callback_options,
                )
            except Exception as e:
                logger.warning(f"Error in on_llm_start callback: {str(e)}")

        if tool_choice is not None and "tool_choice" not in kwargs:
            kwargs["tool_choice"] = tool_choice

        if kwargs.get("stream", False):
            completion_coro = self._astream_with_aggregation_and_retries(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                tools=tools,
                tool_choice=tool_choice,
                **kwargs,
            )
            return await self._aget_stream_results(completion_coro, run_manager)

        logger.debug(f"_agenerate received direct tools: {tools}")
        logger.debug(f"_agenerate received direct tool_choice: {tool_choice}")
        logger.debug(f"_agenerate received kwargs: {kwargs}")

        effective_tools_lc_input = tools
        if effective_tools_lc_input is None and "tools" in kwargs:
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_agenerate (non-streaming): Using 'tools' from **kwargs as "
                "direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[list[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_agenerate (non-streaming): effective_tools_lc_input was not "
                    f"a list ({type(effective_tools_lc_input)}). Wrapping in a list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]

            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_agenerate (non-streaming): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)

        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice

        api_params = self._prepare_api_params(
            messages, tools=prepared_llm_tools, **final_kwargs_for_prepare
        )

        if run_manager:
            self._count_tokens(messages)
            pass

        logger.debug(f"Llama API (async) Request (ainvoke): {api_params}")
        try:
            call_result: CreateChatCompletionResponse = (
                await async_client_to_use.chat.completions.create(**api_params)
            )
            logger.debug(f"Llama API (async) Response (ainvoke): {call_result}")
        except Exception as e:
            if run_manager:
                try:
                    if hasattr(run_manager, "on_llm_error"):
                        await run_manager.on_llm_error(error=e)
                except Exception as callback_err:
                    logger.warning(f"Error in LangSmith error callback: {callback_err}")
            raise e

        result_msg = (
            call_result.completion_message
            if hasattr(call_result, "completion_message")
            else None
        )
        content_str = ""

        # Enhanced content extraction for improved reliability
        if result_msg:
            # Direct attribute access method - attempt 1
            if hasattr(result_msg, "content"):
                content = getattr(result_msg, "content")
                if isinstance(content, dict) and "text" in content:
                    content_str = content["text"]
                elif isinstance(content, str):
                    content_str = content

            # If the above didn't work, try dictionary-based access - attempt 2
            if not content_str and hasattr(result_msg, "to_dict"):
                try:
                    result_dict = result_msg.to_dict()
                    if isinstance(result_dict, dict) and "content" in result_dict:
                        content_dict = result_dict["content"]
                        if isinstance(content_dict, dict) and "text" in content_dict:
                            content_str = content_dict["text"]
                        elif isinstance(content_dict, str):
                            content_str = content_dict
                except (AttributeError, TypeError, KeyError):
                    pass

        if not content_str and hasattr(call_result, "to_dict"):
            try:
                full_result = call_result.to_dict()
                if isinstance(full_result, dict):
                    # Try to extract from completion_message
                    if "completion_message" in full_result:
                        comp_msg = full_result["completion_message"]
                        if isinstance(comp_msg, dict) and "content" in comp_msg:
                            content = comp_msg["content"]
                            if isinstance(content, dict) and "text" in content:
                                content_str = content["text"]
                            elif isinstance(content, str):
                                content_str = content

                    if not content_str and "response_metadata" in full_result:
                        response_meta = full_result["response_metadata"]
                        if (
                            isinstance(response_meta, dict)
                            and "completion_message" in response_meta
                        ):
                            comp_msg = response_meta["completion_message"]
                            if isinstance(comp_msg, dict) and "content" in comp_msg:
                                content = comp_msg["content"]
                                if isinstance(content, dict) and "text" in content:
                                    content_str = content["text"]
                                elif isinstance(content, str):
                                    content_str = content
            except (AttributeError, TypeError, KeyError):
                pass

        generation_info = {}
        tool_calls_data = []
        if result_msg and hasattr(result_msg, "tool_calls") and result_msg.tool_calls:
            processed_tool_calls: list[dict] = []
            for idx, tc in enumerate(result_msg.tool_calls):
                tc_id = (
                    getattr(tc, "id", None) or f"llama_tc_{idx}" or str(uuid.uuid4())
                )
                tc_func = tc.function if hasattr(tc, "function") else None
                tc_name = getattr(tc_func, "name", None) if tc_func else None
                tc_args_str = getattr(tc_func, "arguments", "") if tc_func else ""

                if tc_name and not isinstance(tc_name, str):
                    tc_name = (
                        str(tc_name) if hasattr(tc_name, "__str__") else "unknown_tool"
                    )

                try:
                    parsed_args = json.loads(tc_args_str) if tc_args_str else {}
                    final_args = (
                        {"value": str(parsed_args)}
                        if not isinstance(parsed_args, dict)
                        else parsed_args
                    )
                except json.JSONDecodeError:
                    final_args = {"value": tc_args_str}
                except Exception as e:
                    logger.warning(
                        f"Unexpected error processing tool call arguments: {e}. "
                        "Representing as string."
                    )
                    final_args = {"value": tc_args_str}
                # Defensive: always ensure id, name, args
                if not tc_id:
                    tc_id = str(uuid.uuid4())
                if not tc_name:
                    tc_name = "unknown_tool"
                if not isinstance(final_args, dict):
                    final_args = {"value": str(final_args)}
                processed_tool_calls.append(
                    {
                        "id": tc_id,
                        "type": "function",
                        "name": tc_name,
                        "args": final_args,
                    }
                )
            tool_calls_data = processed_tool_calls

        # Fallback: If no tool_calls from API, try to parse from content_str if
        # tools were provided and content looks like a textual tool call
        if not tool_calls_data and content_str and prepared_llm_tools:
            match = re.fullmatch(
                r"\s*\[\s*([a-zA-Z0-9_]+)\s*(?:\(\s*(.*?)\s*\))?\s*\]\s*", content_str
            )
            if match:
                tool_name_from_content = match.group(1)
                args_str_from_content = match.group(2)
                available_tool_names = [
                    t["function"]["name"]
                    for t in prepared_llm_tools
                    if isinstance(t, dict)
                    and "function" in t
                    and "name" in t["function"]
                ]
                if tool_name_from_content in available_tool_names:
                    logger.info(
                        f"Parsed textual tool call for '{tool_name_from_content}' from content."
                    )
                    tool_call_id = str(uuid.uuid4())
                    parsed_args = {}
                    if args_str_from_content:
                        try:
                            parsed_args = _parse_textual_tool_args(
                                args_str_from_content
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to parse arguments '{args_str_from_content}' for textual tool call '{tool_name_from_content}': {e}. Using raw string as arg."
                            )
                            parsed_args = {"value": args_str_from_content}
                    # Defensive: always ensure id, name, args
                    if not tool_call_id:
                        tool_call_id = str(uuid.uuid4())
                    if not tool_name_from_content:
                        tool_name_from_content = "unknown_tool"
                    if not isinstance(parsed_args, dict):
                        parsed_args = {"value": str(parsed_args)}
                    tool_calls_data.append(
                        {
                            "id": tool_call_id,
                            "name": tool_name_from_content,
                            "args": parsed_args,
                            "type": "function",
                        }
                    )
                    content_str = (
                        ""  # Clear content as it was a tool call representation
                    )
                    generation_info["finish_reason"] = "tool_calls"
                else:
                    logger.warning(
                        f"Textual tool call '{tool_name_from_content}' found in content, but not in available tools: {available_tool_names}"
                    )
            else:
                logger.debug(
                    f"Content '{content_str}' did not match textual tool call pattern."
                )

        message = AIMessage(content=content_str or "", tool_calls=tool_calls_data)

        if result_msg and hasattr(result_msg, "stop_reason") and result_msg.stop_reason:
            generation_info["finish_reason"] = result_msg.stop_reason
        elif hasattr(call_result, "stop_reason") and getattr(
            call_result, "stop_reason", None
        ):
            generation_info["finish_reason"] = getattr(call_result, "stop_reason")

        if (
            hasattr(call_result, "metrics")
            and call_result.metrics
            and isinstance(call_result.metrics, list)
        ):
            usage_meta = {}
            for metric_item in call_result.metrics:
                if hasattr(metric_item, "metric") and hasattr(metric_item, "value"):
                    metric_name = getattr(metric_item, "metric")
                    metric_value = (
                        int(metric_item.value) if metric_item.value is not None else 0
                    )
                    if metric_name == "num_prompt_tokens":
                        usage_meta["input_tokens"] = metric_value
                        prompt_tokens = metric_value
                    elif metric_name == "num_completion_tokens":
                        usage_meta["output_tokens"] = metric_value
                        completion_tokens = metric_value
                    elif metric_name == "num_total_tokens":
                        usage_meta["total_tokens"] = metric_value
            if usage_meta:
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):
                    try:
                        constructed_usage = UsageMetadata(
                            input_tokens=usage_meta.get("input_tokens", 0),
                            output_tokens=usage_meta.get("output_tokens", 0),
                            total_tokens=usage_meta.get("total_tokens", 0),
                        )
                        message.usage_metadata = constructed_usage
                    except Exception as e:
                        logger.warning(f"Could not construct UsageMetadata: {e}")
        elif hasattr(call_result, "usage") and getattr(call_result, "usage", None):
            usage_data = getattr(call_result, "usage")
            input_tokens_val = int(getattr(usage_data, "prompt_tokens", 0))
            output_tokens_val = int(getattr(usage_data, "completion_tokens", 0))
            total_tokens_val = int(getattr(usage_data, "total_tokens", 0))
            usage_meta = {
                "input_tokens": input_tokens_val,
                "output_tokens": output_tokens_val,
                "total_tokens": total_tokens_val,
            }
            prompt_tokens = usage_meta["input_tokens"]
            completion_tokens = usage_meta["output_tokens"]
            usage_meta["total_tokens"]
            if any(usage_meta.values()):
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):
                    try:
                        constructed_usage = UsageMetadata(
                            input_tokens=usage_meta.get("input_tokens", 0),
                            output_tokens=usage_meta.get("output_tokens", 0),
                            total_tokens=usage_meta.get("total_tokens", 0),
                        )
                        message.usage_metadata = constructed_usage
                    except Exception as e:
                        logger.warning(f"Could not construct UsageMetadata: {e}")

        if hasattr(call_result, "x_request_id") and getattr(
            call_result, "x_request_id", None
        ):
            generation_info["x_request_id"] = getattr(call_result, "x_request_id")
        generation_info["response_metadata"] = call_result.to_dict()
        generation_info["llm_output"] = call_result.to_dict()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        generation_info["duration"] = duration

        # Standardize llm_output structure for callbacks
        llm_output_data = {
            "model_name": self.model_name,
            "token_usage": {  # Ensure this structure exists
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": getattr(call_result, "system_fingerprint", None),
            "request_id": getattr(call_result, "x_request_id", None),
            "finish_reason": generation_info.get("finish_reason"),
            # Keep the raw response for detailed inspection if needed
            "raw_response": call_result.to_dict(),
        }
        # Ensure generation_info also has consistent token usage if needed elsewhere
        # We'll use the same dict structure as llm_output for consistency
        generation_info["token_usage"] = llm_output_data["token_usage"]

        result = ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )
        result.llm_output = llm_output_data  # Use the standardized structure

        if run_manager:
            if hasattr(run_manager, "on_llm_end"):
                # Construct LLMResult for the callback
                generations_for_llm_result: list[
                    list[
                        Union[
                            Generation,
                            ChatGeneration,
                            ChatGenerationChunk,
                            ChatGenerationChunk,
                        ]
                    ]
                ] = [
                    cast(
                        list[
                            Union[
                                Generation,
                                ChatGeneration,
                                ChatGenerationChunk,
                                ChatGenerationChunk,
                            ]
                        ],
                        result.generations,
                    )
                ]
                llm_result_for_callback = LLMResult(
                    generations=generations_for_llm_result,
                    llm_output=result.llm_output,
                    run=None,
                )
                await run_manager.on_llm_end(llm_result_for_callback)
        return result

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously streams chat responses using AsyncLlamaAPIClient."""
        self._ensure_client_initialized()
        if self._async_client is None:
            raise ValueError("AsyncLlamaAPIClient not initialized.")

        active_client = kwargs.get("async_client") or self._async_client
        if not active_client:
            raise ValueError("Could not obtain an active AsyncLlamaAPIClient.")

        effective_tools_lc_input = kwargs.get("tools")
        if (
            effective_tools_lc_input is None and "tools" in kwargs
        ):  # Check if tools came from .bind()
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_astream: Using 'tools' from **kwargs as direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[list[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_astream: effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in a list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]
            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_astream: No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)

        api_params = self._prepare_api_params(
            messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=True,
            **final_kwargs_for_prepare,
        )
        # If streaming, llama-api-client might not support tool_choice in create()
        if api_params.get("stream"):
            api_params.pop(
                "tool_choice", None
            )  # Remove tool_choice if present for streaming

        logger.debug(f"Llama API (async stream) Request: {api_params}")

        # Buffer for aggregating a multi-chunk textual tool call
        # This is a simplified approach for now: we look for full textual tool calls in each chunk's content
        # A more robust solution would buffer across chunks if a textual call is split.

        # New approach for async: Keep track of active tool calls by index
        active_tool_streams_by_index_async: dict[int, dict[str, Any]] = {}

        cumulative_usage_for_gen_info: dict[
            str, Any
        ] = {  # For ChatGeneration.generation_info
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        }

        # Add a flag to track if we yielded any chunks
        yielded_any_chunk = False

        try:
            async for raw_api_chunk in await active_client.chat.completions.create(
                **api_params
            ):
                logger.debug(
                    f"Llama API (async stream) Stream Chunk: {raw_api_chunk.to_dict()}"
                )
                raw_api_chunk_dict = raw_api_chunk.to_dict()

                event_data = getattr(raw_api_chunk, "event", None)
                event_type = getattr(event_data, "event_type", None)
                event_delta = getattr(event_data, "delta", None)
                event_metrics_list = getattr(event_data, "metrics", None)
                chunk_stop_reason = getattr(raw_api_chunk, "stop_reason", None)
                event_stop_reason = getattr(event_data, "stop_reason", None)
                final_stop_reason = event_stop_reason or chunk_stop_reason

                current_gen_info: dict[str, Any] = {"model_name": self.model_name}
                if final_stop_reason:
                    current_gen_info["finish_reason"] = final_stop_reason
                if raw_api_chunk_dict.get("x_request_id"):
                    current_gen_info["x_request_id"] = raw_api_chunk_dict.get(
                        "x_request_id"
                    )

                parsed_tool_call_chunks_for_current_raw_chunk: list[ToolCallChunk] = []

                # Attempt to parse tool calls from event.delta if it indicates a tool_call
                if event_delta and getattr(event_delta, "type", None) == "tool_call":
                    tc_obj = event_delta
                    tool_call_id_from_delta = getattr(tc_obj, "id", None)
                    tool_function = getattr(tc_obj, "function", None)
                    tool_name_from_delta = (
                        getattr(tool_function, "name", None) if tool_function else None
                    )
                    tool_args_str_from_delta = (
                        getattr(tool_function, "arguments", "") or ""
                    )
                    tool_index_from_delta = getattr(tc_obj, "index", 0)

                    if tool_call_id_from_delta and tool_name_from_delta:
                        active_tool_streams_by_index_async[tool_index_from_delta] = {
                            "id": tool_call_id_from_delta,
                            "name": tool_name_from_delta,
                            "args_buffer": tool_args_str_from_delta,
                        }
                        logger.debug(
                            f"_astream (async): Started/updated tool stream for index {tool_index_from_delta}: id={tool_call_id_from_delta}, name={tool_name_from_delta}"
                        )
                        parsed_tool_call_chunks_for_current_raw_chunk.append(
                            ToolCallChunk(
                                name=tool_name_from_delta,
                                args=tool_args_str_from_delta,
                                id=tool_call_id_from_delta,
                                index=tool_index_from_delta,
                            )
                        )
                    elif tool_index_from_delta in active_tool_streams_by_index_async:
                        stored_tool_info = active_tool_streams_by_index_async[
                            tool_index_from_delta
                        ]
                        stored_tool_info["args_buffer"] += tool_args_str_from_delta
                        logger.debug(
                            f"_astream (async): Appending args to tool stream for index {tool_index_from_delta}: id={stored_tool_info['id']}, name={stored_tool_info['name']}"
                        )
                        parsed_tool_call_chunks_for_current_raw_chunk.append(
                            ToolCallChunk(
                                name=None,  # Name/ID can be None for arg delta if index is present
                                args=tool_args_str_from_delta,
                                id=None,  # Will be aggregated later using index
                                index=tool_index_from_delta,
                            )
                        )
                    else:
                        logger.warning(
                            f"_astream (async): Received tool_call delta with args for index {tool_index_from_delta} but no active tool stream started. Delta: {tc_obj}. Yielding with available info."
                        )
                        parsed_tool_call_chunks_for_current_raw_chunk.append(
                            ToolCallChunk(
                                name=tool_name_from_delta,
                                args=tool_args_str_from_delta,
                                id=tool_call_id_from_delta,
                                index=tool_index_from_delta,
                            )
                        )
                # Fallback or alternative: parse from completion_message if no tool calls from event_delta
                elif (
                    hasattr(raw_api_chunk, "completion_message")
                    and raw_api_chunk.completion_message
                    and hasattr(raw_api_chunk.completion_message, "tool_calls")
                    and raw_api_chunk.completion_message.tool_calls
                ):
                    logger.debug(
                        "_astream: Attempting to parse tool_calls from completion_message."
                    )
                    api_tool_calls = raw_api_chunk.completion_message.tool_calls
                    for idx, tc in enumerate(api_tool_calls):
                        func = getattr(tc, "function", None)
                        if func:
                            tool_name = getattr(func, "name", None)
                            tool_args_str = getattr(func, "arguments", "")
                            tool_id = getattr(tc, "id", None)
                            tool_index = getattr(
                                tc, "index", idx
                            )  # Use 'idx' as fallback for index
                            if tool_name and tool_id:
                                parsed_tool_call_chunks_for_current_raw_chunk.append(
                                    ToolCallChunk(
                                        name=tool_name,
                                        args=tool_args_str,
                                        id=tool_id,
                                        index=tool_index,
                                    )
                                )
                                logger.debug(
                                    f"_astream: Parsed ToolCallChunk from completion_message: id={tool_id}, name={tool_name}"
                                )
                            else:
                                logger.warning(
                                    f"_astream: Skipping tool call from completion_message due to missing name or id: {tc}"
                                )
                        else:
                            logger.warning(
                                f"_astream: Tool call structure in completion_message missing 'function' field: {tc}"
                            )

                # Warning if stop_reason indicates tool_calls, but none were parsed
                if (
                    not parsed_tool_call_chunks_for_current_raw_chunk
                    and final_stop_reason == "tool_calls"
                ):
                    logger.warning(
                        f"_astream: Expected tool call due to stop_reason='tool_calls' but did not parse any. Chunk: {raw_api_chunk_dict}"
                    )

                if event_type == "metrics" and isinstance(event_metrics_list, list):
                    logger.debug(
                        f"_astream (async): Entered METRICS event block. event_metrics_list: {event_metrics_list}"
                    )
                    parsed_prompt_tokens = 0
                    parsed_completion_tokens = 0
                    parsed_total_tokens = 0
                    for metric_item in event_metrics_list:
                        metric_name = None
                        metric_value_any = None
                        if hasattr(metric_item, "metric") and hasattr(
                            metric_item, "value"
                        ):
                            metric_name = metric_item.metric
                            metric_value_any = metric_item.value
                        elif isinstance(metric_item, dict):
                            metric_name = metric_item.get("metric")
                            metric_value_any = metric_item.get("value")
                        if metric_name and metric_value_any is not None:
                            metric_value = int(float(metric_value_any))
                            if metric_name == "num_prompt_tokens":
                                parsed_prompt_tokens = metric_value
                            elif metric_name == "num_completion_tokens":
                                parsed_completion_tokens = metric_value
                            elif metric_name == "num_total_tokens":
                                parsed_total_tokens = metric_value
                    cumulative_usage_for_gen_info["input_tokens"] = parsed_prompt_tokens
                    cumulative_usage_for_gen_info["output_tokens"] = (
                        parsed_completion_tokens
                    )
                    cumulative_usage_for_gen_info["total_tokens"] = parsed_total_tokens
                    logger.debug(
                        f"_astream (async): METRICS event - Parsed tokens: p={parsed_prompt_tokens}, c={parsed_completion_tokens}, t={parsed_total_tokens}"
                    )
                    if (
                        parsed_prompt_tokens > 0
                        or parsed_completion_tokens > 0
                        or parsed_total_tokens > 0
                    ):
                        usage_metadata_for_metrics_chunk = UsageMetadata(
                            input_tokens=parsed_prompt_tokens,
                            output_tokens=parsed_completion_tokens,
                            total_tokens=parsed_total_tokens,
                        )
                        logger.debug(
                            f"_astream (async): METRICS EVENT - Created usage_metadata_for_metrics_chunk: {usage_metadata_for_metrics_chunk}"
                        )
                        response_metadata_for_metrics_event = current_gen_info.copy()
                        response_metadata_for_metrics_event.pop("usage_metadata", None)
                        metrics_ai_chunk = AIMessageChunk(
                            content="",
                            usage_metadata=usage_metadata_for_metrics_chunk,
                            response_metadata=response_metadata_for_metrics_event,
                            id=raw_api_chunk_dict.get("id"),
                        )
                        logger.debug(
                            f"_astream (async): METRICS EVENT - Yielding metrics_ai_chunk: {metrics_ai_chunk.to_json()} with gen_info: {current_gen_info}"
                        )
                        gen_info_for_metrics_generation_chunk = current_gen_info.copy()
                        gen_info_for_metrics_generation_chunk["usage_metadata"] = (
                            UsageMetadata(
                                input_tokens=cumulative_usage_for_gen_info.get(
                                    "input_tokens", 0
                                ),
                                output_tokens=cumulative_usage_for_gen_info.get(
                                    "output_tokens", 0
                                ),
                                total_tokens=cumulative_usage_for_gen_info.get(
                                    "total_tokens", 0
                                ),
                            )
                        )
                        yield ChatGenerationChunk(
                            message=metrics_ai_chunk,
                            generation_info=gen_info_for_metrics_generation_chunk,
                        )
                        yielded_any_chunk = True  # Ensure this is set
                    else:
                        logger.debug(
                            "_astream (async): METRICS EVENT - No valid token counts parsed, not yielding metrics_ai_chunk."
                        )
                else:  # Handles "start", "progress" (text part), "complete"
                    content_str = ""
                    # Extract text content only if event_delta is text and not a tool_call type
                    if (
                        event_delta
                        and hasattr(event_delta, "text")
                        and isinstance(event_delta.text, str)
                        and getattr(event_delta, "type", None)
                        != "tool_call"  # Ensure not a tool_call delta
                    ):
                        content_str = event_delta.text
                    if (
                        not content_str and raw_api_chunk_dict
                    ):  # Fallback text extraction
                        try:
                            if "completion_message" in raw_api_chunk_dict:
                                comp_msg = raw_api_chunk_dict["completion_message"]
                                if isinstance(comp_msg, dict) and "content" in comp_msg:
                                    content = comp_msg["content"]
                                    if isinstance(content, dict) and "text" in content:
                                        content_str = content["text"]
                                    elif isinstance(content, str):
                                        content_str = content
                            if (
                                not content_str
                                and "response_metadata" in raw_api_chunk_dict
                            ):
                                response_meta = raw_api_chunk_dict["response_metadata"]
                                if (
                                    isinstance(response_meta, dict)
                                    and "completion_message" in response_meta
                                ):
                                    comp_msg = response_meta["completion_message"]
                                    if (
                                        isinstance(comp_msg, dict)
                                        and "content" in comp_msg
                                    ):
                                        content = comp_msg["content"]
                                        if (
                                            isinstance(content, dict)
                                            and "text" in content
                                        ):
                                            content_str = content["text"]
                                        elif isinstance(content, str):
                                            content_str = content
                        except (KeyError, TypeError):
                            pass
                    response_metadata_for_content_chunk = {
                        "model_name": self.model_name
                    }
                    if final_stop_reason:
                        response_metadata_for_content_chunk["finish_reason"] = (
                            final_stop_reason
                        )
                    if raw_api_chunk_dict.get("x_request_id"):
                        response_metadata_for_content_chunk["x_request_id"] = (
                            raw_api_chunk_dict.get("x_request_id")
                        )
                    if (
                        event_type == "start"
                        or content_str
                        or parsed_tool_call_chunks_for_current_raw_chunk
                        or final_stop_reason  # Yield if there's a stop reason, even if no other content
                    ):
                        lc_chunk_message = AIMessageChunk(
                            content=content_str or "",
                            tool_call_chunks=parsed_tool_call_chunks_for_current_raw_chunk,
                            response_metadata=response_metadata_for_content_chunk,
                            id=raw_api_chunk_dict.get("id"),
                        )
                        final_gen_info_for_chunk = current_gen_info.copy()
                        if (
                            cumulative_usage_for_gen_info.get("input_tokens")
                            or cumulative_usage_for_gen_info.get("output_tokens")
                            or cumulative_usage_for_gen_info.get("total_tokens")
                        ):
                            final_gen_info_for_chunk["usage_metadata"] = UsageMetadata(
                                input_tokens=cumulative_usage_for_gen_info.get(
                                    "input_tokens", 0
                                ),
                                output_tokens=cumulative_usage_for_gen_info.get(
                                    "output_tokens", 0
                                ),
                                total_tokens=cumulative_usage_for_gen_info.get(
                                    "total_tokens", 0
                                ),
                            )
                        lc_chunk = ChatGenerationChunk(
                            message=lc_chunk_message,
                            generation_info=final_gen_info_for_chunk,
                        )
                        if run_manager and hasattr(run_manager, "on_llm_new_token"):
                            token_text = (
                                lc_chunk.text if isinstance(lc_chunk.text, str) else ""
                            )
                            await run_manager.on_llm_new_token(
                                token_text, chunk=lc_chunk
                            )
                        yield lc_chunk
                        yielded_any_chunk = True
        except Exception as e:
            logger.error(f"Error during Llama API async stream: {e}", exc_info=True)
            if run_manager and hasattr(run_manager, "on_llm_error"):
                # Pass a generic response or None if not available
                await run_manager.on_llm_error(error=e, response=None)  # type: ignore[call-arg]

        if not yielded_any_chunk:
            logger.debug(
                "No chunks were yielded during async streaming, providing a fallback empty chunk"
            )
            # Create an empty message chunk containing the structured output format
            empty_ai_message_chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[],
                response_metadata={
                    "model_name": self.model_name,
                    "finish_reason": "tool_calls",
                },
            )

            # Include the full generation info
            final_gen_info = {
                "model_name": self.model_name,
                "finish_reason": "tool_calls",
            }

            # Create a tool call chunk for structured output
            effective_tools_lc_input = kwargs.get("tools")
            prepared_llm_tools = None

            if (
                effective_tools_lc_input
                and isinstance(effective_tools_lc_input, list)
                and effective_tools_lc_input
            ):
                try:
                    tool_name = None
                    first_tool = effective_tools_lc_input[0]
                    logger.debug(
                        f"Fallback chunk (async): attempting to get tool_name from first_tool: {first_tool} (type: {type(first_tool)})"
                    )

                    try:
                        tool_schema_for_conversion = first_tool
                        converted_tool = convert_to_openai_tool(
                            tool_schema_for_conversion
                        )
                        tool_name = converted_tool["function"]["name"]
                        logger.debug(
                            f"Fallback chunk (async): tool_name extracted via convert_to_openai_tool as: {tool_name}"
                        )
                    except Exception as conversion_err:
                        logger.warning(
                            f"Fallback chunk (async): Failed to convert first_tool to OpenAI format for name extraction: {conversion_err}",
                            exc_info=True,
                        )
                        logger.warning(
                            "Fallback chunk (async): Attempting fragile tool_name extraction..."
                        )
                        if isinstance(first_tool, dict):
                            if "name" in first_tool:
                                tool_name = first_tool["name"]
                            elif (
                                "function" in first_tool
                                and isinstance(first_tool["function"], dict)
                                and "name" in first_tool["function"]
                            ):
                                tool_name = first_tool["function"]["name"]
                        else:  # If not a dict, assume it might be a Pydantic model or other class
                            try:
                                if hasattr(first_tool, "name") and not callable(
                                    getattr(first_tool, "name")
                                ):
                                    tool_name = getattr(first_tool, "name")
                                elif hasattr(first_tool, "name") and callable(
                                    getattr(first_tool, "name")
                                ):
                                    tool_name = getattr(
                                        first_tool, "name"
                                    )()  # For BaseTool-like objects
                                elif hasattr(
                                    first_tool, "__name__"
                                ):  # For classes (like Pydantic models)
                                    tool_name = getattr(first_tool, "__name__")
                            except Exception as attr_err:
                                logger.warning(
                                    f"Fallback chunk (async): Fragile tool_name extraction via attributes failed: {attr_err}"
                                )
                                pass
                        if not tool_name:
                            try:
                                tool_name = str(first_tool).split()[0]
                            except Exception as str_err:
                                logger.warning(
                                    f"Fallback chunk (async): Fragile tool_name extraction via str() failed: {str_err}"
                                )
                        logger.debug(
                            f"Fallback chunk (async): tool_name from fragile extraction: {tool_name}"
                        )

                    if tool_name:
                        tool_id = str(uuid.uuid4())
                        parsed_tool_call_chunks = [
                            ToolCallChunk(
                                name=tool_name,
                                args="{}",  # Use empty JSON object for args
                                id=tool_id,
                                index=0,
                            )
                        ]
                        empty_ai_message_chunk.tool_call_chunks = (
                            parsed_tool_call_chunks
                        )
                    else:
                        logger.warning(
                            "Could not determine tool_name for fallback chunk, tool_call_chunks will be empty."
                        )

                except Exception as e:
                    logger.warning(f"Failed to create fallback tool call chunk: {e}")

            # Yield the fallback chunk
            yield ChatGenerationChunk(
                message=empty_ai_message_chunk, generation_info=final_gen_info
            )

    async def _astream_with_aggregation_and_retries(
        self,
        messages: list[BaseMessage],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        final_generation_info = {}

        async for chunk in self._astream(
            messages=messages, run_manager=run_manager, **kwargs
        ):
            if chunk.generation_info:
                final_generation_info.update(chunk.generation_info)

            # Process tool_call_chunks from the current yielded chunk
            # These are already in LangChain's ToolCallChunk format
            # AIMessageChunk wants a list of these chunk dicts
            tool_call_chunks_for_aimessagechunk = []
            if (
                hasattr(
                    chunk, "message"
                )  # First check if chunk has a message attribute
                and isinstance(
                    chunk.message, AIMessageChunk
                )  # Then check if it's an AIMessageChunk
                and hasattr(
                    chunk.message, "tool_call_chunks"
                )  # Then check if it has tool_call_chunks
                and chunk.message.tool_call_chunks  # And if that list is not empty
            ):
                tool_call_chunks_for_aimessagechunk = chunk.message.tool_call_chunks

            # Yield a new chunk that correctly represents the state *after* this incoming chunk
            # The AIMessageChunk content should be the delta (chunk.text), not aggregated across all prior chunks for yielding.
            # LangChain expects the AIMessageChunk.content to be the new token(s) in *this* chunk.
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=chunk.text,  # This is the delta for this chunk
                    tool_call_chunks=tool_call_chunks_for_aimessagechunk,
                ),
                generation_info=chunk.generation_info.copy()
                if chunk.generation_info
                else None,  # Pass along this chunk's specific info
            )
            # No further aggregation or yielding needed here; _aget_stream_results will handle the final aggregation.

    async def _aget_stream_results(
        self,
        completion_coro: AsyncIterator[ChatGenerationChunk],
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> ChatResult:
        """Aggregates results from a stream of ChatGenerationChunks."""
        aggregated_content = ""

        # For tool calls, we need to aggregate potentially partial arguments for the same tool call ID.
        # LangChain's AIMessage expects fully formed tool_calls (List[Dict]), not ToolCallChunks.
        # We'll collect all ToolCallChunk data and then reconstruct full ToolCall dicts.

        # Store tool call parts by index, then by id.
        # Each entry in aggregated_tool_call_parts will be a dict like:
        # { 'id': str, 'name': Optional[str], 'args_str_parts': List[str], 'type': str }
        aggregated_tool_call_parts_by_index: dict[int, dict[str, Any]] = {}

        final_generation_info_aggregated = {}
        last_chunk_for_finish_reason = None
        raw_response_dict_for_llm_output = None

        async for chunk in completion_coro:
            aggregated_content += chunk.text
            if chunk.generation_info:
                final_generation_info_aggregated.update(
                    chunk.generation_info
                )  # Overwrite with later info

            last_chunk_for_finish_reason = chunk  # Keep track of the last chunk
            if chunk.generation_info and isinstance(
                chunk.generation_info.get("response_metadata"), dict
            ):
                raw_response_dict_for_llm_output = chunk.generation_info.get(
                    "response_metadata"
                )
            elif chunk.generation_info:  # last resort for some dict
                raw_response_dict_for_llm_output = chunk.generation_info

            if (
                hasattr(
                    chunk, "message"
                )  # First check if chunk has a message attribute
                and isinstance(
                    chunk.message, AIMessageChunk
                )  # Then check if it's an AIMessageChunk
                and hasattr(
                    chunk.message, "tool_call_chunks"
                )  # Then check if it has tool_call_chunks
                and chunk.message.tool_call_chunks  # And if that list is not empty
            ):
                for tc_chunk in chunk.message.tool_call_chunks:
                    idx = tc_chunk.get("index")
                    tc_id = tc_chunk.get("id")
                    if idx is not None:
                        if idx not in aggregated_tool_call_parts_by_index:
                            aggregated_tool_call_parts_by_index[idx] = {
                                "id": tc_id,
                                "name": tc_chunk.get("name"),
                                "args_str_parts": [],
                                "type": tc_chunk.get("type", "function"),
                            }
                        if tc_chunk.get(
                            "name"
                        ) and not aggregated_tool_call_parts_by_index[idx].get("name"):
                            aggregated_tool_call_parts_by_index[idx]["name"] = (
                                tc_chunk.get("name")
                            )
                        if tc_id and not aggregated_tool_call_parts_by_index[idx].get(
                            "id"
                        ):
                            aggregated_tool_call_parts_by_index[idx]["id"] = tc_id
                        args_delta = tc_chunk.get("args")
                        if isinstance(args_delta, str):
                            aggregated_tool_call_parts_by_index[idx][
                                "args_str_parts"
                            ].append(args_delta)

        # Reconstruct final tool_calls for AIMessage
        final_tool_calls_for_aimessage: list[dict[str, Any]] = []
        for _idx, parts in sorted(
            aggregated_tool_call_parts_by_index.items()
        ):  # Process in order of index
            full_args_str = "".join(parts["args_str_parts"])
            parsed_args: Union[dict, str]
            try:
                parsed_args = json.loads(full_args_str) if full_args_str else {}
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse aggregated tool call arguments for tool {parts.get('name')}. Using raw string."
                )
                parsed_args = full_args_str

            # Ensure parsed_args is a dict for the final AIMessage tool_call
            final_args_for_aimessage: dict
            if isinstance(parsed_args, dict):
                final_args_for_aimessage = parsed_args
            else:  # If not a dict (e.g. was a string, or json.loads returned a non-dict)
                final_args_for_aimessage = {"value": str(parsed_args)}

            final_tool_calls_for_aimessage.append(
                {
                    "id": parts.get("id") or str(uuid.uuid4()),  # Ensure ID
                    "name": parts.get("name") or "unknown_tool",
                    "args": final_args_for_aimessage,
                    "type": parts.get("type", "function"),
                }
            )

        # Determine final finish_reason from the very last chunk or accumulated info
        if (
            last_chunk_for_finish_reason
            and last_chunk_for_finish_reason.generation_info
        ):
            final_generation_info_aggregated["finish_reason"] = (
                last_chunk_for_finish_reason.generation_info.get(
                    "finish_reason", "stop"
                )
            )
        elif "finish_reason" not in final_generation_info_aggregated:
            final_generation_info_aggregated["finish_reason"] = "stop"

        # Ensure 'duration' is present if other metrics like usage are (LangSmith might expect it)
        # This might be better handled by run_manager.on_llm_end if it calculates total duration.
        # For now, we'll ensure it's there if other usage data is.
        if (
            final_generation_info_aggregated.get("usage_metadata")
            and "duration" not in final_generation_info_aggregated
        ):
            # Placeholder if not set by _astream's final chunk info.
            # A more accurate duration would be from the start of _agenerate to now.
            final_generation_info_aggregated["duration"] = 0

        # Calculate final token usage (might be incomplete if API didn't send it)
        aggregated_token_usage = final_generation_info_aggregated.get(
            "usage_metadata", {}
        )
        prompt_tokens = aggregated_token_usage.get(
            "input_tokens", 0
        )  # Default to 0 if missing
        completion_tokens = aggregated_token_usage.get(
            "output_tokens", 0
        )  # Default to 0 if missing
        total_tokens = aggregated_token_usage.get(
            "total_tokens", prompt_tokens + completion_tokens
        )

        final_message = AIMessage(
            content=aggregated_content,
            tool_calls=final_tool_calls_for_aimessage,
        )
        # Ensure the message ID is set if not already, for tracking
        if not hasattr(final_message, "id") or not final_message.id:
            final_message.id = str(uuid.uuid4())

        # Attach usage metadata to the final message if available
        if aggregated_token_usage:
            try:
                final_message.usage_metadata = UsageMetadata(
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                )
            except Exception as e:
                logger.warning(
                    f"Could not construct UsageMetadata for final message: {e}"
                )

        # Update generation_info for consistency before creating ChatResult
        final_generation_info_aggregated["token_usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

        # ADDING LOGGING HERE
        logger.debug(
            f"_aget_stream_results: final_message ID: {final_message.id}"
        )  # Log ID
        logger.debug(
            f"_aget_stream_results: final_message.content: '{final_message.content}'"
        )
        logger.debug(
            f"_aget_stream_results: final_message.tool_calls: {final_message.tool_calls}"
        )
        logger.debug(
            f"_aget_stream_results: final_generation_info_aggregated: {final_generation_info_aggregated}"
        )

        final_result = ChatResult(
            generations=[
                ChatGeneration(
                    message=final_message,
                    generation_info=final_generation_info_aggregated,
                )
            ]
        )

        # Standardize llm_output for streaming
        llm_output_data = {
            "model_name": self.model_name,
            "token_usage": final_generation_info_aggregated["token_usage"],
            "system_fingerprint": final_generation_info_aggregated.get(
                "system_fingerprint"
            ),
            "request_id": final_generation_info_aggregated.get("x_request_id"),
            "finish_reason": final_generation_info_aggregated.get("finish_reason"),
            # Maybe include the last raw chunk dict if useful, or None for streaming
            "raw_response": raw_response_dict_for_llm_output,  # Might be None or last chunk's data
        }
        final_result.llm_output = llm_output_data

        if run_manager and hasattr(run_manager, "on_llm_end"):
            # Construct LLMResult for the callback
            generations_for_llm_result_stream: list[
                list[
                    Union[
                        Generation,
                        ChatGeneration,
                        ChatGenerationChunk,
                        ChatGenerationChunk,
                    ]
                ]
            ] = [
                cast(
                    list[
                        Union[
                            Generation,
                            ChatGeneration,
                            ChatGenerationChunk,
                            ChatGenerationChunk,
                        ]
                    ],
                    final_result.generations,
                )
            ]
            llm_result_for_callback = LLMResult(
                generations=generations_for_llm_result_stream,
                llm_output=llm_output_data,
                run=None,
            )
            await run_manager.on_llm_end(llm_result_for_callback)

        return final_result
