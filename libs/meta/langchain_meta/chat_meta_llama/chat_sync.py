import json
import logging
import re  # Added re
import uuid
from collections.abc import Iterator
from datetime import datetime
from typing import (
    Any,
    Callable,
    Literal,
    Optional,
    Union,
)

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,  # Used by _lc_message_to_llama_message_param if that was here,
    ToolCallChunk,  # Added import
)
from langchain_core.messages.ai import UsageMetadata  # Added import
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
    LLMResult,  # Added import for type conversion
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool  # Import added
from llama_api_client import LlamaAPIClient
from llama_api_client.types.chat import completion_create_params

# from llama_api_client.types.create_chat_completion_response import CreateChatCompletionResponse # Only for async
from pydantic import BaseModel

from ..utils import parse_malformed_args_string  # Import from main utils

# Assuming chat_models.py is in langchain_meta.chat_models
# and contains helper functions like _lc_tool_to_llama_tool_param and _prepare_api_params
from .serialization import (
    _lc_tool_to_llama_tool_param,
    _parse_textual_tool_args,
)  # Changed from ..chat_models

logger = logging.getLogger(__name__)


class SyncChatMetaLlamaMixin:
    """Mixin class to hold synchronous methods for ChatMetaLlama."""

    # Type hints for attributes/methods from ChatMetaLlama main class
    # that are used by these sync methods via `self`.
    _client: Optional[LlamaAPIClient]
    model_name: str
    temperature: Optional[float]
    max_tokens: Optional[int]
    repetition_penalty: Optional[float]

    # Methods from the main class or other mixins expected to be available on self
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

    # _lc_tool_to_llama_tool_param is imported and used directly
    # _lc_message_to_llama_message_param is imported and used by _prepare_api_params (assumed to be on self or accessible)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[list[Union[dict, type[BaseModel], Callable, BaseTool]]] = None,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any", "required"]]
        ] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate a chat response using the sync API client."""
        self._ensure_client_initialized()
        if self._client is None:
            raise ValueError("LlamaAPIClient not initialized.")

        active_client = kwargs.get("client") or self._client
        if not active_client:
            raise ValueError("Could not obtain an active LlamaAPIClient.")

        start_time = datetime.now()
        input_tokens = self._count_tokens(messages)

        # === Callback Handling Start ===
        llm_run_manager: Optional[CallbackManagerForLLMRun] = None
        if run_manager:
            if isinstance(run_manager, CallbackManagerForLLMRun):
                llm_run_manager = run_manager
                logger.debug(
                    "Inside _generate: run_manager is already CallbackManagerForLLMRun."
                )
            elif hasattr(run_manager, "get_child"):
                llm_run_manager = run_manager.get_child()
                logger.debug("Inside _generate: Called run_manager.get_child().")
            else:
                logger.warning(
                    f"Inside _generate: run_manager is of unexpected type {type(run_manager)} and has no get_child. Callbacks may not work correctly."
                )
                llm_run_manager = run_manager

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
        if llm_run_manager and hasattr(llm_run_manager, "on_llm_start"):
            try:
                # Pass options to callback
                on_llm_start_fn = getattr(llm_run_manager, "on_llm_start")
                on_llm_start_fn(
                    {"name": self.__class__.__name__},
                    messages,
                    invocation_params=self._get_invocation_params(**kwargs),
                    options=callback_options,
                )
            except Exception as e:
                logger.warning(f"Error in on_llm_start callback: {str(e)}")

        # === Callback Handling End ===

        effective_tools_lc_input = tools
        if effective_tools_lc_input is None and "tools" in kwargs:
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_generate (sync): Using 'tools' from **kwargs as direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[list[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_generate (sync): effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in a list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]
            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_generate (sync): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)
        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice

        api_params = self._prepare_api_params(
            messages=messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=False,
            **final_kwargs_for_prepare,
        )

        if self.temperature is not None and "temperature" not in api_params:
            api_params["temperature"] = self.temperature
        # max_tokens from self.max_tokens is handled by _prepare_api_params if it's in **kwargs as max_completion_tokens
        # or if self.max_tokens is directly used by _prepare_api_params.
        # Here, we ensure it's passed if not already set by _prepare_api_params logic
        if (
            self.max_tokens is not None and "max_completion_tokens" not in api_params
        ):  # llama client uses max_completion_tokens
            if (
                "max_tokens" not in api_params
            ):  # Check if 'max_tokens' alias is also not there
                api_params["max_completion_tokens"] = self.max_tokens

        if (
            self.repetition_penalty is not None
            and "repetition_penalty" not in api_params
        ):
            api_params["repetition_penalty"] = self.repetition_penalty

        logger.debug(f"Llama API (sync) Request: {api_params}")
        try:
            call_result = active_client.chat.completions.create(**api_params)
            logger.debug(f"Llama API (sync) Response: {call_result}")
        except Exception as e:
            if llm_run_manager:  # Check if llm_run_manager was successfully obtained
                llm_run_manager.on_llm_error(error=e)  # type: ignore[attr-defined]
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

        # If still no content but we have response data, traverse known structures - attempt 3
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

                    # If there's still no content but response_metadata exists and has completion_message
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

        tool_calls_data: list[dict] = []
        generation_info: dict[str, Any] = {}  # Initialize generation_info here

        if result_msg and hasattr(result_msg, "tool_calls") and result_msg.tool_calls:
            processed_tool_calls: list[dict] = []
            for idx, tc in enumerate(result_msg.tool_calls):
                tc_id = getattr(tc, "id", None)
                if not tc_id:
                    tc_id = f"llama_tc_{idx}"
                if not tc_id:
                    tc_id = str(uuid.uuid4())

                tc_func = tc.function if hasattr(tc, "function") else None
                tc_name = getattr(tc_func, "name", None) if tc_func else None
                tc_args_str = getattr(tc_func, "arguments", "") or ""

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
                    # Try our malformed args parser for cases like 'name="value", key2="value2"'
                    logger.debug(
                        f"JSON parsing failed, trying malformed args parser for: {tc_args_str}"
                    )
                    final_args = parse_malformed_args_string(tc_args_str)
                except Exception as e:
                    logger.warning(
                        f"Unexpected error processing tool call arguments for {tc_name}: {e}. Representing as string."
                    )
                    final_args = {"value": tc_args_str}

                # Defensive: always ensure id, name, args are properly set
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
        elif (
            prepared_llm_tools
            and content_str
            and content_str.startswith("[")
            and content_str.endswith("]")
        ):
            # If no tool_calls from API, try to parse from content_str if tools were provided and content looks like a textual tool call
            logger.debug(
                f"No structured tool_calls from API. Attempting to parse textual tool call from content: {content_str}"
            )
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
                            # First try the standard LangChain parser
                            parsed_args = _parse_textual_tool_args(
                                args_str_from_content
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to parse arguments '{args_str_from_content}' for textual tool call '{tool_name_from_content}': {e}. Trying fallback parser."
                            )
                            # Use our fallback parser for malformed argument strings
                            parsed_args = parse_malformed_args_string(
                                args_str_from_content
                            )

                    # Defensive: always ensure all fields are properly set
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
                            "type": "function",  # LangChain expects this structure
                        }
                    )
                    content_str = (
                        ""  # Clear content as it was a tool call representation
                    )
                    logger.debug(f"Manually constructed tool_calls: {tool_calls_data}")
                    # If we manually created tool_calls, the stop_reason should reflect that.
                    # We'll store this in generation_info, which AIMessage can use.
                    generation_info["finish_reason"] = "tool_calls"
                else:
                    logger.warning(
                        f"Textual tool call '{tool_name_from_content}' found in content, but not in available tools: {available_tool_names}"
                    )
            else:
                logger.debug(
                    f"Content '{content_str}' did not match textual tool call pattern."
                )

        message = AIMessage(
            content=content_str or "",
            tool_calls=tool_calls_data,
            generation_info=generation_info if generation_info else None,
        )
        prompt_tokens = input_tokens  # re-assign from initial count
        completion_tokens = 0

        if result_msg and hasattr(result_msg, "stop_reason") and result_msg.stop_reason:
            generation_info["finish_reason"] = result_msg.stop_reason
        elif hasattr(call_result, "stop_reason") and call_result.stop_reason:
            generation_info["finish_reason"] = call_result.stop_reason

        if (
            hasattr(call_result, "metrics")
            and call_result.metrics
            and isinstance(call_result.metrics, list)
        ):
            usage_meta = {}
            for item in call_result.metrics:
                if hasattr(item, "metric") and hasattr(item, "value"):
                    # Cast value to int here
                    metric_value = int(item.value) if item.value is not None else 0
                    if item.metric == "num_prompt_tokens":
                        usage_meta["input_tokens"] = metric_value
                        prompt_tokens = metric_value
                    elif item.metric == "num_completion_tokens":
                        usage_meta["output_tokens"] = metric_value
                        completion_tokens = metric_value
                    elif item.metric == "num_total_tokens":
                        usage_meta["total_tokens"] = metric_value
            if usage_meta:
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):  # Check before assigning
                    try:
                        constructed_usage = UsageMetadata(
                            input_tokens=usage_meta.get("input_tokens", 0),
                            output_tokens=usage_meta.get("output_tokens", 0),
                            total_tokens=usage_meta.get("total_tokens", 0),
                        )
                        message.usage_metadata = constructed_usage
                    except Exception as e:
                        logger.warning(
                            f"Could not construct UsageMetadata for AIMessage: {e}"
                        )
        elif hasattr(call_result, "usage") and call_result.usage:  # Fallback
            usage_data = call_result.usage
            # Cast values to int here
            prompt_tokens = int(getattr(usage_data, "prompt_tokens", 0))
            completion_tokens = int(getattr(usage_data, "completion_tokens", 0))
            total_tokens = int(getattr(usage_data, "total_tokens", 0))
            usage_meta = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,  # Use already casted int values
            }
            # prompt_tokens and completion_tokens are already updated above
            if any(usage_meta.values()):
                generation_info["usage_metadata"] = usage_meta
                if hasattr(message, "usage_metadata"):  # Check before assigning
                    try:
                        constructed_usage = UsageMetadata(
                            input_tokens=usage_meta.get("input_tokens", 0),
                            output_tokens=usage_meta.get("output_tokens", 0),
                            total_tokens=usage_meta.get("total_tokens", 0),
                        )
                        message.usage_metadata = constructed_usage
                    except Exception as e:
                        logger.warning(
                            f"Could not construct UsageMetadata (fallback) for AIMessage: {e}"
                        )

        if hasattr(call_result, "x_request_id") and call_result.x_request_id:
            generation_info["x_request_id"] = call_result.x_request_id
        # generation_info["response_metadata"] = call_result.to_dict() # This would overwrite our potential manual finish_reason
        # Preserve existing generation_info and add to it carefully
        response_metadata_dict = call_result.to_dict()
        if (
            "response_metadata" not in generation_info
        ):  # if we haven't manually set parts of it
            generation_info["response_metadata"] = response_metadata_dict
        else:  # Merge, with our manual values taking precedence if keys conflict (e.g. finish_reason)
            generation_info["response_metadata"] = {
                **response_metadata_dict,
                **generation_info.get("response_metadata", {}),
                **generation_info,
            }
            # The above merge is a bit complex, simplify: ensure original response_metadata is base, then overlay our gen_info
            base_response_meta = response_metadata_dict
            current_gen_info = (
                generation_info.copy()
            )  # our potentially modified generation_info
            generation_info = base_response_meta  # start with full API response
            generation_info.update(current_gen_info)  # overlay our modifications

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        generation_info["duration"] = duration

        result = ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ]
        )

        # --- Standardize llm_output for callbacks ---
        llm_output_data = {
            "model_name": self.model_name,
            # Ensure token_usage is a dictionary within llm_output
            "token_usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
            "system_fingerprint": getattr(
                call_result, "system_fingerprint", None
            ),  # Add if available
            "request_id": getattr(
                call_result, "x_request_id", None
            ),  # Add if available
            "finish_reason": generation_info.get("finish_reason"),  # Add if available
            # Include the raw response if needed for debugging, but maybe exclude from standard callback data
            "raw_response_metadata": call_result.to_dict(),
        }
        result.llm_output = llm_output_data  # Assign the standardized dict
        # --- End Standardization ---

        # === Callback Handling Start for on_llm_end ===
        if llm_run_manager:
            try:
                # Convert ChatResult to LLMResult for on_llm_end callback
                # The on_llm_end method expects an LLMResult, not a ChatResult
                # Use a type ignore comment to bypass the type checker
                llm_result = LLMResult(
                    generations=[[gen for gen in result.generations]],  # type: ignore
                    llm_output=result.llm_output,
                )
                llm_run_manager.on_llm_end(llm_result)
            except Exception as e:
                logger.warning(f"Error in on_llm_end callback: {str(e)}")
                if (
                    isinstance(e, KeyError) and e.args and e.args[0] == 0
                ):  # Check args exist before indexing
                    logger.error(
                        f"(Sync - Still seeing KeyError(0)) Detail: Result llm_output: {result.llm_output}"
                    )

        # === Callback Handling End for on_llm_end ===

        return result

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        tools: Optional[list[Union[dict, type[BaseModel], Callable, BaseTool]]] = None,
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "any", "required"]]
        ] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Synchronously streams chat responses using LlamaAPIClient."""
        self._ensure_client_initialized()
        if self._client is None:
            raise ValueError("LlamaAPIClient not initialized.")

        # Use client from kwargs if provided (e.g. for testing), else use self._client
        active_client = kwargs.get("client") or self._client
        if not active_client:  # Should be caught by above, but defensive
            raise ValueError("Could not obtain an active LlamaAPIClient.")

        effective_tools_lc_input = tools
        if effective_tools_lc_input is None and "tools" in kwargs:
            effective_tools_lc_input = kwargs.get("tools")
            logger.debug(
                "_stream (sync): Using 'tools' from **kwargs as direct 'tools' parameter was None."
            )

        prepared_llm_tools: Optional[list[completion_create_params.Tool]] = None
        if effective_tools_lc_input:
            tools_list_to_prepare = effective_tools_lc_input
            if not isinstance(effective_tools_lc_input, list):
                logger.warning(
                    f"_stream (sync): effective_tools_lc_input was not a list ({type(effective_tools_lc_input)}). Wrapping in list."
                )
                tools_list_to_prepare = [effective_tools_lc_input]
            prepared_llm_tools = [
                _lc_tool_to_llama_tool_param(tool) for tool in tools_list_to_prepare
            ]
        else:
            logger.debug("_stream (sync): No effective tools to prepare.")

        final_kwargs_for_prepare = kwargs.copy()
        final_kwargs_for_prepare.pop("tools", None)
        if tool_choice is not None:
            final_kwargs_for_prepare["tool_choice"] = tool_choice
        # tool_choice is implicitly handled by Llama API if tools are present or not for streaming.
        # If tool_choice needs to be explicitly passed for streaming, it would go into final_kwargs_for_prepare.

        api_params = self._prepare_api_params(
            messages,
            tools=prepared_llm_tools,
            stop=stop,
            stream=True,
            **final_kwargs_for_prepare,
        )
        logger.debug(f"Llama API (sync stream) Request: {api_params}")

        cumulative_usage_for_gen_info: dict[
            str, Any
        ] = {  # For ChatGeneration.generation_info
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            # model_name will be added per chunk's generation_info
        }

        # New approach: Keep track of active tool calls by index to get their ID and Name for subsequent arg chunks
        # Key: index (int), Value: {"id": str, "name": str, "args_buffer": str} (args_buffer for full reconstruction if needed, not for yielding chunks)
        active_tool_streams_by_index: dict[int, dict[str, Any]] = {}

        yielded_any_chunk = False

        try:
            streaming_chunks = active_client.chat.completions.create(**api_params)
            for chunk_result in streaming_chunks:
                logger.debug(
                    f"Llama API (sync stream) Stream Chunk: {chunk_result.to_dict()}"
                )
                chunk_dict = (
                    chunk_result.to_dict()
                )  # Keep for potential x_request_id or other metadata

                event_data = getattr(chunk_result, "event", None)
                event_type = getattr(event_data, "event_type", None)
                event_delta = getattr(event_data, "delta", None)
                event_metrics_list = getattr(event_data, "metrics", None)
                chunk_stop_reason = getattr(chunk_result, "stop_reason", None)
                event_stop_reason = getattr(event_data, "stop_reason", None)
                final_stop_reason = event_stop_reason or chunk_stop_reason

                current_gen_info: dict[str, Any] = {"model_name": self.model_name}
                if final_stop_reason:
                    current_gen_info["finish_reason"] = final_stop_reason
                if chunk_dict.get("x_request_id"):
                    current_gen_info["x_request_id"] = chunk_dict.get("x_request_id")

                parsed_tool_call_chunks: list[ToolCallChunk] = []

                if event_delta and getattr(event_delta, "type", None) == "tool_call":
                    tc_delta_obj = event_delta
                    tool_call_id_from_delta = getattr(tc_delta_obj, "id", None)
                    tool_function = getattr(tc_delta_obj, "function", None)
                    tool_name_from_delta = (
                        getattr(tool_function, "name", None) if tool_function else None
                    )
                    tool_args_str_from_delta = (
                        getattr(tool_function, "arguments", "") or ""
                    )
                    tool_index_from_delta = getattr(tc_delta_obj, "index", 0)

                    # Determine if this is the first chunk for this tool call index

                    if tool_call_id_from_delta and tool_name_from_delta:
                        # This chunk establishes or re-establishes the tool call for this index
                        active_tool_streams_by_index[tool_index_from_delta] = {
                            "id": tool_call_id_from_delta,
                            "name": tool_name_from_delta,
                            "args_buffer": tool_args_str_from_delta,
                        }
                        logger.debug(
                            f"_stream (sync): Started/updated tool stream for index {tool_index_from_delta}: id={tool_call_id_from_delta}, name={tool_name_from_delta}"
                        )
                        # Yield with full ID and Name, as this is the defining chunk for this tool call (or a re-affirmation)
                        parsed_tool_call_chunks.append(
                            ToolCallChunk(
                                name=tool_name_from_delta,
                                args=tool_args_str_from_delta,
                                id=tool_call_id_from_delta,
                                index=tool_index_from_delta,
                            )
                        )
                    elif tool_index_from_delta in active_tool_streams_by_index:
                        # This is a continuation chunk (only args, or args with redundant/no id/name)
                        # BaseOpenAIToolsParser expects subsequent chunks for an index to have name=None, id=None
                        stored_tool_info = active_tool_streams_by_index[
                            tool_index_from_delta
                        ]
                        stored_tool_info["args_buffer"] += (
                            tool_args_str_from_delta  # Continue buffering full args for internal tracking
                        )
                        logger.debug(
                            f"_stream (sync): Appending args to tool stream for index {tool_index_from_delta}: id={stored_tool_info['id']}, name={stored_tool_info['name']}"
                        )
                        # Yield with name=None, id=None for continuation chunks
                        parsed_tool_call_chunks.append(
                            ToolCallChunk(
                                name=None,  # Critical: subsequent chunks for same index should not repeat name
                                args=tool_args_str_from_delta,
                                id=None,  # Critical: subsequent chunks for same index should not repeat id
                                index=tool_index_from_delta,
                            )
                        )
                    else:
                        # Argument chunk arrived for a tool index we haven't seen a start for.
                        logger.warning(
                            f"_stream (sync): Received tool_call delta with args for index {tool_index_from_delta} but no active tool stream started. Delta: {tc_delta_obj}. Yielding with available info."
                        )
                        # Yield what we have, parser might handle or error appropriately
                        parsed_tool_call_chunks.append(
                            ToolCallChunk(
                                name=tool_name_from_delta,  # Might be None
                                args=tool_args_str_from_delta,
                                id=tool_call_id_from_delta,  # Might be None
                                index=tool_index_from_delta,
                            )
                        )

                if event_type == "metrics" and isinstance(event_metrics_list, list):
                    logger.debug(
                        f"_stream (sync): Entered METRICS event block. event_metrics_list: {event_metrics_list}"
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
                        f"_stream (sync): METRICS event - Parsed tokens: p={parsed_prompt_tokens}, c={parsed_completion_tokens}, t={parsed_total_tokens}"
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
                            f"_stream (sync): METRICS EVENT - Created usage_metadata_for_metrics_chunk: {usage_metadata_for_metrics_chunk}"
                        )
                        response_metadata_for_metrics_event = current_gen_info.copy()
                        response_metadata_for_metrics_event.pop("usage_metadata", None)
                        metrics_ai_chunk = AIMessageChunk(
                            content="",
                            usage_metadata=usage_metadata_for_metrics_chunk,
                            response_metadata=response_metadata_for_metrics_event,
                            id=chunk_dict.get("id"),
                        )
                        logger.debug(
                            f"_stream (sync): METRICS EVENT - Yielding metrics_ai_chunk: {metrics_ai_chunk.to_json()} with gen_info: {current_gen_info}"
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
                        yielded_any_chunk = True
                    else:
                        logger.debug(
                            "_stream (sync): METRICS EVENT - No valid token counts parsed, not yielding metrics_ai_chunk."
                        )
                else:
                    content_str = ""
                    if (
                        event_delta
                        and hasattr(event_delta, "text")
                        and isinstance(event_delta.text, str)
                    ):
                        content_str = event_delta.text
                    if not content_str and chunk_dict:
                        try:
                            if "completion_message" in chunk_dict:
                                comp_msg = chunk_dict["completion_message"]
                                if isinstance(comp_msg, dict) and "content" in comp_msg:
                                    content = comp_msg["content"]
                                    if isinstance(content, dict) and "text" in content:
                                        content_str = content["text"]
                                    elif isinstance(content, str):
                                        content_str = content
                            if not content_str and "response_metadata" in chunk_dict:
                                response_meta = chunk_dict["response_metadata"]
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
                    if chunk_dict.get("x_request_id"):
                        response_metadata_for_content_chunk["x_request_id"] = (
                            chunk_dict.get("x_request_id")
                        )
                    if (
                        event_type == "start"
                        or content_str
                        or parsed_tool_call_chunks
                        or final_stop_reason
                    ):
                        ai_message_chunk = AIMessageChunk(
                            content=content_str or "",
                            tool_call_chunks=parsed_tool_call_chunks
                            if parsed_tool_call_chunks
                            else [],
                            response_metadata=response_metadata_for_content_chunk,
                            id=chunk_dict.get("id"),
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
                        chat_gen_chunk = ChatGenerationChunk(
                            message=ai_message_chunk,
                            generation_info=final_gen_info_for_chunk,
                        )
                        yield chat_gen_chunk
                        if run_manager and hasattr(run_manager, "on_llm_new_token"):
                            token_text = (
                                ai_message_chunk.text
                                if isinstance(ai_message_chunk.text, str)
                                else ""
                            )
                            run_manager.on_llm_new_token(
                                token_text,
                                chunk=chat_gen_chunk,
                            )
                        yielded_any_chunk = True
        except Exception as e:
            logger.error(f"Error during Llama API stream: {e}", exc_info=True)
            if run_manager and hasattr(run_manager, "on_llm_error"):
                # Pass a generic response or None if not available
                run_manager.on_llm_error(error=e, response=None)  # type: ignore[call-arg]

        if not yielded_any_chunk:
            logger.debug(
                "No chunks were yielded during streaming, providing a fallback empty chunk"
            )
            # Create an empty message chunk containing the structured output format
            empty_ai_message_chunk = AIMessageChunk(
                content="",
                tool_call_chunks=[],
                response_metadata={
                    "model_name": self.model_name,
                    "finish_reason": "tool_calls",  # Changed from "stop" to "tool_calls"
                },
            )

            # Include the full generation info
            final_gen_info = {
                "model_name": self.model_name,
                "finish_reason": "tool_calls",
            }

            # Create a tool call chunk for structured output
            if (
                effective_tools_lc_input
                and isinstance(effective_tools_lc_input, list)
                and effective_tools_lc_input
            ):
                try:
                    tool_name = None
                    first_tool = effective_tools_lc_input[0]
                    logger.debug(
                        f"Fallback chunk: attempting to get tool_name from first_tool: {first_tool} (type: {type(first_tool)})"
                    )

                    try:
                        # Ensure schema is a Pydantic model or a dict for convert_to_openai_tool
                        tool_schema_for_conversion = first_tool
                        # if isinstance(first_tool, type) and issubclass(first_tool, BaseModel):
                        #     tool_schema_for_conversion = first_tool # Already good
                        # elif not isinstance(first_tool, dict):
                        #     # Attempt to convert to dict if it's some other callable or BaseTool, though this path is less likely for structured output schema
                        #     logger.warning(f"Fallback first_tool is not a Pydantic class or dict, attempting conversion for tool name. This might be unstable.")
                        #     # This path might be too complex; convert_to_openai_tool should handle Pydantic models directly.

                        converted_tool = convert_to_openai_tool(
                            tool_schema_for_conversion
                        )
                        tool_name = converted_tool["function"]["name"]
                        logger.debug(
                            f"Fallback chunk: tool_name extracted via convert_to_openai_tool as: {tool_name}"
                        )
                    except Exception as conversion_err:
                        logger.warning(
                            f"Fallback chunk: Failed to convert first_tool to OpenAI format for name extraction: {conversion_err}",
                            exc_info=True,
                        )
                        logger.warning(
                            "Fallback chunk: Attempting fragile tool_name extraction..."
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
                                    f"Fallback chunk: Fragile tool_name extraction via attributes failed: {attr_err}"
                                )
                                pass
                        if not tool_name:
                            try:
                                tool_name = str(first_tool).split()[0]
                            except Exception as str_err:
                                logger.warning(
                                    f"Fallback chunk: Fragile tool_name extraction via str() failed: {str_err}"
                                )
                        logger.debug(
                            f"Fallback chunk: tool_name from fragile extraction: {tool_name}"
                        )

                    if tool_name:
                        tool_id = str(uuid.uuid4())
                        parsed_tool_call_chunks = [
                            ToolCallChunk(
                                name=tool_name,
                                args="{}",
                                id=tool_id,
                                index=0,
                            )
                        ]
                        empty_ai_message_chunk.tool_call_chunks = (
                            parsed_tool_call_chunks
                        )
                except Exception as e:
                    logger.warning(f"Failed to create fallback tool call chunk: {e}")

            # Yield the fallback chunk
            yield ChatGenerationChunk(
                message=empty_ai_message_chunk, generation_info=final_gen_info
            )
