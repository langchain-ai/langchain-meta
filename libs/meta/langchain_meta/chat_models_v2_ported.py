# https://python.langchain.com/docs/how_to/custom_chat_model/

import logging
import os
import warnings
from collections.abc import AsyncIterator, Iterator, Sequence
from operator import itemgetter
from typing import (
    Any,
    Callable,
    ClassVar,
    Literal,
    Optional,
    Union,
)

from langchain_core.language_models.base import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.runnables import (
    Runnable,
    RunnableMap,
    RunnablePassthrough,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import (
    is_basemodel_subclass,
)
from llama_api_client import AsyncLlamaAPIClient, LlamaAPIClient
from llama_api_client.types.chat import (
    completion_create_params,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    SecretStr,
    ValidationInfo,
    field_validator,
)

# Import the mixin
from .chat_meta_llama.chat_async import AsyncChatMetaLlamaMixin
from .chat_meta_llama.chat_sync import SyncChatMetaLlamaMixin
from .chat_meta_llama.serialization import (
    _lc_message_to_llama_message_param,
)

logger = logging.getLogger(__name__)

# Valid models for the Llama API
VALID_MODELS = {
    "Llama-4-Scout-17B-16E-Instruct-FP8",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Llama-3.3-70B-Instruct",
    "Llama-3.3-8B-Instruct",
}

LLAMA_KNOWN_MODELS = {
    "Llama-3.3-70B-Instruct": {
        "model_name": "Llama-3.3-70B-Instruct",
    },
    "Llama-3.3-8B-Instruct": {
        "model_name": "Llama-3.3-8B-Instruct",
    },
    "Llama-4-Scout-17B-16E-Instruct-FP8": {  # Example, adjust as needed
        "model_name": "Llama-4-Scout-17B-16E-Instruct-FP8",
    },
    "Llama-4-Maverick-17B-128E-Instruct-FP8": {  # Example, adjust as needed
        "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
    },
}

LLAMA_DEFAULT_MODEL_NAME = "Llama-4-Maverick-17B-128E-Instruct-FP8"

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.outputs import ChatGenerationChunk


# Custom PydanticToolsParser that raises errors instead of returning None
class RaisingPydanticToolsParser(PydanticToolsParser):
    def _parse_tool_call_error(
        self,
        tool_call_name: str,
        tool_call_args: dict,
        err: Exception,
        schema: Any,
    ) -> Any:
        """Handle error when parsing tool call by raising the original error."""
        raise err


# STEEXZDdafsdfgasdfg
class ChatMetaLlama(SyncChatMetaLlamaMixin, AsyncChatMetaLlamaMixin, BaseChatModel):
    """LangChain ChatModel wrapper for the native Meta Llama API using llama-api-client.

    Key features:
    - Supports tool calling (model-driven, no tool_choice parameter).
    - Handles message history and tool execution results.
    - Provides streaming and asynchronous generation.
    - Fully compatible with LangSmith tracing and monitoring.

    Differences from OpenAI client:
    - No `tool_choice` parameter to force tool use.
    - Response structure is `response.completion_message` instead of `response.choices[0].message`.
    - `ToolCall` objects in the response do not have a direct `.type` attribute.

    To use, you need to have the `llama-api-client` Python package installed and
    configure your Meta Llama API key and base URL.

    Example:
        ```python
        from llama_api_client import LlamaAPIClient
        from langchain_meta.chat_models_v2_ported import ChatMetaLlama

        client = LlamaAPIClient(
            api_key=os.environ.get("META_API_KEY"),
            base_url=os.environ.get("META_API_BASE_URL", "https://api.llama.com/v1/")
        )
        llm = ChatMetaLlama(client=client, model_name="Llama-4-Maverick-17B-128E-Instruct-FP8")

        # Basic invocation
        response = llm.invoke([HumanMessage(content="Hello Llama!")])
        print(response.content)

        # Tool calling
        from langchain_core.tools import tool
        @tool
        def get_weather(location: str) -> str:
            '''Gets the current weather in a given location.'''
            return f"The weather in {location} is sunny."

        llm_with_tools = llm.bind_tools([get_weather])
        response = llm_with_tools.invoke("What is the weather in London?")
        print(response.tool_calls)
        ```

    LangSmith integration:
        To enable LangSmith tracing, set these environment variables:
        ```
        LANGSMITH_TRACING=true
        LANGSMITH_API_KEY="your-api-key"
        LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
        LANGSMITH_PROJECT="your-project-name"
        ```
    """

    _client: LlamaAPIClient | None = PrivateAttr(default=None)
    _async_client: AsyncLlamaAPIClient | None = PrivateAttr(default=None)

    # START_EDIT
    # Revert to Pydantic fields with aliases
    model_name: Optional[str] = Field(default=LLAMA_DEFAULT_MODEL_NAME, alias="model")
    temperature: Optional[float] = Field(default=None, alias="temperature")
    max_tokens: Optional[int] = Field(default=None, alias="max_completion_tokens")
    repetition_penalty: Optional[float] = Field(
        default=None, alias="repetition_penalty"
    )

    llama_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    llama_api_url: Optional[str] = Field(default=None, alias="base_url")
    # END_EDIT

    SUPPORTED_PARAMS: ClassVar[set] = {
        "model",  # API expects "model"
        "messages",
        "temperature",
        "max_completion_tokens",  # API expects "max_completion_tokens"
        "tools",
        "stream",
        "repetition_penalty",
        "top_p",
        "top_k",
        "user",
        "response_format",
    }

    lc_secrets: ClassVar[dict[str, str]] = {
        "llama_api_key": "LLAMA_API_KEY",  # Pydantic field name
    }

    lc_attributes: ClassVar[dict[str, Any]] = {}

    model_config = ConfigDict(
        validate_assignment=True,
        populate_by_name=True,  # V2 equivalent of allow_population_by_field_name
        extra="allow",
    )

    def __init__(
        self,
        *,  # Make all args keyword-only
        # __init__ params should match the Pydantic field names (pre-alias names)
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
        llama_api_key: Optional[str] = None,
        llama_api_url: Optional[str] = None,
        client: Optional[LlamaAPIClient] = None,
        async_client: Optional[AsyncLlamaAPIClient] = None,
        **kwargs: Any,
    ):
        # --- Temporary forceful logging setup ---
        # import sys
        #
        # temp_logger = logging.getLogger("ChatMetaLlama_INIT_DEBUG")
        # temp_logger.setLevel(logging.DEBUG)
        # if not temp_logger.hasHandlers():
        #     handler = logging.StreamHandler(sys.stdout)
        #     handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        #     temp_logger.addHandler(handler)
        # --- End temporary logging setup ---

        # START_EDIT
        # No need to map old names from kwargs if __init__ params are the old names.
        # kwargs might contain aliased names if called with them, Pydantic V1 with populate_by_name handles this.
        # However, env var lookup needs to be careful.
        _model_name_resolved = model_name
        _max_tokens_resolved = max_tokens
        _llama_api_key_resolved = (
            llama_api_key  # Will be converted to SecretStr later if string
        )
        _llama_api_url_resolved = llama_api_url  # Direct __init__ param

        # Handle direct kwargs that might be aliases for a more robust init from dicts
        if _model_name_resolved is None and "model" in kwargs:
            _model_name_resolved = kwargs.pop("model")
        if _max_tokens_resolved is None and "max_completion_tokens" in kwargs:
            _max_tokens_resolved = kwargs.pop("max_completion_tokens")
        if _llama_api_key_resolved is None and "api_key" in kwargs:
            _llama_api_key_resolved = kwargs.pop("api_key")
        if _llama_api_url_resolved is None and "base_url" in kwargs:
            _llama_api_url_resolved = kwargs.pop("base_url")

        # Environment variable lookup
        if _llama_api_key_resolved is None:
            # Prioritize LLAMA_API_KEY, then META_API_KEY
            key_from_llama_env = os.environ.get("LLAMA_API_KEY")
            key_from_meta_env = os.environ.get("META_API_KEY")
            if key_from_llama_env:
                _llama_api_key_resolved = key_from_llama_env
            elif key_from_meta_env:
                _llama_api_key_resolved = key_from_meta_env

        if _llama_api_url_resolved is None:
            url_from_llama_env = os.environ.get("LLAMA_API_URL")
            url_from_meta_env = os.environ.get("META_API_BASE_URL")
            if url_from_llama_env:
                _llama_api_url_resolved = url_from_llama_env
            elif url_from_meta_env:
                _llama_api_url_resolved = url_from_meta_env

        # Ensure model_name gets its default if still None.
        # Pydantic's Field(default=...) for model_name handles cases where it's not provided.
        # The validator for model_name also ensures a default if it becomes None post-init.
        # The _model_name_resolved logic here tries to pre-apply the default if no input was given at all.
        if (
            model_name is None
            and "model_name" not in kwargs
            and "model" not in kwargs
            and _model_name_resolved is None
        ):
            _model_name_resolved = LLAMA_DEFAULT_MODEL_NAME
        # temp_logger.debug(
        #     f"Pre-init_values: _llama_api_url_resolved='{_llama_api_url_resolved}', _llama_api_key_resolved type: {type(_llama_api_key_resolved)}"
        # )

        # Construct init_values using Pydantic field names as keys
        init_values: dict[str, Any] = {
            "model_name": _model_name_resolved,
            "temperature": temperature,  # Direct __init__ param
            "max_tokens": _max_tokens_resolved,
            "repetition_penalty": repetition_penalty,  # Direct __init__ param
            # Ensure SecretStr conversion for llama_api_key before super().__init__
            "llama_api_key": SecretStr(_llama_api_key_resolved)
            if isinstance(_llama_api_key_resolved, str)
            else _llama_api_key_resolved,
            "llama_api_url": _llama_api_url_resolved,
        }

        # Add any other kwargs that were passed in (e.g., client, async_client, top_p)
        # These might be Pydantic fields themselves if defined on the model, or go to __pydantic_extra__
        init_values.update(kwargs)

        # END_EDIT
        # init_values_filtered = {k: v for k, v in init_values.items() if v is not None}
        # The filtering of None values:
        # If a field has default=None, passing {key: None} is fine.
        # If a key is absent, Pydantic uses its default.
        # Filtering out None means if a user explicitly passes temperature=None,
        # it's treated as "not provided" rather than "explicitly None".
        # For API params, sometimes "not provided" means "use client/API default",
        # while "explicitly None" might be an error or mean something else.
        # The original filtering was likely intentional for this reason. Let's keep it.
        init_values_filtered = {k: v for k, v in init_values.items() if v is not None}

        # START_EDIT
        # The SecretStr conversion for 'llama_api_key' is now handled directly during init_values construction.
        # Old logic:
        # if "llama_api_key" in init_values_filtered and isinstance(init_values_filtered["llama_api_key"], str):
        #     init_values_filtered["llama_api_key"] = SecretStr(init_values_filtered["llama_api_key"])
        # END_EDIT

        super().__init__(**init_values_filtered)
        # temp_logger.debug(
        #     f"Post-super().__init__(): self.llama_api_url='{self.llama_api_url}', self.llama_api_key type: {type(self.llama_api_key)}"
        # )

        # START_EDIT: Explicitly set fields if Pydantic failed to from init_values_filtered
        if self.llama_api_url is None and _llama_api_url_resolved is not None:
            # temp_logger.debug(
            #     f"llama_api_url is None and _resolved is not. Overwriting self.llama_api_url='{_llama_api_url_resolved}'"
            # )
            self.llama_api_url = _llama_api_url_resolved

        _key_to_set = None
        if isinstance(_llama_api_key_resolved, str):
            _key_to_set = SecretStr(_llama_api_key_resolved)
        elif isinstance(_llama_api_key_resolved, SecretStr):
            _key_to_set = _llama_api_key_resolved

        if self.llama_api_key is None and _key_to_set is not None:
            self.llama_api_key = _key_to_set

        if client is not None:
            self._client = client
        if async_client is not None:
            self._async_client = async_client

        self._ensure_client_initialized()

    @field_validator("model_name", mode="before")  # Was model_name
    @classmethod
    def validate_model_name(  # Renamed for clarity, validates 'model' field
        cls, v: Any, info: ValidationInfo
    ):
        if v is None:
            # Allow None to pass through if the field is Optional and default=None applies
            # For model_name, Field has default=LLAMA_DEFAULT_MODEL_NAME, so it won't be None unless explicitly set.
            # If it were Field(default=None, ...) this would be critical.
            # Here, if v is None it means it was explicitly passed as None, overriding the Field default.
            # The Pydantic Field default (LLAMA_DEFAULT_MODEL_NAME) only applies if the key is missing or value is Undefined,
            # not if value is None.
            # So, if we get None here, we should probably still default it, or the __init__ logic
            # already defaulted _model_name_resolved if the input `model_name` param was None.
            # The current __init__ ensures _model_name_resolved gets LLAMA_DEFAULT_MODEL_NAME if initial param is None.
            # So v here should not be None if that logic holds.
            # However, if someone does chat_model.model_name = None, this validator runs.
            default_model = LLAMA_DEFAULT_MODEL_NAME
            logger.warning(f"model_name was None, defaulting to {default_model}")
            return default_model

        v_str = str(v).strip()
        if not v_str:
            default_model = LLAMA_DEFAULT_MODEL_NAME
            logger.warning(
                f"model (model_name) was empty. Defaulting to {default_model}"
            )
            return default_model

        if v_str not in LLAMA_KNOWN_MODELS:
            warnings.warn(
                f"Model '{v_str}' is not in the list of known Llama models.\n"
                f"Known models: {', '.join(LLAMA_KNOWN_MODELS.keys())}\n"
                "Your model may still work if the Meta API accepts it, but hasn't been tested."
            )
        return v_str

    @property
    def _llm_type(self) -> str:
        return "meta-llama"

    @property
    def client(self) -> LlamaAPIClient | None:
        return self._client

    @property
    def _identifying_params(self) -> dict[str, Any]:
        # START_EDIT
        # Uses new field names
        params = {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
        }
        if self.llama_api_url is not None:
            params["llama_api_url"] = self.llama_api_url
        return params

    def _ensure_client_initialized(self) -> None:
        # Uses new field names api_key (SecretStr) and base_url (str)
        key_val = self.llama_api_key.get_secret_value() if self.llama_api_key else None
        url_val = self.llama_api_url

        if self._client is None:
            if not key_val:
                logger.warning(
                    "LlamaAPIClient: API key is missing. Sync client cannot be initialized."
                )
            else:
                self._client = LlamaAPIClient(api_key=key_val, base_url=url_val)
        # ... similar for _async_client
        if self._async_client is None:
            if not key_val:
                logger.warning(
                    "AsyncLlamaAPIClient: API key is missing. Async client cannot be initialized."
                )
            else:
                self._async_client = AsyncLlamaAPIClient(
                    api_key=key_val, base_url=url_val
                )

    def _detect_supervisor_request(self, messages: list[BaseMessage]) -> bool:
        """Detect if this looks like a supervisor routing request.

        Examines the messages to see if they appear to be a supervisor routing request
        by checking for "route" and "next" keywords in system messages.
        """
        for msg in messages:
            if (
                isinstance(msg, SystemMessage)
                and isinstance(msg.content, str)
                and "route" in msg.content.lower()
                and "next" in msg.content.lower()
            ):
                logger.debug("Supervisor request detected in messages")
                return True
        return False

    def _prepare_api_params(
        self,
        messages: list[BaseMessage],
        tools: Optional[list[completion_create_params.Tool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Prepare parameters for the Llama API call.

        Args:
            messages: List of LangChain BaseMessages.
            tools: Optional list of Llama API formatted tools.
            stop: Optional list of stop sequences.
            stream: Boolean indicating if streaming is enabled.
            **kwargs: Additional keyword arguments for the API.

        Returns:
            A dictionary of parameters for the Llama API.
        """
        # Convert LangChain messages to Llama API message format
        # Pass the Llama-formatted tools (self.tools or tools from kwargs) to assist with argument coercion
        llama_messages = [
            _lc_message_to_llama_message_param(m, available_tools_with_schema=tools)
            for m in messages
        ]

        # Remove tool_choice from kwargs as it's not supported by Llama API directly
        # and would cause an error if passed to llama-api-client
        if "tool_choice" in kwargs:
            logger.debug(
                f"Llama API does not support 'tool_choice'. Removing '{kwargs['tool_choice']}' from API call."
            )
            kwargs.pop("tool_choice")

        # Construct the base parameters
        params: dict[str, Any] = {
            "model": self.model_name
            or LLAMA_DEFAULT_MODEL_NAME,  # Use alias-resolved name
            "messages": llama_messages,
        }

        # Add optional parameters from self or kwargs
        # Temperature
        if self.temperature is not None:
            params["temperature"] = self.temperature
        elif "temperature" in kwargs:
            params["temperature"] = kwargs["temperature"]

        # Max tokens (Llama API expects max_completion_tokens)
        max_tokens_val = kwargs.get(
            "max_tokens", kwargs.get("max_completion_tokens", self.max_tokens)
        )
        if max_tokens_val is not None:
            params["max_completion_tokens"] = max_tokens_val

        # Repetition Penalty
        repetition_penalty_val = kwargs.get(
            "repetition_penalty", self.repetition_penalty
        )
        if repetition_penalty_val is not None:
            params["repetition_penalty"] = repetition_penalty_val

        # Tools
        if tools:
            params["tools"] = tools

        # Streaming
        if stream:
            params["stream"] = True

        # Add any other supported kwargs directly
        for k, v in kwargs.items():
            if (
                k in self.SUPPORTED_PARAMS and k not in params
            ):  # Avoid overwriting already set params
                params[k] = v
            elif k not in self.SUPPORTED_PARAMS and k not in [
                "max_tokens",
                "tools",
            ]:
                logger.debug(f"Ignoring unsupported Llama API parameter: {k}")

        return params

    def _get_invocation_params(self, **kwargs: Any) -> dict[str, Any]:
        """Gets the parameters for a chat completion invocation."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "repetition_penalty": self.repetition_penalty,
            "top_p": kwargs.get("top_p", 1.0),
            "top_k": kwargs.get("top_k", 0),
        }

    def _count_tokens(self, messages: list[BaseMessage]) -> int:
        """Counts the number of tokens in a list of messages."""
        return sum(len(message.content) for message in messages)

    def _extract_content_from_response(self, response: Any) -> str:
        """Extracts content from a chat completion response."""
        if isinstance(response, dict) and "choices" in response:
            for choice in response["choices"]:
                if "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
        return ""

    def get_token_ids(self, text: str) -> list[int]:
        """Approximate token IDs using character length."""
        # This is a simple fallback. A more accurate method would use a proper tokenizer.
        # For basic testing and fallback, counting characters or simple splitting is sufficient.
        # We return a list of integers to match the expected return type.
        return [
            ord(c) for c in text
        ]  # Using ASCII values as a placeholder for token IDs

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in a text string.

        Uses character count as a simple approximation.
        """
        return len(self.get_token_ids(text))

    def bind_tools(
        self,
        tools: Sequence[Union[dict[str, Any], type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[str, dict, Literal["any", "auto"]]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:  # MODIFIED (removed quotes)
        """Bind tool-like objects to this chat model.

        Args:
            tools: A list of tools to bind to the model.
            tool_choice: Optional tool choice.
            **kwargs: Aditional keyword arguments.

        Returns:
            A new Runnable with the tools bound.
        """
        # Correctly delegate to the model's own .bind() method,
        # passing the tools under the 'tools' keyword.
        logger.debug(
            f"ChatMetaLlama.bind_tools called with tools: {[getattr(t, 'name', t) for t in tools]}, tool_choice: {tool_choice}, and kwargs: {kwargs}"
        )
        return self.bind(tools=tools, tool_choice=tool_choice, **kwargs)

    def with_structured_output(
        self,
        schema: Any,
        *,
        method: str = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **other_kwargs: Any,
    ) -> Runnable:
        if other_kwargs:
            raise ValueError(f"Received unsupported arguments {other_kwargs}")

        # Prepare the special kwargs for bind_tools that test callbacks expect
        bind_kwargs_for_structured_output = {
            "ls_structured_output_format": {
                "kwargs": {"method": method, "strict": strict},
                "schema": schema,
            }
        }

        if method == "function_calling":
            # tool_choice="any" is a hint, Llama is model-driven.
            # Binding the single schema makes it the only choice if the model uses tools.
            llm = self.bind_tools([schema], **bind_kwargs_for_structured_output)
            if is_basemodel_subclass(schema):
                parser: Any = PydanticToolsParser(tools=[schema], first_tool_only=True)
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        elif method == "json_mode":
            warnings.warn(
                "JSON mode for ChatMetaLlama's with_structured_output might not be fully "
                "implemented to Llama API's native JSON mode yet. "
                "Falling back to function_calling behavior for tool binding."
            )
            # For Llama, json_mode might mean setting response_format directly.
            # However, to use existing parsers that expect tool calls, we still bind the schema as a tool.
            # The actual API call in _prepare_api_params would need to set response_format for true json_mode.
            llm = self.bind_tools([schema], **bind_kwargs_for_structured_output)
            if is_basemodel_subclass(schema):
                parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        elif method == "json_schema":
            warnings.warn(
                "json_schema method for with_structured_output is not standardly "
                "supported by Llama API in the same way as OpenAI. "
                "Falling back to function_calling behavior for tool binding."
            )
            llm = self.bind_tools([schema], **bind_kwargs_for_structured_output)
            if is_basemodel_subclass(schema):
                parser = PydanticToolsParser(tools=[schema], first_tool_only=True)
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        else:
            raise ValueError(
                f"Unsupported method {method}. Must be one of 'function_calling', 'json_mode', 'json_schema'."
            )

        if include_raw:
            parser_with_raw: Runnable = RunnablePassthrough.assign(
                parsed=itemgetter("parsed") | parser, raw=itemgetter("raw")
            )
            return (
                {"input": RunnablePassthrough()}
                | RunnableMap(raw=llm, parsed=llm)
                | parser_with_raw
            )
        else:
            return llm | parser

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream chat messages, returning an iterator of ChatGenerationChunks."""
        # START_EDIT
        # Explicitly call the mixin's method
        return SyncChatMetaLlamaMixin._stream(
            self, messages=messages, stop=stop, run_manager=run_manager, **kwargs
        )
        # END_EDIT

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously stream chat messages, returning an async iterator of ChatGenerationChunks."""
        # START_EDIT
        # Explicitly iterate and yield from the mixin's async generator.
        # This ensures that ChatMetaLlama._astream itself is unequivocally an async generator.
        async for chunk in AsyncChatMetaLlamaMixin._astream(
            self, messages=messages, stop=stop, run_manager=run_manager, **kwargs
        ):
            yield chunk
        # END_EDIT

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain standard methods."""
        return True
