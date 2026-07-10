# Copyright (c) Meta Platforms, Inc. and affiliates
"""Wrapper around Meta's Model API (OpenAI-compatible)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast

import openai
from langchain_core.utils import from_env, secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_meta.data._profiles import _PROFILES

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.language_models import ModelProfile, ModelProfileRegistry
    from langchain_core.language_models.chat_models import (
        LangSmithParams,
        LanguageModelInput,
    )
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.runnables import Runnable
    from pydantic import BaseModel

_DEFAULT_API_BASE = "https://api.meta.ai/v1/"

_MODEL_PROFILES = cast("ModelProfileRegistry", _PROFILES)


def _get_default_model_profile(model_name: str) -> ModelProfile:
    default = _MODEL_PROFILES.get(model_name) or {}
    return default.copy()


class ChatMetaModel(BaseChatOpenAI):  # type: ignore[override]
    r"""ChatMetaModel chat model.

    Wrapper around [Meta's Model API](https://dev.meta.ai/docs/getting-started/overview/),
    which is OpenAI-compatible. By default it uses the Responses API.

    Setup:
        Install ``langchain-meta`` and set environment variable ``MODEL_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-meta
            export MODEL_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of model to use (e.g. ``"muse-spark-1.1"``).
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client params:
        timeout: Union[float, Tuple[float, float], Any, None]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Meta API key. If not passed in will be read from env var ``MODEL_API_KEY``.

    Instantiate:
        .. code-block:: python

            from langchain_meta import ChatMetaModel

            llm = ChatMetaModel(
                model="muse-spark-1.1",
                temperature=0,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)
    """

    model_name: str = Field(alias="model")
    """Model name to use."""

    meta_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env("MODEL_API_KEY", default=None),
    )
    """Meta API key.

    Automatically read from env variable ``MODEL_API_KEY`` if not provided.
    """

    meta_api_base: str = Field(
        alias="base_url",
        default_factory=from_env("MODEL_API_BASE", default=_DEFAULT_API_BASE),
    )
    """Base URL path for API requests.

    Automatically read from env variable ``MODEL_API_BASE`` if not provided.
    """

    openai_api_key: SecretStr | None = None

    openai_api_base: str | None = None

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example, ``{"meta_api_key": "MODEL_API_KEY"}``
        """
        return {"meta_api_key": "MODEL_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_meta", "chat_models"]

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "meta-model-chat"

    def _get_ls_params(
        self, stop: list[str] | None = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "meta"
        return params

    def _resolve_model_profile(self) -> ModelProfile | None:
        return _get_default_model_profile(self.model_name) or None

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        client_params: dict = {
            "api_key": (
                self.meta_api_key.get_secret_value() if self.meta_api_key else None
            ),
            "base_url": self.meta_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            raise ValueError(
                "Meta API key is not set. Please set it in the `meta_api_key` field "
                "or in the `MODEL_API_KEY` environment variable."
            )

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
            self.client = self.root_client.chat.completions
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
            self.async_client = self.root_async_client.chat.completions

        # Default to the Responses API, which exposes the full agentic feature set.
        if self.use_responses_api is None:
            self.use_responses_api = True

        # Enable streaming usage metadata by default.
        if self.stream_usage is not False:
            self.stream_usage = True

        return self

    def _stream(self, *args: Any, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API.

        ``BaseChatOpenAI`` only routes non-streaming calls to the Responses API,
        so we route streaming calls here to honor ``use_responses_api``.
        """
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            return super()._stream_responses(*args, **kwargs)
        return super()._stream(*args, **kwargs)

    async def _astream(
        self, *args: Any, **kwargs: Any
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Route to Chat Completions or Responses API."""
        if self._use_responses_api({**kwargs, **self.model_kwargs}):
            async for chunk in super()._astream_responses(*args, **kwargs):
                yield chunk
        else:
            async for chunk in super()._astream(*args, **kwargs):
                yield chunk

    def with_structured_output(
        self,
        schema: dict | type | None = None,
        *,
        method: Literal["function_calling", "json_mode", "json_schema"] = "json_schema",
        include_raw: bool = False,
        strict: bool | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Defaults to ``method="json_schema"``. Meta only supports
        ``tool_choice="auto"``, so the default ``"function_calling"`` method (which
        forces a named ``tool_choice``) is rejected by the API; ``"json_schema"``
        uses ``response_format`` instead and is the supported path.
        """
        return super().with_structured_output(
            schema,
            method=method,
            include_raw=include_raw,
            strict=strict,
            **kwargs,
        )
