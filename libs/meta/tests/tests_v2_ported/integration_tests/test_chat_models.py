# Standard library imports
import logging
import os
from typing import Type

from langchain_tests.integration_tests.chat_models import ChatModelIntegrationTests

from langchain_meta.chat_models_v2_ported import ChatMetaLlama
import pytest
from langchain_core.language_models import BaseChatModel

logging.basicConfig(level=logging.DEBUG)
logging.getLogger("langchain_meta.chat_meta_llama.chat_sync").setLevel(logging.DEBUG)
logging.getLogger("langchain_meta.chat_models").setLevel(logging.DEBUG)


class TestChatMetaLlamaIntegration(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[ChatMetaLlama]:
        return ChatMetaLlama

    @property
    def chat_model_params(self) -> dict:
        return {
            "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "temperature": 0,
            "llama_api_url": "https://api.llama.com/v1/",
            "llama_api_key": os.getenv("LLAMA_API_KEY"),
        }

    @property
    def supports_json_mode(self) -> bool:
        """Indicates that ChatMetaLlama supports JSON mode via response_format."""
        return True

    @property
    def supports_structured_output(self) -> bool:
        """Indicates that ChatMetaLlama supports structured output."""
        return True

    @property
    def has_tool_choice(self) -> bool:
        """Indicates whether the model supports forced tool calling via tool_choice.

        The Llama API does not support forced tool calling in the same way as OpenAI.
        It uses a model-driven approach for tool choice.
        """
        return False

    @pytest.mark.xfail(
        reason=(
            "Llama model exercises discretion in tool use with 'function_calling' "
            "method of with_structured_output when tool_choice cannot be API-enforced. "
            "The specific prompt in this test ('no punchline') leads the model to not call the tool, "
            "resulting in a None output where a structured object is expected."
        )
    )
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)
