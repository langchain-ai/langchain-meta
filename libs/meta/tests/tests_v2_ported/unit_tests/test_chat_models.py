"""Test chat model integration."""

import os
from typing import Any
from unittest import mock

import pytest
from langchain_tests.unit_tests import ChatModelUnitTests
from pydantic import SecretStr

from langchain_meta.chat_models_v2_ported import ChatMetaLlama


class TestChatMetaLlamaUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatMetaLlama]:
        return ChatMetaLlama

    @property
    def chat_model_params(self) -> dict:
        # These should be parameters used to initialize your integration for testing
        return {
            "model_name": "Llama-4-Maverick-17B-128E-Instruct-FP8",
            "temperature": 0,
            "llama_api_url": "https://fake-chat-model-params.llama.com/v1/",
            "llama_api_key": "fake_llama_api_key",
        }

    @property
    def init_from_env_params(
        self,
    ) -> tuple[dict[str, str], dict[str, Any], dict[str, Any]]:
        """Provides parameters for testing initialization from environment variables."""
        return (
            {
                "LLAMA_API_KEY": "fake_env_api_key",
                "LLAMA_API_URL": "",
                "META_API_BASE_URL": "https://fake-env-api.llama.com/v1/",
            },
            {  # Additional model init kwargs
                "model_name": "Llama-3.3-8B-Instruct",
            },
            {  # Expected attributes on the instantiated model
                "llama_api_key": "fake_env_api_key",
                "model_name": "Llama-3.3-8B-Instruct",
                "llama_api_url": "https://fake-env-api.llama.com/v1/",
            },
        )

    @pytest.mark.xfail(
        reason=(
            "Custom SecretStr handling and init logic for Llama API environment setup."
        )
    )
    def test_init_from_env(self) -> None:
        """Test initialization from environment variables, handling SecretStr."""
        if not hasattr(self, "init_from_env_params"):  # Check if the property exists
            raise AttributeError(
                "init_from_env_params is not defined in the test class."
            )

        env_vars, init_kwargs, expected_attrs = self.init_from_env_params

        ChatModelClass = self.chat_model_class  # Get the model class via property

        with mock.patch.dict(os.environ, env_vars):
            chat = ChatModelClass(**init_kwargs)
            for key, expected_value in expected_attrs.items():
                actual_value = getattr(chat, key)
                if isinstance(actual_value, SecretStr):
                    assert actual_value.get_secret_value() == expected_value, (
                        f"Mismatch for SecretStr attribute '{key}'. "
                        f"Expected: '{expected_value}', Got SecretStr"
                    )
                else:
                    assert actual_value == expected_value, (
                        f"Mismatch for attribute '{key}'. "
                        f"Expected: '{expected_value}', Got: '{actual_value}'"
                    )
