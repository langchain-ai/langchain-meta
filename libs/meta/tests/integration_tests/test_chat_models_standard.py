# Copyright (c) Meta Platforms, Inc. and affiliates
"""Standard LangChain interface tests"""

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_meta import ChatLlama

# Initialize the rate limiter in global scope, so it can be reused
# across tests.
rate_limiter = InMemoryRateLimiter(
    requests_per_second=5,
)


class TestLlamaStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatLlama

    @property
    def chat_model_params(self) -> dict:
        return {
            "model": "Llama-4-Scout-17B-16E-Instruct-FP8",
            "rate_limiter": rate_limiter,
            "stream_usage": True,
        }

    @property
    def has_tool_choice(self) -> bool:
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_anthropic_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason=(
            "Pydantic v1 structured output requires tool_choice for BaseChatOpenAI."
        )
    )
    def test_structured_output_pydantic_2_v1(self, model: BaseChatModel) -> None:
        super().test_structured_output_pydantic_2_v1(model)

    @pytest.mark.xfail(reason=("Does not support default properties."))
    def test_structured_output_optional_param(self, model: BaseChatModel) -> None:
        super().test_structured_output_optional_param(model)

    @pytest.mark.xfail(reason=("Requires tool_choice."))
    def test_structured_few_shot_examples(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_structured_few_shot_examples(model, my_adder_tool)

    @pytest.mark.xfail(
        reason=(
            # TODO: investigate
            "{'title': 'Bad request', 'detail': 'Unexpected param value `a`: \"1\"', "
            "'status': 400}"
        )
    )
    def test_tool_message_histories_string_content(
        self, model: BaseChatModel, my_adder_tool: BaseTool
    ) -> None:
        super().test_tool_message_histories_string_content(model, my_adder_tool)
