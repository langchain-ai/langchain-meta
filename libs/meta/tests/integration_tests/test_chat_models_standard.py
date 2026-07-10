# Copyright (c) Meta Platforms, Inc. and affiliates
"""Standard LangChain interface tests"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import (  # type: ignore[import-not-found]
    ChatModelIntegrationTests,  # type: ignore[import-not-found]
)

from langchain_meta import ChatMetaModel

MODEL_NAME = "muse-spark-1.1"


class TestMetaModelStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatMetaModel

    @property
    def chat_model_params(self) -> dict:
        return {"model": MODEL_NAME}

    @property
    def has_tool_choice(self) -> bool:
        # Meta only supports ``tool_choice="auto"``; forcing a specific tool or
        # ``"required"`` returns HTTP 400.
        return False

    @property
    def supports_image_inputs(self) -> bool:
        return True

    @property
    def supports_pdf_inputs(self) -> bool:
        return True

    @pytest.mark.xfail(
        reason=(
            'Test forces `tool_choice="any"`, but Meta only supports '
            '`tool_choice="auto"`.'
        )
    )
    def test_structured_few_shot_examples(self, *args: Any) -> None:
        super().test_structured_few_shot_examples(*args)
