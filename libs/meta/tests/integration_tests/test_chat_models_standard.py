# Copyright (c) Meta Platforms, Inc. and affiliates
"""Standard LangChain interface tests"""

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
