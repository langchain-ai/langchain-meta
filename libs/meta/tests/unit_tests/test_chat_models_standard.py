# Copyright (c) Meta Platforms, Inc. and affiliates
"""Standard LangChain interface tests"""

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import (  # type: ignore[import-not-found]
    ChatModelUnitTests,
)

from langchain_meta import ChatMetaModel

MODEL_NAME = "muse-spark-1.1"


class TestMetaModelStandard(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[BaseChatModel]:
        return ChatMetaModel

    @property
    def chat_model_params(self) -> dict:
        # Supply a dummy key so tests that construct from bare ``chat_model_params``
        # (e.g. ``test_init_time``) don't require ``MODEL_API_KEY`` in the env.
        return {"model": MODEL_NAME, "api_key": "test"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {
                "MODEL_API_KEY": "api_key",
            },
            {
                "model": MODEL_NAME,
            },
            {
                "meta_api_key": "api_key",
                "meta_api_base": "https://api.meta.ai/v1/",
            },
        )

    @pytest.mark.xfail(
        reason="langchain_meta is not yet in langchain-core's trusted "
        "serialization allowlist (allowed_objects='all')",
    )
    def test_serdes(self, model: BaseChatModel, snapshot: Any) -> None:
        super().test_serdes(model, snapshot)
