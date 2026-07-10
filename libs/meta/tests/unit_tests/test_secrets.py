# Copyright (c) Meta Platforms, Inc. and affiliates
from langchain_meta import ChatLlama, ChatMetaModel


def test_chat_llama_secrets() -> None:
    o = ChatLlama(model="Llama-4-Scout-17B-16E-Instruct-FP8", llama_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s


def test_chat_meta_model_secrets() -> None:
    o = ChatMetaModel(model="muse-spark-1.1", meta_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
