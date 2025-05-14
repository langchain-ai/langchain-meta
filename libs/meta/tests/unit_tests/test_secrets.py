# Copyright (c) Meta Platforms, Inc. and affiliates
from langchain_meta import ChatLlama


def test_chat_llama_secrets() -> None:
    o = ChatLlama(model="Llama-4-Scout-17B-16E-Instruct-FP8", llama_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
