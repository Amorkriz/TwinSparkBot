"""Unit tests for :mod:`twinspark.core.llm` using mocks (no real API calls)."""

from __future__ import annotations

import os

# Provide a fake key before importing config-dependent code.
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

from types import SimpleNamespace
from typing import Any

import httpx
import openai
import pytest

from twinspark.core.llm import LLMClient, is_retryable_error


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
def _make_completion(content: str) -> SimpleNamespace:
    """Build an object shaped like a non-streaming chat completion."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


def _make_chunk(text: str | None) -> SimpleNamespace:
    """Build an object shaped like a streaming chat completion chunk."""
    delta = SimpleNamespace(content=text)
    choice = SimpleNamespace(delta=delta)
    return SimpleNamespace(choices=[choice])


class _FakeAsyncStream:
    """Async iterator yielding pre-baked streaming chunks."""

    def __init__(self, texts: list[str | None]) -> None:
        self._chunks = [_make_chunk(t) for t in texts]

    def __aiter__(self) -> "_FakeAsyncStream":
        self._it = iter(self._chunks)
        return self

    async def __anext__(self) -> SimpleNamespace:
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class _FakeCompletions:
    """Stand-in for ``client.chat.completions`` with scripted behavior."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._result: Any = None
        self._error_sequence: list[Exception] = []

    def set_result(self, result: Any) -> None:
        self._result = result

    def set_error_sequence(self, errors: list[Exception], final: Any) -> None:
        """Raise each error in order, then return ``final``."""
        self._error_sequence = list(errors)
        self._result = final

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self._error_sequence:
            raise self._error_sequence.pop(0)
        result = self._result
        return result() if callable(result) else result


class _FakeAsyncOpenAI:
    """Minimal fake exposing ``.chat.completions`` and ``.close``."""

    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.chat = SimpleNamespace(completions=self.completions)
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def _status_error(status: int) -> openai.APIStatusError:
    """Construct an APIStatusError with a given HTTP status code."""
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    response = httpx.Response(status, request=request)
    return openai.APIStatusError("boom", response=response, body=None)


# --------------------------------------------------------------------------- #
# is_retryable_error                                                           #
# --------------------------------------------------------------------------- #
def test_is_retryable_error_classification() -> None:
    assert is_retryable_error(_status_error(429)) is True
    assert is_retryable_error(_status_error(500)) is True
    assert is_retryable_error(_status_error(503)) is True
    assert is_retryable_error(_status_error(400)) is False
    assert is_retryable_error(_status_error(401)) is False
    assert is_retryable_error(_status_error(404)) is False
    assert is_retryable_error(httpx.ConnectTimeout("t")) is True
    assert is_retryable_error(ValueError("nope")) is False


# --------------------------------------------------------------------------- #
# chat                                                                         #
# --------------------------------------------------------------------------- #
async def test_chat_returns_full_text() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_result(_make_completion("Hello, world!"))
    llm = LLMClient(client=fake, model="qwen-plus")

    reply = await llm.chat([{"role": "user", "content": "hi"}])

    assert reply == "Hello, world!"
    assert fake.completions.calls[0]["stream"] is False
    assert fake.completions.calls[0]["model"] == "qwen-plus"


async def test_chat_minimal_params_by_default() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_result(_make_completion("ok"))
    llm = LLMClient(client=fake)

    await llm.chat([{"role": "user", "content": "hi"}])

    params = fake.completions.calls[0]
    assert set(params.keys()) == {"model", "messages", "stream"}


async def test_chat_forwards_kwargs() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_result(_make_completion("ok"))
    llm = LLMClient(client=fake)

    await llm.chat(
        [{"role": "user", "content": "hi"}], model="qwen-max", temperature=0.3
    )

    params = fake.completions.calls[0]
    assert params["model"] == "qwen-max"
    assert params["temperature"] == 0.3


async def test_chat_retries_then_succeeds() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_error_sequence(
        [_status_error(429), _status_error(503)],
        _make_completion("recovered"),
    )
    llm = LLMClient(client=fake, max_retries=3)

    reply = await llm.chat([{"role": "user", "content": "hi"}])

    assert reply == "recovered"
    assert len(fake.completions.calls) == 3


async def test_chat_raises_on_client_error_without_retry() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_error_sequence([_status_error(400)], _make_completion("x"))
    llm = LLMClient(client=fake, max_retries=3)

    with pytest.raises(openai.APIStatusError):
        await llm.chat([{"role": "user", "content": "hi"}])

    assert len(fake.completions.calls) == 1  # no retry on 4xx


# --------------------------------------------------------------------------- #
# stream                                                                       #
# --------------------------------------------------------------------------- #
async def test_stream_yields_chunks() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_result(_FakeAsyncStream(["Hel", "lo", None, "!"]))
    llm = LLMClient(client=fake)

    collected = [
        piece async for piece in llm.stream([{"role": "user", "content": "hi"}])
    ]

    assert collected == ["Hel", "lo", "!"]
    assert "".join(collected) == "Hello!"
    assert fake.completions.calls[0]["stream"] is True


async def test_stream_retries_before_first_chunk() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_error_sequence(
        [_status_error(429)],
        lambda: _FakeAsyncStream(["done"]),
    )
    llm = LLMClient(client=fake, max_retries=3)

    collected = [
        piece async for piece in llm.stream([{"role": "user", "content": "hi"}])
    ]

    assert collected == ["done"]
    assert len(fake.completions.calls) == 2


# --------------------------------------------------------------------------- #
# lifecycle                                                                    #
# --------------------------------------------------------------------------- #
async def test_context_manager_closes_client() -> None:
    fake = _FakeAsyncOpenAI()
    fake.completions.set_result(_make_completion("hi"))

    async with LLMClient(client=fake) as llm:
        await llm.chat([{"role": "user", "content": "hi"}])

    assert fake.closed is True
