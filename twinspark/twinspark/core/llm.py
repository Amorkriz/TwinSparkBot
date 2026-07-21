"""Async LLM client for TwinSpark.

Thin wrapper around the OpenAI-compatible DashScope (Bailian) endpoint that
exposes two ergonomic coroutines:

* :meth:`LLMClient.chat` -- non-streaming, returns the full reply text.
* :meth:`LLMClient.stream` -- streaming, yields text deltas as they arrive.

A single ``openai.AsyncOpenAI`` instance is reused for the lifetime of the
client so the underlying ``httpx`` connection pool is shared across calls.

Design notes
------------
* Bailian's OpenAI-compatible mode only reliably supports the minimal request
  shape (``model`` + ``messages`` + ``stream``). Optional sampling parameters
  (``temperature``, ``top_p``, ``max_tokens`` ...) are only forwarded when the
  caller passes them explicitly via ``**kwargs``; we never inject fields such
  as ``stream_options`` that may be rejected by the backend.
* Transient failures (HTTP 429, 5xx, connection errors, timeouts) are retried
  with bounded exponential backoff. Non-retryable client errors (4xx other
  than 429) are raised immediately.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import AsyncIterator
from typing import Any

import httpx
import openai
from openai import AsyncOpenAI

from twinspark.config import get_config

__all__ = ["LLMClient", "is_retryable_error"]

logger = logging.getLogger(__name__)

# --- Timeout defaults --------------------------------------------------------
# Long read timeout supports lengthy streaming responses; the connect timeout
# stays short so genuine connection problems fail fast into the retry loop.
DEFAULT_CONNECT_TIMEOUT = 10.0
DEFAULT_READ_TIMEOUT = 300.0
DEFAULT_WRITE_TIMEOUT = 30.0
DEFAULT_POOL_TIMEOUT = 10.0

# --- Retry defaults ----------------------------------------------------------
DEFAULT_MAX_RETRIES = 3
DEFAULT_BACKOFF_BASE = 0.5  # seconds
DEFAULT_BACKOFF_CAP = 20.0  # seconds


def is_retryable_error(exc: BaseException) -> bool:
    """Classify whether an exception is worth retrying.

    Retryable:
        * HTTP 429 (rate limit) and 5xx (server) responses.
        * Connection / timeout errors (network transport level).

    Non-retryable:
        * HTTP 4xx client errors other than 429 (bad request, auth, etc.).
        * Anything else not recognized as transient.

    Args:
        exc: The exception raised by an OpenAI SDK call.

    Returns:
        ``True`` if the caller should retry after backoff, else ``False``.
    """
    # Network transport failures (no HTTP status): connection reset, DNS,
    # read timeouts, etc. These are always safe to retry.
    if isinstance(exc, (openai.APIConnectionError, openai.APITimeoutError)):
        return True
    if isinstance(exc, (httpx.TimeoutException, httpx.TransportError)):
        return True

    # Errors carrying an HTTP status code.
    if isinstance(exc, openai.APIStatusError):
        status = exc.status_code
        if status == 429:
            return True
        if 500 <= status < 600:
            return True
        return False

    # RateLimitError subclasses APIStatusError, but guard explicitly in case a
    # provider surfaces it without a status code.
    if isinstance(exc, openai.RateLimitError):
        return True

    return False


class LLMClient:
    """Async client for the DashScope OpenAI-compatible chat endpoint.

    The client owns a single :class:`openai.AsyncOpenAI` instance (and its
    connection pool). Create one instance and reuse it; call :meth:`aclose`
    (or use it as an async context manager) to release resources.

    Example:
        >>> async with LLMClient() as llm:
        ...     reply = await llm.chat([{"role": "user", "content": "hi"}])
        ...     async for delta in llm.stream([{"role": "user", "content": "hi"}]):
        ...         print(delta, end="")
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        connect_timeout: float = DEFAULT_CONNECT_TIMEOUT,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        write_timeout: float = DEFAULT_WRITE_TIMEOUT,
        pool_timeout: float = DEFAULT_POOL_TIMEOUT,
        client: AsyncOpenAI | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            api_key: DashScope API key. Defaults to ``config.dashscope_api_key``.
            base_url: OpenAI-compatible endpoint. Defaults to ``config.base_url``.
            model: Default model slug. Defaults to ``config.model``.
            max_retries: Max retry attempts for transient failures.
            connect_timeout: Socket connect timeout in seconds.
            read_timeout: Read timeout in seconds (widened for streaming).
            write_timeout: Write timeout in seconds.
            pool_timeout: Connection-pool acquisition timeout in seconds.
            client: Pre-built ``AsyncOpenAI`` instance (mainly for testing).
                When provided, the timeout/key/url arguments are ignored.
        """
        cfg = get_config()
        self.model = model or cfg.model
        self.max_retries = max(0, int(max_retries))

        if client is not None:
            self._client = client
        else:
            # Disable the SDK's built-in retries; we manage retries ourselves so
            # backoff/classification stay consistent across chat and stream.
            timeout = httpx.Timeout(
                read_timeout,
                connect=connect_timeout,
                write=write_timeout,
                pool=pool_timeout,
            )
            self._client = AsyncOpenAI(
                api_key=api_key or cfg.dashscope_api_key,
                base_url=base_url or cfg.base_url,
                timeout=timeout,
                max_retries=0,
            )

    # -- lifecycle ------------------------------------------------------------
    async def aclose(self) -> None:
        """Close the underlying HTTP client and release the connection pool."""
        await self._client.close()

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    # -- internals ------------------------------------------------------------
    def _build_params(
        self,
        messages: list[dict],
        *,
        model: str | None,
        stream: bool,
        extra: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble request kwargs, keeping the base payload minimal.

        Only ``model``, ``messages`` and ``stream`` are always present. Any
        additional sampling parameters are taken verbatim from ``extra`` so we
        never send fields the Bailian backend might reject.
        """
        params: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": stream,
        }
        # Forward caller-supplied options as-is (temperature, top_p, ...).
        params.update(extra)
        return params

    def _backoff_delay(self, attempt: int) -> float:
        """Compute exponential backoff with full jitter for ``attempt`` (0-based)."""
        raw = DEFAULT_BACKOFF_BASE * (2 ** attempt)
        capped = min(raw, DEFAULT_BACKOFF_CAP)
        return random.uniform(0.0, capped)

    async def _sleep_before_retry(self, attempt: int, exc: BaseException) -> None:
        """Log and sleep before the next retry attempt."""
        delay = self._backoff_delay(attempt)
        logger.warning(
            "LLM call failed (attempt %d/%d): %s -- retrying in %.2fs",
            attempt + 1,
            self.max_retries + 1,
            exc,
            delay,
        )
        await asyncio.sleep(delay)

    # -- public API -----------------------------------------------------------
    async def chat(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Return the full assistant reply for ``messages`` (non-streaming).

        Args:
            messages: OpenAI-style chat messages (``[{"role", "content"}, ...]``).
            model: Override the default model for this call.
            **kwargs: Extra request parameters (e.g. ``temperature``) forwarded
                verbatim to the completions API.

        Returns:
            The concatenated assistant message content (empty string if none).

        Raises:
            openai.APIStatusError: For non-retryable client errors (4xx != 429).
            openai.APIError: If all retry attempts are exhausted.
        """
        params = self._build_params(
            messages, model=model, stream=False, extra=kwargs
        )

        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = await self._client.chat.completions.create(**params)
                choices = completion.choices
                if not choices:
                    return ""
                content = choices[0].message.content
                return content or ""
            except Exception as exc:  # noqa: BLE001 - reclassified below
                last_exc = exc
                if attempt < self.max_retries and is_retryable_error(exc):
                    await self._sleep_before_retry(attempt, exc)
                    continue
                raise

        # Unreachable in practice; keeps type-checkers satisfied.
        assert last_exc is not None
        raise last_exc

    async def chat_raw(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """返回完整的 assistant 消息对象（含 tool_calls）。

        与 chat() 采用相同的重试逻辑，但返回原始 Message 对象
        而非仅文本内容，供 Agent 工具循环使用。

        Args:
            messages: OpenAI 格式的聊天消息列表。
            model: 覆盖默认模型。
            **kwargs: 额外请求参数（如 tools, temperature）。

        Returns:
            OpenAI ChatCompletionMessage 对象，包含 content 和 tool_calls 字段；
            若响应为空则返回 None。

        Raises:
            openai.APIStatusError: 不可重试的客户端错误。
            openai.APIError: 所有重试耗尽后的 API 错误。
        """
        params = self._build_params(
            messages, model=model, stream=False, extra=kwargs
        )

        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                completion = await self._client.chat.completions.create(**params)
                choices = completion.choices
                if not choices:
                    return None
                return choices[0].message
            except Exception as exc:  # noqa: BLE001 - reclassified below
                last_exc = exc
                if attempt < self.max_retries and is_retryable_error(exc):
                    await self._sleep_before_retry(attempt, exc)
                    continue
                raise

        # 不可达；满足类型检查器
        assert last_exc is not None
        raise last_exc

    async def stream(
        self,
        messages: list[dict],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Yield assistant text deltas for ``messages`` as they stream in.

        Retries (with backoff) only cover establishing the stream. Once the
        first chunk has been yielded a mid-stream failure is surfaced to the
        caller rather than silently restarted (which could duplicate output).

        Args:
            messages: OpenAI-style chat messages.
            model: Override the default model for this call.
            **kwargs: Extra request parameters forwarded verbatim.

        Yields:
            Non-empty text increments extracted from ``choice.delta.content``.

        Raises:
            openai.APIStatusError: For non-retryable client errors (4xx != 429).
            openai.APIError: If establishing the stream fails after all retries.
        """
        params = self._build_params(
            messages, model=model, stream=True, extra=kwargs
        )

        for attempt in range(self.max_retries + 1):
            started = False
            try:
                stream_obj = await self._client.chat.completions.create(**params)
                async for chunk in stream_obj:
                    choices = getattr(chunk, "choices", None)
                    if not choices:
                        continue
                    delta = choices[0].delta
                    text = getattr(delta, "content", None)
                    if text:
                        started = True
                        yield text
                return
            except Exception as exc:  # noqa: BLE001 - reclassified below
                # Never retry once partial output has been emitted.
                if (
                    not started
                    and attempt < self.max_retries
                    and is_retryable_error(exc)
                ):
                    await self._sleep_before_retry(attempt, exc)
                    continue
                raise
