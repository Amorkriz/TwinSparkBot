"""DingTalk AI Card streaming support for TwinSpark.

Provides three core components:

1. :class:`DingTalkTokenManager` – manages access_token lifecycle with
   automatic refresh and in-memory caching.
2. :class:`DingTalkCardClient` – creates, streams, and finalizes AI Card
   instances via the DingTalk Open API.
3. :func:`stream_to_card` – bridges an async token stream to a live card,
   flushing accumulated content at a configurable interval.

Only ``httpx`` (already a project dependency) is used for outbound HTTP;
no additional external packages are required.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import AsyncIterator

import httpx

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token management
# ---------------------------------------------------------------------------


class DingTalkTokenManager:
    """Manage DingTalk access_token with in-memory caching.

    Tokens are refreshed proactively 5 minutes (300 s) before expiry to avoid
    edge-case failures at the boundary.
    """

    _TOKEN_URL = "https://api.dingtalk.com/v1.0/oauth2/accessToken"
    _REFRESH_MARGIN_S = 300  # 提前 5 分钟刷新

    def __init__(self, app_key: str, app_secret: str) -> None:
        self._app_key = app_key
        self._app_secret = app_secret
        self._access_token: str = ""
        self._expire_at: float = 0.0  # unix timestamp
        self._lock = asyncio.Lock()

    def _is_token_valid(self) -> bool:
        """Check if the cached token is still usable."""
        return bool(self._access_token) and time.time() < self._expire_at

    async def get_token(self) -> str:
        """Return a valid access_token, refreshing if necessary.

        Uses a double-check locking pattern to avoid thundering herd.

        Raises:
            httpx.HTTPStatusError: If the token endpoint returns a non-2xx status.
            httpx.RequestError: On network-level failures.
        """
        if self._is_token_valid():
            return self._access_token

        async with self._lock:
            # Double-check after acquiring the lock
            if self._is_token_valid():
                return self._access_token
            await self._refresh_token()
        return self._access_token

    async def _refresh_token(self) -> None:
        """Fetch a new access_token from DingTalk OAuth2 endpoint."""
        payload = {"appKey": self._app_key, "appSecret": self._app_secret}
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(self._TOKEN_URL, json=payload)
            resp.raise_for_status()
            data = resp.json()

        token = data.get("accessToken", "")
        expire_in = int(data.get("expireIn", 7200))

        if not token:
            raise RuntimeError("DingTalk accessToken response is empty")

        self._access_token = token
        self._expire_at = time.time() + expire_in - self._REFRESH_MARGIN_S
        logger.info(
            "DingTalk access_token refreshed, expires in %d s (margin=%d s)",
            expire_in,
            self._REFRESH_MARGIN_S,
        )


# ---------------------------------------------------------------------------
# Card client
# ---------------------------------------------------------------------------


class DingTalkCardClient:
    """Create and update DingTalk AI Card instances via streaming API.

    Depends on :class:`DingTalkTokenManager` for authentication and a
    pre-configured *template_id* for the card layout.
    """

    _CREATE_URL = "https://api.dingtalk.com/v1.0/card/instances/createAndDeliver"
    _STREAMING_URL = "https://api.dingtalk.com/v1.0/card/streaming"
    _MAX_RETRIES = 2

    def __init__(self, token_manager: DingTalkTokenManager, template_id: str) -> None:
        self._token_manager = token_manager
        self._template_id = template_id

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def create_card(
        self, conversation_id: str, content: str = "正在思考..."
    ) -> str:
        """Create and deliver a new AI Card, returning the cardInstanceId.

        Args:
            conversation_id: The DingTalk group conversation ID.
            content: Initial card content (shown while streaming starts).

        Returns:
            The ``outTrackId`` used as the card instance identifier.
        """
        out_track_id = str(uuid.uuid4())
        payload = {
            "cardTemplateId": self._template_id,
            "outTrackId": out_track_id,
            "cardData": {"cardParamMap": {"content": content}},
            "openSpaceModel": {
                "conversationType": "GROUP",
                "conversationId": conversation_id,
            },
        }
        await self._request_with_retry("POST", self._CREATE_URL, payload)
        logger.debug("Card created: outTrackId=%s", out_track_id)
        return out_track_id

    async def streaming_update(
        self, card_instance_id: str, content: str
    ) -> None:
        """Push a streaming content update to an existing card.

        Uses full-replacement mode (isFull=True) which is required for
        Markdown content rendering.

        Args:
            card_instance_id: The outTrackId returned from :meth:`create_card`.
            content: Complete current content to display.
        """
        payload = {
            "outTrackId": card_instance_id,
            "key": "content",
            "content": content,
            "isFull": True,
            "isFinalize": False,
            "guid": str(uuid.uuid4()),
        }
        await self._request_with_retry("PUT", self._STREAMING_URL, payload)

    async def finish_card(
        self, card_instance_id: str, final_content: str
    ) -> None:
        """Finalize a streaming card with the complete content.

        After this call no further updates are possible for this card.

        Args:
            card_instance_id: The outTrackId returned from :meth:`create_card`.
            final_content: The final complete content.
        """
        payload = {
            "outTrackId": card_instance_id,
            "key": "content",
            "content": final_content,
            "isFull": True,
            "isFinalize": True,
            "guid": str(uuid.uuid4()),
        }
        await self._request_with_retry("PUT", self._STREAMING_URL, payload)
        logger.debug("Card finalized: outTrackId=%s", card_instance_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _request_with_retry(
        self, method: str, url: str, payload: dict
    ) -> dict:
        """Send an authenticated request with up to 2 retries.

        Args:
            method: HTTP method ("POST" or "PUT").
            url: Target API endpoint.
            payload: JSON body to send.

        Returns:
            Parsed JSON response body.
        """
        last_exc: BaseException | None = None

        for attempt in range(self._MAX_RETRIES + 1):
            try:
                token = await self._token_manager.get_token()
                headers = {"x-acs-dingtalk-access-token": token}
                async with httpx.AsyncClient(timeout=10.0) as client:
                    if method == "PUT":
                        resp = await client.put(url, json=payload, headers=headers)
                    else:
                        resp = await client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    return resp.json()
            except (httpx.HTTPStatusError, httpx.RequestError) as exc:
                last_exc = exc
                logger.warning(
                    "DingTalk card API %s %s attempt %d failed: %s",
                    method,
                    url,
                    attempt + 1,
                    exc,
                )
                if attempt < self._MAX_RETRIES:
                    await asyncio.sleep(1 * (attempt + 1))

        logger.error(
            "DingTalk card API %s %s failed after %d attempts",
            method,
            url,
            self._MAX_RETRIES + 1,
        )
        raise RuntimeError(
            f"DingTalk card API request failed: {last_exc}"
        ) from last_exc


# ---------------------------------------------------------------------------
# Stream-to-card bridge
# ---------------------------------------------------------------------------


async def stream_to_card(
    card_client: DingTalkCardClient,
    card_instance_id: str,
    token_stream: AsyncIterator[str],
    flush_interval: float = 0.5,
) -> str:
    """Bridge an async token stream to a live DingTalk AI Card.

    Accumulates tokens and flushes the complete text to the card at most
    once every *flush_interval* seconds, minimizing API calls while keeping
    the card visually responsive.

    Args:
        card_client: A configured :class:`DingTalkCardClient` instance.
        card_instance_id: The outTrackId of the target card.
        token_stream: An async iterator yielding string tokens.
        flush_interval: Minimum seconds between streaming updates.

    Returns:
        The final complete reply text.
    """
    buffer = ""
    last_flush = time.time()

    async for token in token_stream:
        buffer += token
        now = time.time()
        if now - last_flush >= flush_interval:
            try:
                await card_client.streaming_update(card_instance_id, buffer)
            except Exception as exc:  # noqa: BLE001
                logger.warning("streaming_update failed, will retry next flush: %s", exc)
            last_flush = now

    # 流结束，发送最终完整内容并 finalize
    try:
        await card_client.finish_card(card_instance_id, buffer)
    except Exception as exc:  # noqa: BLE001
        logger.error("finish_card failed: %s", exc)

    return buffer
