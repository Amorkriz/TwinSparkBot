"""DingTalk integration for TwinSpark.

Handles the full lifecycle of a DingTalk enterprise bot callback:

1. Parse the incoming webhook request body (:class:`DingTalkCallbackBody`).
2. Verify the HMAC-SHA256 signature sent by DingTalk (:func:`verify_dingtalk_signature`).
3. Send replies back to the group chat via the session webhook
   (:func:`send_dingtalk_reply`).

Only standard-library cryptographic modules are used for signature verification;
``httpx`` (already a project dependency) is used for outbound HTTP calls.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import logging
import time
from typing import TYPE_CHECKING, Any, Optional

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class DingTalkCallbackBody(BaseModel):
    """Parsed DingTalk webhook callback request body.

    All fields have sensible defaults so that DingTalk URL-verification
    requests and edge-case payloads never trigger a Pydantic ValidationError.
    Only ``msgtype`` and ``text`` are semantically required for processing;
    everything else defaults to empty/neutral values.
    """

    conversationId: str = ""
    senderNick: str = ""
    senderStaffId: str = ""
    text: dict = Field(default_factory=lambda: {"content": ""})
    msgtype: str = "text"
    sessionWebhook: str = ""
    sessionWebhookExpiredTime: int = 0
    conversationType: str = "2"

    # Optional fields that DingTalk may include
    msgId: Optional[str] = None
    createAt: Optional[int] = None
    atUsers: Optional[list[dict[str, Any]]] = None
    chatbotCorpId: Optional[str] = None
    chatbotUserId: Optional[str] = None
    isAdmin: Optional[bool] = None
    conversationTitle: Optional[str] = None


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

# Maximum age (in seconds) for a valid timestamp – 1 hour.
_MAX_TIMESTAMP_AGE_S = 3600


def verify_dingtalk_signature(
    timestamp: str, sign: str, app_secret: str
) -> bool:
    """Verify the HMAC-SHA256 signature from DingTalk.

    Algorithm: ``Base64(HMAC-SHA256(timestamp + "\\n" + secret, secret_as_key))``

    Args:
        timestamp: The ``Timestamp`` header value (milliseconds since epoch).
        sign: The ``Sign`` header value (Base64-encoded HMAC digest).
        app_secret: The robot's AppSecret configured in the DingTalk console.

    Returns:
        ``True`` if the signature is valid and the timestamp is within the
        allowed window; ``False`` otherwise.
    """
    if not timestamp or not sign or not app_secret:
        return False

    # Timestamp freshness check (DingTalk sends milliseconds).
    try:
        ts_ms = int(timestamp)
    except (ValueError, TypeError):
        logger.warning("Invalid DingTalk timestamp: %r", timestamp)
        return False

    now_ms = int(time.time() * 1000)
    if abs(now_ms - ts_ms) > _MAX_TIMESTAMP_AGE_S * 1000:
        logger.warning(
            "DingTalk timestamp expired: received=%s, now=%s", ts_ms, now_ms
        )
        return False

    # Compute expected signature.
    sign_str = f"{timestamp}\n{app_secret}"
    hmac_code = hmac.new(
        app_secret.encode("utf-8"),
        sign_str.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    calculated = base64.b64encode(hmac_code).decode("utf-8")

    return hmac.compare_digest(calculated, sign)


# ---------------------------------------------------------------------------
# Reply sender
# ---------------------------------------------------------------------------


async def send_dingtalk_reply(
    session_webhook: str, reply_text: str, sender_staff_id: str = ""
) -> None:
    """Send a text reply back to the DingTalk group via *sessionWebhook*.

    Features:
    * 3 retries with linear back-off (1 s, 2 s, 3 s).
    * 410/403 responses are treated as expired webhooks and abort immediately.
    * If *sender_staff_id* is provided, the reply will @-mention the sender.

    Args:
        session_webhook: The callback URL provided in the original request.
        reply_text: Plain text to send as a reply.
        sender_staff_id: Optional staff ID to @-mention in the reply.
    """
    payload: dict[str, Any] = {
        "msgtype": "text",
        "text": {"content": reply_text},
    }
    if sender_staff_id:
        payload["at"] = {"atUserIds": [sender_staff_id], "isAtAll": False}

    async with httpx.AsyncClient(timeout=10.0) as client:
        for attempt in range(3):
            try:
                resp = await client.post(session_webhook, json=payload)
                resp.raise_for_status()
                logger.debug("DingTalk reply sent successfully.")
                return
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code in (410, 403):
                    logger.warning(
                        "sessionWebhook expired (HTTP %s): %s",
                        exc.response.status_code,
                        exc,
                    )
                    return
                if attempt == 2:
                    logger.error(
                        "Failed to send DingTalk reply after 3 attempts: %s",
                        exc,
                    )
            except httpx.RequestError as exc:
                if attempt == 2:
                    logger.error(
                        "Network error sending DingTalk reply: %s", exc
                    )
            # Linear back-off: 1 s, 2 s, 3 s.
            await asyncio.sleep(1 * (attempt + 1))


# ---------------------------------------------------------------------------
# Card streaming
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from twinspark.core.agent import Agent


async def process_dingtalk_card_stream(
    agent: "Agent",
    user_message: str,
    conversation_id: str,
    sender_staff_id: str = "",
    session_webhook: str = "",
) -> None:
    """AI 卡片流式模式处理（后台任务）

    1. 创建卡片（显示"思考中..."）
    2. 调用 agent.run_stream() 逐块生成
    3. 通过 stream_to_card() 定时推送更新
    4. 异常时降级发送纯文本错误提示
    """
    from twinspark.config import get_config
    from twinspark.dingtalk_card import (
        DingTalkTokenManager, DingTalkCardClient, stream_to_card
    )

    config = get_config()

    try:
        token_mgr = DingTalkTokenManager(config.dingtalk_app_key, config.dingtalk_app_secret)
        card_client = DingTalkCardClient(token_mgr, config.dingtalk_card_template_id)

        # 创建卡片
        card_id = await card_client.create_card(conversation_id)

        # 流式生成 + 推送
        stream = agent.run_stream(user_message, session_id=conversation_id)
        await stream_to_card(
            card_client, card_id, stream,
            flush_interval=config.dingtalk_stream_interval_ms / 1000.0,
        )

    except Exception:
        logger.exception("Card streaming failed, falling back to text")
        # 降级：发送错误提示（不重复调用 agent 避免重复持久化）
        try:
            await send_dingtalk_reply(session_webhook, "消息处理出错，请稍后再试。", sender_staff_id)
        except Exception:
            logger.exception("Text fallback also failed")
