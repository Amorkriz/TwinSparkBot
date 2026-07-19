"""DingTalk Stream mode integration for TwinSpark.

Implements the DingTalk Stream protocol (WebSocket-based) for receiving bot
messages without requiring a public HTTPS endpoint or ICP filing.

Core components:

1. :class:`TwinSparkStreamHandler` – processes incoming chatbot messages via
   the dingtalk-stream SDK's ``ChatbotHandler`` interface.
2. :class:`DingTalkStreamManager` – manages the DingTalkStreamClient lifecycle
   including startup, reconnection, and graceful shutdown.

The handler supports two reply modes:
- **card**: Creates an AI card and streams the response token-by-token using
  the existing :mod:`twinspark.dingtalk_card` infrastructure.
- **text**: Calls ``agent.run()`` and replies with plain text via the SDK's
  built-in ``reply_text()`` method.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from twinspark.config import Config
    from twinspark.core.agent import Agent


# ---------------------------------------------------------------------------
# Stream message handler
# ---------------------------------------------------------------------------


class TwinSparkStreamHandler:
    """Handle DingTalk Stream chatbot message callbacks.

    This class is registered with the DingTalkStreamClient as the chatbot
    message handler. When a user @-mentions the bot, the SDK invokes
    :meth:`process` with the callback payload.

    Args:
        agent: The TwinSpark agent instance.
        config: Runtime configuration (for message mode, card settings, etc.).
    """

    def __init__(self, agent: "Agent", config: "Config") -> None:
        self._agent = agent
        self._config = config

    async def process(self, callback) -> tuple[str, str]:
        """Process an incoming chatbot message callback.

        Parses the message, dispatches to card or text mode in a background
        task, and immediately returns an ACK to the DingTalk gateway.

        Args:
            callback: The CallbackMessage from dingtalk-stream SDK containing
                      headers and data fields.

        Returns:
            A tuple of (status, response) for the SDK ACK mechanism.
        """
        from dingtalk_stream import AckMessage

        try:
            incoming_message = json.loads(callback.data)
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning("Failed to parse callback data: %s", exc)
            return AckMessage.STATUS_OK, "ok"

        conversation_id = incoming_message.get("conversationId", "")
        text_field = incoming_message.get("text", {})
        user_message = ""
        if isinstance(text_field, dict):
            user_message = text_field.get("content", "").strip()
        elif isinstance(text_field, str):
            user_message = text_field.strip()

        sender_nick = incoming_message.get("senderNick", "unknown")

        # Skip empty messages
        if not user_message:
            logger.debug("Empty message from %s, skipping", sender_nick)
            return AckMessage.STATUS_OK, "ok"

        logger.info(
            "Stream message from %s in %s: %s",
            sender_nick,
            conversation_id,
            user_message[:50],
        )

        # Dispatch async processing — return ACK immediately
        asyncio.create_task(
            self._handle_message(
                callback, incoming_message, user_message, conversation_id, sender_nick
            )
        )

        return AckMessage.STATUS_OK, "ok"

    async def _handle_message(
        self,
        callback,
        incoming_message: dict,
        user_message: str,
        conversation_id: str,
        sender_nick: str,
    ) -> None:
        """Background task: call agent and send reply.

        Selects between card mode and text mode based on configuration.
        Catches all exceptions to avoid crashing the stream handler.
        """
        try:
            if (
                self._config.dingtalk_message_mode in ("card", "auto")
                and self._config.dingtalk_app_key
                and self._config.dingtalk_card_template_id
            ):
                await self._handle_card_mode(user_message, conversation_id)
            else:
                await self._handle_text_mode(
                    callback, incoming_message, user_message, conversation_id
                )
        except Exception:
            logger.exception(
                "Error processing stream message from %s", sender_nick
            )
            # Attempt to send an error notification via text
            try:
                await self._reply_text_safe(
                    callback, incoming_message, "消息处理出错，请稍后再试。"
                )
            except Exception:
                logger.exception("Error reply also failed")

    async def _handle_card_mode(
        self, user_message: str, conversation_id: str
    ) -> None:
        """Process message using AI card streaming mode.

        Reuses existing DingTalkTokenManager, DingTalkCardClient, and
        stream_to_card() from the dingtalk_card module.
        """
        from twinspark.dingtalk_card import (
            DingTalkCardClient,
            DingTalkTokenManager,
            stream_to_card,
        )

        token_mgr = DingTalkTokenManager(
            self._config.dingtalk_app_key, self._config.dingtalk_app_secret
        )
        card_client = DingTalkCardClient(
            token_mgr,
            self._config.dingtalk_card_template_id,
            robot_code=self._config.dingtalk_app_key,
        )

        # Create card (shows "正在思考...")
        card_id = await card_client.create_card(conversation_id)

        # Stream generation + push to card
        stream = self._agent.run_stream(
            user_message, session_id=conversation_id
        )
        await stream_to_card(
            card_client,
            card_id,
            stream,
            flush_interval=self._config.dingtalk_stream_interval_ms / 1000.0,
        )

    async def _handle_text_mode(
        self,
        callback,
        incoming_message: dict,
        user_message: str,
        conversation_id: str,
    ) -> None:
        """Process message using plain text reply mode."""
        reply = await self._agent.run(
            user_message, session_id=conversation_id
        )
        await self._reply_text_safe(callback, incoming_message, reply)

    async def _reply_text_safe(
        self, callback, incoming_message: dict, text: str
    ) -> None:
        """Send a text reply using the dingtalk-stream SDK.

        The SDK's reply_text is synchronous (uses requests internally),
        so we run it in a thread to avoid blocking the event loop.
        """
        from dingtalk_stream import ChatbotMessage

        chatbot_msg = ChatbotMessage.from_dict(incoming_message)
        chatbot_msg.headers = getattr(callback, "headers", {})

        await asyncio.to_thread(self._sync_reply_text, chatbot_msg, text)

    @staticmethod
    def _sync_reply_text(chatbot_message, text: str) -> None:
        """Synchronous wrapper for ChatbotMessage reply."""
        from dingtalk_stream import ChatbotHandler

        handler = ChatbotHandler()
        handler.reply_text(text, chatbot_message)


# ---------------------------------------------------------------------------
# Stream client manager
# ---------------------------------------------------------------------------


class DingTalkStreamManager:
    """Manage the DingTalk Stream client lifecycle.

    Handles starting the stream client in a background task, automatic
    reconnection on failure, and graceful shutdown.

    Args:
        config: The TwinSpark runtime configuration containing DingTalk
                credentials (dingtalk_app_key, dingtalk_app_secret).
    """

    def __init__(self, config: "Config") -> None:
        self._config = config
        self._task: asyncio.Task | None = None
        self._running = False
        self._agent: Agent | None = None

    async def start(self, agent: "Agent") -> None:
        """Start the DingTalk Stream client in a background asyncio task.

        Args:
            agent: The TwinSpark agent to use for processing messages.
        """
        self._agent = agent
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info(
            "DingTalk Stream manager started (app_key=%s...)",
            self._config.dingtalk_app_key[:8]
            if self._config.dingtalk_app_key
            else "N/A",
        )

    async def stop(self) -> None:
        """Gracefully stop the stream client and cancel the background task."""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None
        logger.info("DingTalk Stream manager stopped.")

    async def _run_loop(self) -> None:
        """Main reconnection loop: start_forever() with retry on failure.

        The SDK's start_forever() is blocking, so we run it in a thread.
        On failure, waits 5 seconds before retrying.
        """
        from dingtalk_stream import DingTalkStreamClient, Credential

        while self._running:
            try:
                credential = Credential(
                    self._config.dingtalk_app_key,
                    self._config.dingtalk_app_secret,
                )
                client = DingTalkStreamClient(credential)

                # Register our handler for chatbot messages
                handler = TwinSparkStreamHandler(self._agent, self._config)
                client.register_callback_handler(
                    "/v1.0/im/bot/messages/get", handler
                )

                logger.info("DingTalk Stream client connecting...")
                # start_forever() is blocking — run in a thread
                await asyncio.to_thread(client.start_forever)

            except asyncio.CancelledError:
                logger.info("DingTalk Stream loop cancelled.")
                break
            except Exception:
                logger.exception(
                    "DingTalk Stream client disconnected, retrying in 5s..."
                )
                if self._running:
                    await asyncio.sleep(5)
