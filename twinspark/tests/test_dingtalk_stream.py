"""Tests for DingTalk Stream mode integration.

覆盖：
1. TwinSparkStreamHandler 消息解析和处理
2. Card 模式（mock DingTalkCardClient）
3. Text 模式（mock agent.run）
4. 错误处理（agent 异常时发送错误提示）
5. 空消息忽略
6. DingTalkStreamManager start/stop 生命周期
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

import pytest


# --------------------------------------------------------------------------- #
# Mock dingtalk_stream SDK before importing our module
# --------------------------------------------------------------------------- #


class MockAckMessage:
    STATUS_OK = "SUCCESS"


class MockChatbotMessage:
    def __init__(self):
        self.headers = {}

    @classmethod
    def from_dict(cls, data):
        msg = cls()
        msg._data = data
        return msg


class MockChatbotHandler:
    def reply_text(self, text, message):
        pass


class MockCredential:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret


class MockDingTalkStreamClient:
    def __init__(self, credential):
        self.credential = credential
        self._handlers = {}

    def register_callback_handler(self, topic, handler):
        self._handlers[topic] = handler

    def start_forever(self):
        # Simulate blocking — just return immediately in tests
        pass


# Create a mock module
mock_dingtalk_stream = MagicMock()
mock_dingtalk_stream.AckMessage = MockAckMessage
mock_dingtalk_stream.ChatbotMessage = MockChatbotMessage
mock_dingtalk_stream.ChatbotHandler = MockChatbotHandler
mock_dingtalk_stream.Credential = MockCredential
mock_dingtalk_stream.DingTalkStreamClient = MockDingTalkStreamClient

# Patch the module before importing dingtalk_stream
sys.modules.setdefault("dingtalk_stream", mock_dingtalk_stream)

from twinspark.dingtalk_stream import DingTalkStreamManager, TwinSparkStreamHandler


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class FakeAgent:
    """Minimal agent stub for testing."""

    def __init__(self, reply: str = "fake reply"):
        self._reply = reply
        self.run_calls: list[dict] = []
        self.stream_calls: list[dict] = []

    async def run(self, message: str, *, session_id: str = "", **kwargs) -> str:
        self.run_calls.append({"message": message, "session_id": session_id})
        return self._reply

    async def run_stream(
        self, message: str, *, session_id: str = "", **kwargs
    ) -> AsyncIterator[str]:
        self.stream_calls.append({"message": message, "session_id": session_id})
        return self._stream_gen()

    async def _stream_gen(self) -> AsyncIterator[str]:
        for token in ["Hello", " ", "World"]:
            yield token

    async def aclose(self) -> None:
        pass


class FakeConfig:
    """Controllable config stub."""

    def __init__(
        self,
        message_mode: str = "text",
        app_key: str = "test-app-key",
        app_secret: str = "test-app-secret",
        card_template_id: str = "",
        stream_interval_ms: int = 500,
        dingtalk_mode: str = "stream",
    ):
        self.dingtalk_mode = dingtalk_mode
        self.dingtalk_app_key = app_key
        self.dingtalk_app_secret = app_secret
        self.dingtalk_message_mode = message_mode
        self.dingtalk_card_template_id = card_template_id
        self.dingtalk_stream_interval_ms = stream_interval_ms
        self.dingtalk_verify_signature = False


def _make_callback(
    content: str = "你好",
    conversation_id: str = "conv-stream-001",
    sender_nick: str = "测试用户",
):
    """Construct a mock callback object mimicking dingtalk-stream SDK."""
    data = json.dumps(
        {
            "conversationId": conversation_id,
            "senderNick": sender_nick,
            "text": {"content": content},
            "msgtype": "text",
            "senderId": "user-001",
        }
    )
    callback = MagicMock()
    callback.data = data
    callback.headers = {"messageId": "msg-123"}
    return callback


# --------------------------------------------------------------------------- #
# 1. 消息解析测试
# --------------------------------------------------------------------------- #


class TestStreamHandlerParsing:
    """Test TwinSparkStreamHandler message parsing."""

    async def test_process_returns_ack(self):
        """process() 返回 ACK 状态"""
        agent = FakeAgent()
        config = FakeConfig()
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback("hello")

        status, response = await handler.process(callback)

        assert status == MockAckMessage.STATUS_OK
        assert response == "ok"

    async def test_process_invalid_json(self):
        """无效 JSON 数据不崩溃，返回 ACK"""
        agent = FakeAgent()
        config = FakeConfig()
        handler = TwinSparkStreamHandler(agent, config)

        callback = MagicMock()
        callback.data = "not valid json{"

        status, response = await handler.process(callback)

        assert status == MockAckMessage.STATUS_OK
        assert response == "ok"


# --------------------------------------------------------------------------- #
# 2. 空消息测试
# --------------------------------------------------------------------------- #


class TestStreamHandlerEmptyMessage:
    """Test empty message handling."""

    async def test_empty_message_skipped(self):
        """空消息不触发 agent 调用"""
        agent = FakeAgent()
        config = FakeConfig()
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback(content="")

        await handler.process(callback)
        # Give time for any potential background tasks
        await asyncio.sleep(0.05)

        assert len(agent.run_calls) == 0
        assert len(agent.stream_calls) == 0

    async def test_whitespace_only_skipped(self):
        """纯空白消息被忽略"""
        agent = FakeAgent()
        config = FakeConfig()
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback(content="   \n  ")

        await handler.process(callback)
        await asyncio.sleep(0.05)

        assert len(agent.run_calls) == 0


# --------------------------------------------------------------------------- #
# 3. Text 模式测试
# --------------------------------------------------------------------------- #


class TestStreamHandlerTextMode:
    """Test text reply mode."""

    async def test_text_mode_calls_agent_run(self):
        """text 模式调用 agent.run 并 reply_text"""
        agent = FakeAgent(reply="机器人回复")
        config = FakeConfig(message_mode="text")
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback("你好世界")

        with patch.object(handler, "_reply_text_safe", new_callable=AsyncMock) as mock_reply:
            await handler.process(callback)
            # Wait for background task
            await asyncio.sleep(0.1)

            assert len(agent.run_calls) == 1
            assert agent.run_calls[0]["message"] == "你好世界"
            assert agent.run_calls[0]["session_id"] == "conv-stream-001"
            mock_reply.assert_called_once()
            # Verify reply text
            call_args = mock_reply.call_args[0]
            assert call_args[2] == "机器人回复"

    async def test_text_mode_uses_conversation_id_as_session(self):
        """text 模式使用 conversationId 作为 session_id"""
        agent = FakeAgent()
        config = FakeConfig(message_mode="text")
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback("测试", conversation_id="conv-xyz")

        with patch.object(handler, "_reply_text_safe", new_callable=AsyncMock):
            await handler.process(callback)
            await asyncio.sleep(0.1)

        assert agent.run_calls[0]["session_id"] == "conv-xyz"


# --------------------------------------------------------------------------- #
# 4. Card 模式测试
# --------------------------------------------------------------------------- #


class TestStreamHandlerCardMode:
    """Test card streaming mode."""

    async def test_card_mode_creates_card_and_streams(self):
        """card 模式创建卡片并流式推送"""
        agent = FakeAgent()
        config = FakeConfig(
            message_mode="card",
            app_key="key-123",
            card_template_id="tpl-001",
        )
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback("测试卡片")

        with patch(
            "twinspark.dingtalk_card.DingTalkTokenManager"
        ) as MockTokenMgr, patch(
            "twinspark.dingtalk_card.DingTalkCardClient"
        ) as MockCardClient, patch(
            "twinspark.dingtalk_card.stream_to_card", new_callable=AsyncMock
        ) as mock_stream:
            mock_client_instance = AsyncMock()
            mock_client_instance.create_card = AsyncMock(return_value="card-id-001")
            MockCardClient.return_value = mock_client_instance
            mock_stream.return_value = "Hello World"

            await handler.process(callback)
            await asyncio.sleep(0.1)

            # Token manager created with correct credentials
            MockTokenMgr.assert_called_once_with("key-123", "test-app-secret")
            # Card client created with correct params
            MockCardClient.assert_called_once()
            # Card created
            mock_client_instance.create_card.assert_called_once_with("conv-stream-001")
            # stream_to_card called
            mock_stream.assert_called_once()

    async def test_card_mode_not_used_without_template(self):
        """没有 card_template_id 时回退到 text 模式"""
        agent = FakeAgent(reply="text fallback")
        config = FakeConfig(
            message_mode="card",
            app_key="key-123",
            card_template_id="",  # No template
        )
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback("测试")

        with patch.object(handler, "_reply_text_safe", new_callable=AsyncMock) as mock_reply:
            await handler.process(callback)
            await asyncio.sleep(0.1)

            # Falls back to text mode
            assert len(agent.run_calls) == 1
            mock_reply.assert_called_once()


# --------------------------------------------------------------------------- #
# 5. 错误处理测试
# --------------------------------------------------------------------------- #


class TestStreamHandlerErrors:
    """Test error handling."""

    async def test_agent_error_sends_error_message(self):
        """agent.run 异常时发送错误提示"""
        agent = FakeAgent()
        agent.run = AsyncMock(side_effect=RuntimeError("LLM timeout"))
        config = FakeConfig(message_mode="text")
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback("会失败")

        with patch.object(handler, "_reply_text_safe", new_callable=AsyncMock) as mock_reply:
            await handler.process(callback)
            await asyncio.sleep(0.1)

            # Should have been called with error message
            mock_reply.assert_called()
            error_call = mock_reply.call_args_list[-1][0]
            assert "出错" in error_call[2]

    async def test_card_mode_error_sends_text_fallback(self):
        """card 模式失败时降级发送文本错误提示"""
        agent = FakeAgent()
        config = FakeConfig(
            message_mode="card",
            app_key="key-123",
            card_template_id="tpl-001",
        )
        handler = TwinSparkStreamHandler(agent, config)
        callback = _make_callback("会失败")

        with patch(
            "twinspark.dingtalk_card.DingTalkTokenManager"
        ), patch(
            "twinspark.dingtalk_card.DingTalkCardClient"
        ) as MockCardClient, patch.object(
            handler, "_reply_text_safe", new_callable=AsyncMock
        ) as mock_reply:
            mock_client_instance = AsyncMock()
            mock_client_instance.create_card = AsyncMock(
                side_effect=RuntimeError("Card API down")
            )
            MockCardClient.return_value = mock_client_instance

            await handler.process(callback)
            await asyncio.sleep(0.1)

            # Error fallback message sent
            mock_reply.assert_called()
            error_call = mock_reply.call_args_list[-1][0]
            assert "出错" in error_call[2]


# --------------------------------------------------------------------------- #
# 6. DingTalkStreamManager start/stop 测试
# --------------------------------------------------------------------------- #


class TestStreamManager:
    """Test DingTalkStreamManager lifecycle."""

    async def test_start_creates_background_task(self):
        """start() 创建后台任务"""
        config = FakeConfig(dingtalk_mode="stream")
        manager = DingTalkStreamManager(config)
        agent = FakeAgent()

        with patch(
            "dingtalk_stream.DingTalkStreamClient",
            MockDingTalkStreamClient,
        ), patch(
            "dingtalk_stream.Credential", MockCredential
        ):
            await manager.start(agent)

            assert manager._task is not None
            assert not manager._task.done()

            await manager.stop()

    async def test_stop_cancels_task(self):
        """stop() 取消后台任务"""
        config = FakeConfig(dingtalk_mode="stream")
        manager = DingTalkStreamManager(config)
        agent = FakeAgent()

        with patch(
            "dingtalk_stream.DingTalkStreamClient",
            MockDingTalkStreamClient,
        ), patch(
            "dingtalk_stream.Credential", MockCredential
        ):
            await manager.start(agent)
            await manager.stop()

            assert manager._task is None

    async def test_stop_when_not_started(self):
        """stop() 在未启动时不崩溃"""
        config = FakeConfig(dingtalk_mode="stream")
        manager = DingTalkStreamManager(config)

        # Should not raise
        await manager.stop()
        assert manager._task is None

    async def test_manager_sets_running_flag(self):
        """start/stop 正确管理 running 标志"""
        config = FakeConfig(dingtalk_mode="stream")
        manager = DingTalkStreamManager(config)
        agent = FakeAgent()

        with patch(
            "dingtalk_stream.DingTalkStreamClient",
            MockDingTalkStreamClient,
        ), patch(
            "dingtalk_stream.Credential", MockCredential
        ):
            await manager.start(agent)
            assert manager._running is True

            await manager.stop()
            assert manager._running is False
