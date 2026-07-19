"""Tests for DingTalk AI Card module (token management, card client, stream bridge).

覆盖四个维度：
1. TokenManager – token 获取 / 缓存 / 过期刷新
2. CardClient – 创建 / 流式更新 / finalize
3. stream_to_card – 按间隔刷新 + 返回完整文本
4. 端点集成 – card 模式路由 + 失败降级
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

import pytest

from twinspark.dingtalk_card import DingTalkTokenManager, DingTalkCardClient, stream_to_card


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _fake_token_response(token: str = "fake-token-123", expire_in: int = 7200):
    """构造 mock token API 响应"""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"accessToken": token, "expireIn": expire_in}
    return resp


def _fake_card_response(data: dict | None = None):
    """构造 mock card API 响应"""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = data or {"success": True}
    return resp


async def _fake_token_stream(tokens: list[str]) -> AsyncIterator[str]:
    """生成用于测试的 async token 迭代器"""
    for t in tokens:
        yield t


# --------------------------------------------------------------------------- #
# 1. TokenManager 测试                                                         #
# --------------------------------------------------------------------------- #


@pytest.fixture()
def token_manager():
    return DingTalkTokenManager(app_key="key1", app_secret="secret1")


class TestTokenManager:
    """DingTalkTokenManager 单元测试"""

    async def test_get_token_fetches_on_first_call(self, token_manager):
        """首次调用获取 token，验证 HTTP 调用"""
        mock_post = AsyncMock(return_value=_fake_token_response("tok-abc"))

        with patch("twinspark.dingtalk_card.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            result = await token_manager.get_token()

        assert result == "tok-abc"
        mock_post.assert_called_once()
        # 验证请求参数
        call_kwargs = mock_post.call_args
        assert call_kwargs[1]["json"]["appKey"] == "key1"
        assert call_kwargs[1]["json"]["appSecret"] == "secret1"

    async def test_get_token_returns_cached(self, token_manager):
        """第二次调用返回缓存值，不再请求 API"""
        mock_post = AsyncMock(return_value=_fake_token_response("tok-cached"))

        with patch("twinspark.dingtalk_card.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            first = await token_manager.get_token()
            second = await token_manager.get_token()

        assert first == second == "tok-cached"
        # 只调用一次 HTTP
        assert mock_post.call_count == 1

    async def test_get_token_refreshes_when_expired(self, token_manager):
        """token 过期后自动刷新"""
        mock_post = AsyncMock(
            side_effect=[
                _fake_token_response("tok-old", expire_in=7200),
                _fake_token_response("tok-new", expire_in=7200),
            ]
        )

        with patch("twinspark.dingtalk_card.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            first = await token_manager.get_token()
            assert first == "tok-old"

            # 手动让 token 过期
            token_manager._expire_at = time.time() - 1

            second = await token_manager.get_token()
            assert second == "tok-new"
            assert mock_post.call_count == 2


# --------------------------------------------------------------------------- #
# 2. CardClient 测试                                                            #
# --------------------------------------------------------------------------- #


class TestCardClient:
    """DingTalkCardClient 单元测试"""

    @pytest.fixture()
    def card_client(self):
        """创建一个 mock token_manager 的 CardClient"""
        token_mgr = AsyncMock()
        token_mgr.get_token = AsyncMock(return_value="test-token")
        return DingTalkCardClient(token_mgr, template_id="tpl-001")

    async def test_create_card_success(self, card_client):
        """创建卡片成功，返回 outTrackId (UUID)"""
        mock_post = AsyncMock(return_value=_fake_card_response({"success": True}))

        with patch("twinspark.dingtalk_card.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.post = mock_post
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            card_id = await card_client.create_card("conv-001", "正在思考...")

        assert card_id  # 非空 UUID 字符串
        assert len(card_id) == 36  # UUID 格式
        # 验证请求包含正确的 template_id
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["cardTemplateId"] == "tpl-001"
        assert call_kwargs["json"]["openSpaceModel"]["conversationId"] == "conv-001"
        assert call_kwargs["headers"]["x-acs-dingtalk-access-token"] == "test-token"

    async def test_streaming_update_sends_content(self, card_client):
        """流式更新发送正确请求"""
        mock_put = AsyncMock(return_value=_fake_card_response())

        with patch("twinspark.dingtalk_card.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.put = mock_put
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await card_client.streaming_update("card-123", "Hello world")

        call_kwargs = mock_put.call_args[1]
        payload = call_kwargs["json"]
        assert payload["outTrackId"] == "card-123"
        assert payload["content"] == "Hello world"
        assert payload["isFull"] is True
        assert payload["isFinalize"] is False

    async def test_finish_card_sets_finalize(self, card_client):
        """finish 标记 isFinalize=True"""
        mock_put = AsyncMock(return_value=_fake_card_response())

        with patch("twinspark.dingtalk_card.httpx.AsyncClient") as MockClient:
            instance = AsyncMock()
            instance.put = mock_put
            instance.__aenter__ = AsyncMock(return_value=instance)
            instance.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = instance

            await card_client.finish_card("card-456", "Final content")

        call_kwargs = mock_put.call_args[1]
        payload = call_kwargs["json"]
        assert payload["outTrackId"] == "card-456"
        assert payload["content"] == "Final content"
        assert payload["isFinalize"] is True


# --------------------------------------------------------------------------- #
# 3. stream_to_card 测试                                                        #
# --------------------------------------------------------------------------- #


class TestStreamToCard:
    """stream_to_card 桥接函数测试"""

    async def test_stream_to_card_flushes_on_interval(self):
        """验证按时间间隔刷新：使用短间隔加速测试"""
        mock_card_client = AsyncMock(spec=DingTalkCardClient)
        mock_card_client.streaming_update = AsyncMock()
        mock_card_client.finish_card = AsyncMock()

        # 使用多个 token，flush_interval 极短确保能触发多次 flush
        tokens = ["Hello", " ", "World", "!", " Nice"]

        async def _slow_stream() -> AsyncIterator[str]:
            for t in tokens:
                yield t
                await asyncio.sleep(0.02)  # 每个 token 间隔 20ms

        result = await stream_to_card(
            mock_card_client, "card-789", _slow_stream(), flush_interval=0.01
        )

        # streaming_update 应该被调用至少 1 次（第一个 token 后间隔满足条件）
        assert mock_card_client.streaming_update.call_count >= 1
        # finish_card 必须被调用一次
        mock_card_client.finish_card.assert_called_once()
        assert result == "Hello World! Nice"

    async def test_stream_to_card_returns_full_text(self):
        """验证返回完整文本"""
        mock_card_client = AsyncMock(spec=DingTalkCardClient)
        mock_card_client.streaming_update = AsyncMock()
        mock_card_client.finish_card = AsyncMock()

        tokens = ["你", "好", "，", "世", "界"]

        result = await stream_to_card(
            mock_card_client, "card-full", _fake_token_stream(tokens), flush_interval=10.0
        )

        # flush_interval 很大，不会触发 streaming_update
        assert mock_card_client.streaming_update.call_count == 0
        # 但 finish_card 仍然被调用
        mock_card_client.finish_card.assert_called_once_with("card-full", "你好，世界")
        assert result == "你好，世界"


# --------------------------------------------------------------------------- #
# 4. 端点集成测试                                                               #
# --------------------------------------------------------------------------- #


class TestWebhookCardIntegration:
    """测试 webhook 端点与 card 模式的集成"""

    @pytest.fixture()
    def fake_agent(self):
        """FakeAgent，支持 run 和 run_stream"""
        agent = AsyncMock()
        agent.run = AsyncMock(return_value="text reply")

        async def _stream(message, *, session_id="", **kwargs):
            for t in ["Hello", " ", "World"]:
                yield t

        agent.run_stream = _stream
        return agent

    @pytest.fixture()
    def client(self, monkeypatch, fake_agent):
        from fastapi.testclient import TestClient
        from twinspark import api

        monkeypatch.setattr(api, "build_agent", lambda: fake_agent)
        monkeypatch.setattr(
            api, "build_skill_loader",
            lambda: type("L", (), {"list_skills": lambda self: []})(),
        )
        with TestClient(api.app) as c:
            yield c

    def _dingtalk_body(self, content: str = "你好") -> dict:
        return {
            "conversationId": "conv-card-test",
            "senderNick": "测试用户",
            "senderStaffId": "staff-001",
            "text": {"content": content},
            "msgtype": "text",
            "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession",
        }

    def _fake_config(self, mode: str = "card"):
        """返回 card 模式的 FakeConfig"""
        cfg = MagicMock()
        cfg.dingtalk_app_secret = "test-secret"
        cfg.dingtalk_app_key = "test-key"
        cfg.dingtalk_verify_signature = False
        cfg.dingtalk_message_mode = mode
        cfg.dingtalk_card_template_id = "tpl-001"
        cfg.dingtalk_stream_interval_ms = 500
        return cfg

    def test_webhook_routes_to_card_mode(self, client, monkeypatch):
        """card 模式下 webhook 正确路由到 process_dingtalk_card_stream"""
        monkeypatch.setattr(
            "twinspark.config.get_config", lambda: self._fake_config("card")
        )
        with patch(
            "twinspark.dingtalk.process_dingtalk_card_stream", new_callable=AsyncMock
        ) as mock_card:
            resp = client.post("/v1/dingtalk/webhook", json=self._dingtalk_body("测试卡片"))
            assert resp.status_code == 200
            assert resp.json() == {"code": 0, "msg": "ok"}
            # 等待后台 task 执行
            time.sleep(0.15)
            mock_card.assert_called_once()

    def test_webhook_fallback_on_card_failure(self, client, monkeypatch):
        """卡片失败时降级发送纯文本错误提示"""
        monkeypatch.setattr(
            "twinspark.config.get_config", lambda: self._fake_config("card")
        )
        with patch(
            "twinspark.dingtalk.process_dingtalk_card_stream",
            new_callable=AsyncMock,
            side_effect=RuntimeError("card API down"),
        ), patch(
            "twinspark.dingtalk.send_dingtalk_reply", new_callable=AsyncMock
        ) as mock_reply:
            resp = client.post("/v1/dingtalk/webhook", json=self._dingtalk_body("会失败"))
            assert resp.status_code == 200
            # 等待后台 task 执行（包含异常处理）
            time.sleep(0.15)
            mock_reply.assert_called_once()
            # 验证降级消息内容
            call_args = mock_reply.call_args[0]
            assert "出错" in call_args[1]
