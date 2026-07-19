"""Tests for DingTalk integration (signature verification + webhook endpoint).

覆盖三个维度：签名验证逻辑、端点行为、消息处理流程。
monkeypatch 注入 FakeAgent 和 fake config，不触发真实网络调用。
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import time
from unittest.mock import AsyncMock, patch

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

import pytest
from fastapi.testclient import TestClient

from twinspark import api
from twinspark.dingtalk import verify_dingtalk_signature


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #
def _make_signature(timestamp: str, secret: str) -> str:
    """用与钉钉相同的算法生成签名"""
    sign_str = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        secret.encode("utf-8"),
        sign_str.encode("utf-8"),
        hashlib.sha256,
    ).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


APP_SECRET = "test-dingtalk-secret"


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeAgent:
    """最小化 Agent 替身，仅实现 dingtalk webhook 需要的 run 方法"""

    def __init__(self) -> None:
        self.run_calls: list[dict] = []

    async def run(self, message: str, *, session_id: str = "", **kwargs) -> str:
        self.run_calls.append({"message": message, "session_id": session_id})
        return "fake reply"

    async def aclose(self) -> None:
        pass


class FakeConfig:
    """可控的配置替身"""

    def __init__(self, app_secret: str = APP_SECRET, verify: bool = True):
        self.dingtalk_app_secret = app_secret
        self.dingtalk_verify_signature = verify


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_agent() -> FakeAgent:
    return FakeAgent()


@pytest.fixture()
def client(monkeypatch, fake_agent: FakeAgent):
    """TestClient，注入 FakeAgent 并 mock get_config"""
    monkeypatch.setattr(api, "build_agent", lambda: fake_agent)
    monkeypatch.setattr(api, "build_skill_loader", lambda: type("L", (), {"list_skills": lambda self: []})())
    with TestClient(api.app) as c:
        yield c


def _dingtalk_body(content: str = "你好") -> dict:
    """构造钉钉回调请求体"""
    return {
        "conversationId": "conv-123",
        "senderNick": "测试用户",
        "senderStaffId": "staff-001",
        "text": {"content": content},
        "msgtype": "text",
        "sessionWebhook": "https://oapi.dingtalk.com/robot/sendBySession",
    }


# --------------------------------------------------------------------------- #
# 1. 签名验证测试                                                               #
# --------------------------------------------------------------------------- #
def test_verify_signature_valid():
    """正确签名 + 有效时间戳 → 返回 True"""
    ts = str(int(time.time() * 1000))
    sign = _make_signature(ts, APP_SECRET)
    assert verify_dingtalk_signature(ts, sign, APP_SECRET) is True


def test_verify_signature_invalid():
    """错误签名 → 返回 False"""
    ts = str(int(time.time() * 1000))
    assert verify_dingtalk_signature(ts, "wrong-signature", APP_SECRET) is False


def test_verify_signature_expired_timestamp():
    """超过 1 小时的时间戳 → 返回 False"""
    expired_ts = str(int((time.time() - 3700) * 1000))
    sign = _make_signature(expired_ts, APP_SECRET)
    assert verify_dingtalk_signature(expired_ts, sign, APP_SECRET) is False


# --------------------------------------------------------------------------- #
# 2. 端点行为测试                                                               #
# --------------------------------------------------------------------------- #
def test_dingtalk_webhook_not_configured(client: TestClient, monkeypatch):
    """dingtalk_app_secret 为空 → 503"""
    monkeypatch.setattr(
        "twinspark.config.get_config", lambda: FakeConfig(app_secret="")
    )
    resp = client.post("/v1/dingtalk/webhook", json=_dingtalk_body())
    assert resp.status_code == 503


def test_dingtalk_webhook_invalid_signature(client: TestClient, monkeypatch):
    """签名验证失败 → 401"""
    monkeypatch.setattr(
        "twinspark.config.get_config", lambda: FakeConfig(verify=True)
    )
    resp = client.post(
        "/v1/dingtalk/webhook",
        json=_dingtalk_body(),
        headers={"Timestamp": "123", "Sign": "bad"},
    )
    assert resp.status_code == 401


def test_dingtalk_webhook_success(client: TestClient, monkeypatch):
    """正常请求 → 200 + {"code": 0, "msg": "ok"}"""
    ts = str(int(time.time() * 1000))
    sign = _make_signature(ts, APP_SECRET)
    monkeypatch.setattr(
        "twinspark.config.get_config", lambda: FakeConfig(verify=True)
    )
    with patch("twinspark.dingtalk.send_dingtalk_reply", new_callable=AsyncMock):
        resp = client.post(
            "/v1/dingtalk/webhook",
            json=_dingtalk_body(),
            headers={"Timestamp": ts, "Sign": sign},
        )
    assert resp.status_code == 200
    assert resp.json() == {"code": 0, "msg": "ok"}


# --------------------------------------------------------------------------- #
# 3. 消息处理测试                                                               #
# --------------------------------------------------------------------------- #
def test_dingtalk_webhook_empty_message(client: TestClient, monkeypatch):
    """空消息 → {"code": 0, "msg": "empty message"}"""
    monkeypatch.setattr(
        "twinspark.config.get_config", lambda: FakeConfig(verify=False)
    )
    resp = client.post("/v1/dingtalk/webhook", json=_dingtalk_body(content=""))
    assert resp.status_code == 200
    assert resp.json() == {"code": 0, "msg": "empty message"}


def test_dingtalk_session_id_uses_conversation_id(
    client: TestClient, fake_agent: FakeAgent, monkeypatch
):
    """验证 agent.run 使用 conversationId 作为 session_id"""
    import time

    monkeypatch.setattr(
        "twinspark.config.get_config", lambda: FakeConfig(verify=False)
    )
    with patch("twinspark.dingtalk.send_dingtalk_reply", new_callable=AsyncMock):
        resp = client.post("/v1/dingtalk/webhook", json=_dingtalk_body("测试"))
        assert resp.status_code == 200
        # 等待后台 asyncio.create_task 在 ASGI 事件循环中执行完毕
        time.sleep(0.15)
    assert len(fake_agent.run_calls) == 1
    assert fake_agent.run_calls[0]["session_id"] == "conv-123"
