"""Tests for the TwinSpark HTTP API (FastAPI + SSE).

The network is never touched: a fake LLM client with scripted ``chat`` /
``stream`` behaviour is injected into a *real* :class:`Agent` backed by an
in-memory SQLite store, and a fake skill loader is supplied. The two module
factories :func:`twinspark.api.build_agent` / :func:`build_skill_loader` are
monkeypatched so the lifespan wires the fakes -- meaning no real
``DASHSCOPE_API_KEY`` is required.

Coverage:

* ``GET  /health``
* ``POST /v1/chat``                            (reply + session echo/generate)
* ``POST /v1/chat/stream``                     (SSE deltas + done marker)
* ``GET  /v1/sessions/{id}/messages``          (persisted history)
* ``GET  /v1/memory/search``
* ``GET  /v1/skills``
"""

from __future__ import annotations

import os

# Provide a fake key so importing config-dependent modules never fails. The
# tests always inject their own collaborators, so this key is never used to
# talk to a real backend.
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

from typing import Any

import pytest
from fastapi.testclient import TestClient

from twinspark import api
from twinspark.core.agent import Agent
from twinspark.memory.store import MemoryStore
from twinspark.skills.retriever import SkillRetriever


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Scripted stand-in for :class:`~twinspark.core.llm.LLMClient`."""

    def __init__(
        self,
        *,
        reply: str = "完整回复",
        deltas: list[str] | None = None,
    ) -> None:
        self.reply = reply
        self.deltas = deltas if deltas is not None else ["块1", "块2", "块3"]
        self.chat_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []
        self.closed = False

    async def chat(self, messages: list[dict], **kwargs: Any) -> str:
        self.chat_calls.append(dict(kwargs))
        return self.reply

    async def stream(self, messages: list[dict], **kwargs: Any):
        self.stream_calls.append(dict(kwargs))
        for delta in self.deltas:
            yield delta

    async def aclose(self) -> None:
        self.closed = True


class FakeSkillLoader:
    """Minimal skill loader exposing only ``list_skills``."""

    def __init__(self, skills: list[dict] | None = None) -> None:
        self._skills = skills if skills is not None else [
            {
                "name": "python-help",
                "description": "协助编写 Python 代码",
                "category": "general",
                "tags": ["python", "code"],
            }
        ]

    def list_skills(self) -> list[dict]:
        return list(self._skills)


# --------------------------------------------------------------------------- #
# Fixtures                                                                     #
# --------------------------------------------------------------------------- #
@pytest.fixture()
def fake_llm() -> FakeLLM:
    return FakeLLM(reply="这是完整回复", deltas=["Hel", "lo", "!"])


@pytest.fixture()
def fake_agent(fake_llm: FakeLLM) -> Agent:
    """A real Agent with a fake LLM + in-memory store (no network)."""
    store = MemoryStore(":memory:")
    agent = Agent(
        llm=fake_llm,
        memory=store,
        skill_retriever=SkillRetriever([]),
        session_id="default-session",
    )
    return agent


@pytest.fixture()
def fake_loader() -> FakeSkillLoader:
    return FakeSkillLoader()


@pytest.fixture()
def client(
    monkeypatch: pytest.MonkeyPatch,
    fake_agent: Agent,
    fake_loader: FakeSkillLoader,
):
    """A TestClient whose lifespan wires the injected fakes."""
    monkeypatch.setattr(api, "build_agent", lambda: fake_agent)
    monkeypatch.setattr(api, "build_skill_loader", lambda: fake_loader)
    # ``with`` triggers startup/shutdown (lifespan) events.
    with TestClient(api.app) as test_client:
        yield test_client


# --------------------------------------------------------------------------- #
# /health                                                                      #
# --------------------------------------------------------------------------- #
def test_health(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# --------------------------------------------------------------------------- #
# /v1/chat                                                                     #
# --------------------------------------------------------------------------- #
def test_chat_returns_reply_and_echoes_session(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat", json={"message": "你好", "session_id": "abc"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "这是完整回复"
    assert body["session_id"] == "abc"


def test_chat_generates_session_when_missing(client: TestClient) -> None:
    resp = client.post("/v1/chat", json={"message": "你好"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["reply"] == "这是完整回复"
    # Server generated a non-empty session id and returned it.
    assert isinstance(body["session_id"], str) and body["session_id"]


def test_chat_forwards_temperature(
    client: TestClient, fake_llm: FakeLLM
) -> None:
    resp = client.post(
        "/v1/chat", json={"message": "hi", "temperature": 0.3}
    )
    assert resp.status_code == 200
    assert fake_llm.chat_calls[-1].get("temperature") == 0.3


def test_chat_persists_history(client: TestClient) -> None:
    client.post("/v1/chat", json={"message": "消息A", "session_id": "sess1"})
    resp = client.get("/v1/sessions/sess1/messages")
    assert resp.status_code == 200
    messages = resp.json()["messages"]
    assert [(m["role"], m["content"]) for m in messages] == [
        ("user", "消息A"),
        ("assistant", "这是完整回复"),
    ]


# --------------------------------------------------------------------------- #
# /v1/chat/stream                                                              #
# --------------------------------------------------------------------------- #
def test_chat_stream_returns_sse_chunks_and_done(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/stream", json={"message": "hi", "session_id": "stream1"}
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    text = resp.text
    # Each delta arrives as its own data frame.
    assert "data: Hel" in text
    assert "data: lo" in text
    assert "data: !" in text
    # Session announced up front, and a terminating done marker is present.
    assert "event: session" in text
    assert "data: stream1" in text
    assert "event: done" in text
    assert "data: [DONE]" in text


def test_chat_stream_persists_full_reply(client: TestClient) -> None:
    resp = client.post(
        "/v1/chat/stream", json={"message": "提问", "session_id": "stream2"}
    )
    assert resp.status_code == 200

    history = client.get("/v1/sessions/stream2/messages").json()["messages"]
    assert [(m["role"], m["content"]) for m in history] == [
        ("user", "提问"),
        ("assistant", "Hello!"),
    ]


# --------------------------------------------------------------------------- #
# /v1/sessions/{id}/messages                                                   #
# --------------------------------------------------------------------------- #
def test_session_messages_empty_for_unknown(client: TestClient) -> None:
    resp = client.get("/v1/sessions/does-not-exist/messages")
    assert resp.status_code == 200
    assert resp.json() == {"session_id": "does-not-exist", "messages": []}


def test_sessions_are_isolated(client: TestClient) -> None:
    client.post("/v1/chat", json={"message": "A", "session_id": "one"})
    client.post("/v1/chat", json={"message": "B", "session_id": "two"})

    one = client.get("/v1/sessions/one/messages").json()["messages"]
    two = client.get("/v1/sessions/two/messages").json()["messages"]
    assert [m["content"] for m in one if m["role"] == "user"] == ["A"]
    assert [m["content"] for m in two if m["role"] == "user"] == ["B"]


# --------------------------------------------------------------------------- #
# /v1/memory/search                                                            #
# --------------------------------------------------------------------------- #
def test_memory_search_returns_recalled_facts(
    client: TestClient, fake_agent: Agent
) -> None:
    fake_agent.memory.add_fact("用户名叫 Jarvis", tags="profile", trust_score=0.9)

    resp = client.get("/v1/memory/search", params={"q": "Jarvis", "limit": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["query"] == "Jarvis"
    contents = [r["content"] for r in body["results"]]
    assert "用户名叫 Jarvis" in contents


def test_memory_search_empty(client: TestClient) -> None:
    resp = client.get("/v1/memory/search", params={"q": "nothing-here"})
    assert resp.status_code == 200
    assert resp.json()["results"] == []


# --------------------------------------------------------------------------- #
# /v1/skills                                                                   #
# --------------------------------------------------------------------------- #
def test_list_skills(client: TestClient) -> None:
    resp = client.get("/v1/skills")
    assert resp.status_code == 200
    skills = resp.json()["skills"]
    assert len(skills) == 1
    assert skills[0]["name"] == "python-help"
    assert skills[0]["tags"] == ["python", "code"]
