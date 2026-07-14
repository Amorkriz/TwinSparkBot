"""End-to-end integration smoke tests for TwinSpark.

These tests exercise full cross-module interactions with a *mock* LLM so no
network access or real API key is ever needed. They cover:

* **CLI path**: Agent multi-turn with history carried forward and persisted.
* **API path**: TestClient multi-turn, SSE streaming, session isolation.
* **Memory + Skills through-flow**: add_fact recall hit + skill retrieval from
  a temporary skills directory.
"""

from __future__ import annotations

import os
import textwrap
import tempfile
from pathlib import Path
from typing import Any

# Provide a fake key so importing config-dependent modules never fails.
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

import pytest
from fastapi.testclient import TestClient

from twinspark import api
from twinspark.core.agent import Agent
from twinspark.memory.store import MemoryStore
from twinspark.skills.loader import Skill, SkillLoader
from twinspark.skills.retriever import SkillRetriever


# --------------------------------------------------------------------------- #
# Fake LLM                                                                     #
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Scripted stand-in for LLMClient; records messages sent."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self._replies = list(replies) if replies else ["回复1", "回复2", "回复3"]
        self._call_idx = 0
        self.chat_calls: list[list[dict[str, Any]]] = []
        self.stream_calls: list[list[dict[str, Any]]] = []
        self.closed = False

    def _next_reply(self) -> str:
        reply = self._replies[min(self._call_idx, len(self._replies) - 1)]
        self._call_idx += 1
        return reply

    async def chat(self, messages: list[dict], **kwargs: Any) -> str:
        self.chat_calls.append([dict(m) for m in messages])
        return self._next_reply()

    async def stream(self, messages: list[dict], **kwargs: Any):
        self.stream_calls.append([dict(m) for m in messages])
        reply = self._next_reply()
        # Yield char-by-char to simulate streaming
        for ch in reply:
            yield ch

    async def aclose(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# CLI path: multi-turn conversation via Agent directly                         #
# --------------------------------------------------------------------------- #
class TestCLIPath:
    """Simulate what the CLI does: call Agent.run multiple turns."""

    @pytest.fixture()
    def store(self) -> MemoryStore:
        s = MemoryStore(":memory:")
        try:
            yield s
        finally:
            s.close()

    async def test_multiturn_history_and_persistence(self, store: MemoryStore) -> None:
        """Two turns: reply produced, both persisted, second gets first's history."""
        llm = FakeLLM(replies=["你好呀！", "第二轮回复"])
        agent = Agent(
            llm=llm,
            memory=store,
            skill_retriever=SkillRetriever([]),
            session_id="cli-sess",
        )

        # --- Turn 1 ---
        reply1 = await agent.run("你好")
        assert reply1 == "你好呀！"

        # Verify user+assistant persisted
        history = store.get_history("cli-sess")
        assert [(m["role"], m["content"]) for m in history] == [
            ("user", "你好"),
            ("assistant", "你好呀！"),
        ]

        # --- Turn 2 ---
        reply2 = await agent.run("第二轮提问")
        assert reply2 == "第二轮回复"

        # Verify second turn carries first turn's history into the LLM call
        second_call_msgs = llm.chat_calls[1]
        roles = [m["role"] for m in second_call_msgs]
        assert roles == ["system", "user", "assistant", "user"]
        assert second_call_msgs[1]["content"] == "你好"
        assert second_call_msgs[2]["content"] == "你好呀！"
        assert second_call_msgs[3]["content"] == "第二轮提问"

        # All four messages persisted
        history2 = store.get_history("cli-sess")
        assert len(history2) == 4
        assert history2[2]["role"] == "user"
        assert history2[2]["content"] == "第二轮提问"
        assert history2[3]["role"] == "assistant"
        assert history2[3]["content"] == "第二轮回复"


# --------------------------------------------------------------------------- #
# API path: TestClient multi-turn + SSE + session isolation                    #
# --------------------------------------------------------------------------- #
class TestAPIPath:
    """Integration tests via FastAPI TestClient with monkeypatched agent."""

    @pytest.fixture()
    def fake_llm(self) -> FakeLLM:
        return FakeLLM(replies=["API回复A", "API回复B", "API回复C"])

    @pytest.fixture()
    def fake_agent(self, fake_llm: FakeLLM) -> Agent:
        store = MemoryStore(":memory:")
        return Agent(
            llm=fake_llm,
            memory=store,
            skill_retriever=SkillRetriever([]),
            session_id="api-default",
        )

    @pytest.fixture()
    def client(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_agent: Agent,
    ):
        monkeypatch.setattr(api, "build_agent", lambda: fake_agent)
        monkeypatch.setattr(api, "build_skill_loader", lambda: SkillLoader("/dev/null"))
        with TestClient(api.app) as tc:
            yield tc

    def test_multiturn_same_session(
        self, client: TestClient, fake_llm: FakeLLM
    ) -> None:
        """Two POSTs with same session_id; second carries first's history."""
        sid = "integ-sess-1"
        r1 = client.post("/v1/chat", json={"message": "问题1", "session_id": sid})
        assert r1.status_code == 200
        assert r1.json()["reply"] == "API回复A"

        r2 = client.post("/v1/chat", json={"message": "问题2", "session_id": sid})
        assert r2.status_code == 200
        assert r2.json()["reply"] == "API回复B"

        # Verify history accumulation via GET endpoint
        history_resp = client.get(f"/v1/sessions/{sid}/messages")
        messages = history_resp.json()["messages"]
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "问题1"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "API回复A"
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "问题2"
        assert messages[3]["role"] == "assistant"
        assert messages[3]["content"] == "API回复B"

        # Verify second LLM call received first turn in history
        second_msgs = fake_llm.chat_calls[1]
        roles = [m["role"] for m in second_msgs]
        assert roles == ["system", "user", "assistant", "user"]

    def test_stream_sse_chunks_and_done(self, client: TestClient) -> None:
        """POST /v1/chat/stream returns SSE with deltas + done marker."""
        resp = client.post(
            "/v1/chat/stream",
            json={"message": "流式问题", "session_id": "stream-integ"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")

        text = resp.text
        # Session event present
        assert "event: session" in text
        assert "data: stream-integ" in text
        # Done marker present
        assert "event: done" in text
        assert "data: [DONE]" in text
        # At least one data frame before done (the streamed reply)
        # Our FakeLLM yields char-by-char of "API回复A"
        assert "data: A" in text

    def test_session_isolation(self, client: TestClient) -> None:
        """Two different sessions do not bleed history."""
        client.post("/v1/chat", json={"message": "sessA-msg", "session_id": "sessA"})
        client.post("/v1/chat", json={"message": "sessB-msg", "session_id": "sessB"})

        hist_a = client.get("/v1/sessions/sessA/messages").json()["messages"]
        hist_b = client.get("/v1/sessions/sessB/messages").json()["messages"]

        user_a = [m["content"] for m in hist_a if m["role"] == "user"]
        user_b = [m["content"] for m in hist_b if m["role"] == "user"]

        assert user_a == ["sessA-msg"]
        assert user_b == ["sessB-msg"]
        # Ensure no cross-contamination
        assert "sessB-msg" not in [m["content"] for m in hist_a]
        assert "sessA-msg" not in [m["content"] for m in hist_b]


# --------------------------------------------------------------------------- #
# Memory + Skills through-flow                                                 #
# --------------------------------------------------------------------------- #
class TestMemorySkillsFlow:
    """Verify fact recall and skill retrieval work end-to-end in one Agent."""

    @pytest.fixture()
    def store(self) -> MemoryStore:
        s = MemoryStore(":memory:")
        try:
            yield s
        finally:
            s.close()

    def test_add_fact_then_recall_hits(self, store: MemoryStore) -> None:
        """add_fact + recall returns the fact when query overlaps."""
        store.add_fact("用户的名字是 Jarvis", tags="profile name", trust_score=0.9)
        store.add_fact("用户喜欢 Python 编程", tags="hobby", trust_score=0.8)

        results = store.recall("Jarvis")
        contents = [r["content"] for r in results]
        assert "用户的名字是 Jarvis" in contents

    def test_skill_retriever_from_temp_dir(self, store: MemoryStore) -> None:
        """Load a skill from a temporary directory and retrieve it by query."""
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "general" / "python-help"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                textwrap.dedent("""\
                    ---
                    name: python-help
                    description: 协助编写 Python 代码和调试
                    tags: [python, code, debug]
                    ---

                    # Python Help
                    Follow PEP8. Use type hints.
                """),
                encoding="utf-8",
            )

            loader = SkillLoader(skills_dir=tmpdir)
            skills = loader.load_all()
            assert len(skills) == 1
            assert skills[0].name == "python-help"

            retriever = SkillRetriever(skills)
            hits = retriever.retrieve("python code")
            assert len(hits) >= 1
            assert hits[0].name == "python-help"

    async def test_agent_injects_recalled_fact_and_skill(
        self, store: MemoryStore
    ) -> None:
        """Full agent turn: recalled fact and skill land in the system prompt."""
        # Use a fact with English tokens so FTS5 tokenization matches reliably.
        store.add_fact(
            "User prefers concise answers", tags="preference concise", trust_score=0.9
        )

        skill = Skill(
            name="concise-answer",
            description="Guide model to give concise answers",
            tags=["concise", "answer"],
            body="回答尽量简短,不超过三句话。",
            category="general",
            path=None,
        )
        llm = FakeLLM(replies=["好的，简洁回答"])
        agent = Agent(
            llm=llm,
            memory=store,
            skill_retriever=SkillRetriever([skill]),
            session_id="inject-test",
        )

        # Query overlaps fact ("concise") and skill (tag "concise", name has it)
        reply = await agent.run("concise answer please")
        assert reply == "好的，简洁回答"

        # Verify the system prompt contains the recalled fact
        system_msg = llm.chat_calls[0][0]
        assert system_msg["role"] == "system"
        assert "User prefers concise answers" in system_msg["content"]
        # Verify the skill was injected
        assert "concise-answer" in system_msg["content"]
