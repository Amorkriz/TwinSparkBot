"""Unit tests for the TwinSpark agent core (mocked LLM, real in-memory store).

These tests never touch the network: a fake LLM client with scripted
``chat`` / ``stream`` behaviour is injected, memory uses an in-memory SQLite
database, and skills are supplied as an explicit list. The assertions cover:

* :meth:`Agent.run` returning the full reply (non-streaming),
* :meth:`Agent.run_stream` yielding deltas (streaming),
* both user and assistant messages being persisted,
* multi-turn history being carried into the next request,
* recalled facts and skill injection landing in the system message,
* partial output being persisted when streaming is interrupted,
* the ``conversation.build_messages`` assembly contract in isolation.
"""

from __future__ import annotations

import os

# Provide a fake key so importing config-dependent modules never fails, even
# though these tests always inject their own collaborators.
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

from typing import Any

import pytest

from twinspark.core import conversation
from twinspark.core.agent import Agent
from twinspark.memory.store import MemoryStore
from twinspark.skills.loader import Skill
from twinspark.skills.retriever import SkillRetriever


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeLLM:
    """Scripted stand-in for :class:`~twinspark.core.llm.LLMClient`.

    Records the ``messages`` passed to each call so tests can assert on prompt
    assembly. ``chat`` returns a fixed reply; ``stream`` yields fixed deltas.
    """

    def __init__(
        self,
        *,
        reply: str = "完整回复",
        deltas: list[str] | None = None,
        stream_error: Exception | None = None,
    ) -> None:
        self.reply = reply
        self.deltas = deltas if deltas is not None else ["块1", "块2", "块3"]
        self.stream_error = stream_error
        self.chat_calls: list[list[dict[str, Any]]] = []
        self.stream_calls: list[list[dict[str, Any]]] = []
        self.closed = False

    async def chat(self, messages: list[dict], **kwargs: Any) -> str:
        # Store a copy so later mutation cannot affect the recorded snapshot.
        self.chat_calls.append([dict(m) for m in messages])
        return self.reply

    async def stream(self, messages: list[dict], **kwargs: Any):
        self.stream_calls.append([dict(m) for m in messages])
        for delta in self.deltas:
            yield delta
        if self.stream_error is not None:
            raise self.stream_error

    async def aclose(self) -> None:
        self.closed = True


def _skill(name: str, description: str, tags: list[str], body: str) -> Skill:
    """Build a :class:`Skill` for injection tests."""
    return Skill(
        name=name,
        description=description,
        tags=tags,
        body=body,
        category="test",
        path=None,  # type: ignore[arg-type]
    )


@pytest.fixture()
def store() -> MemoryStore:
    """An ephemeral in-memory store, closed at teardown."""
    s = MemoryStore(":memory:")
    try:
        yield s
    finally:
        s.close()


def _make_agent(store: MemoryStore, llm: FakeLLM, skills=None, **kwargs) -> Agent:
    """Build an agent wired to the fakes with a deterministic session id."""
    retriever = SkillRetriever(list(skills or []))
    return Agent(
        llm=llm,
        memory=store,
        skill_retriever=retriever,
        session_id="s-test",
        **kwargs,
    )


# --------------------------------------------------------------------------- #
# conversation.build_messages                                                  #
# --------------------------------------------------------------------------- #
def test_build_messages_order_and_defaults() -> None:
    msgs = conversation.build_messages("你好")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert conversation.DEFAULT_SYSTEM_PERSONA in msgs[0]["content"]
    assert msgs[-1] == {"role": "user", "content": "你好"}


def test_build_messages_includes_memory_and_skills() -> None:
    facts = [{"content": "用户喜欢简洁回答"}, {"content": "用户在做 Python"}]
    injection = "## 可用技能参考\n\n### demo\n说明"
    msgs = conversation.build_messages(
        "问题",
        history=[{"role": "user", "content": "旧", "created_at": "t"}],
        memory_facts=facts,
        skill_injection=injection,
        system_persona="自定义人格",
    )
    system = msgs[0]["content"]
    assert "自定义人格" in system
    assert conversation.MEMORY_BLOCK_HEADER in system
    assert "用户喜欢简洁回答" in system
    assert "可用技能参考" in system
    # History rows are reduced to role/content (created_at stripped).
    assert msgs[1] == {"role": "user", "content": "旧"}
    assert msgs[-1] == {"role": "user", "content": "问题"}


def test_build_messages_empty_sections_omitted() -> None:
    msgs = conversation.build_messages(
        "hi", memory_facts=[], skill_injection=""
    )
    system = msgs[0]["content"]
    assert conversation.MEMORY_BLOCK_HEADER not in system
    assert "可用技能参考" not in system


def test_format_memory_block_blank_facts() -> None:
    assert conversation.format_memory_block([]) == ""
    assert conversation.format_memory_block([{"content": "   "}]) == ""


# --------------------------------------------------------------------------- #
# Agent.run (non-streaming)                                                    #
# --------------------------------------------------------------------------- #
async def test_run_returns_full_text(store: MemoryStore) -> None:
    llm = FakeLLM(reply="这是完整回复")
    agent = _make_agent(store, llm)

    result = await agent.run("你好")

    assert result == "这是完整回复"
    assert len(llm.chat_calls) == 1


async def test_run_persists_user_and_assistant(store: MemoryStore) -> None:
    llm = FakeLLM(reply="回复A")
    agent = _make_agent(store, llm)

    await agent.run("消息A")

    history = store.get_history("s-test")
    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "消息A"),
        ("assistant", "回复A"),
    ]


async def test_run_uses_session_override(store: MemoryStore) -> None:
    llm = FakeLLM(reply="r")
    agent = _make_agent(store, llm)

    await agent.run("hi", session_id="other")

    assert store.get_history("other")  # stored under override
    assert store.get_history("s-test") == []  # default untouched


# --------------------------------------------------------------------------- #
# Multi-turn history carried into next request                                 #
# --------------------------------------------------------------------------- #
async def test_multiturn_history_carried_forward(store: MemoryStore) -> None:
    llm = FakeLLM(reply="第一轮回复")
    agent = _make_agent(store, llm)

    await agent.run("第一轮提问")
    llm.reply = "第二轮回复"
    await agent.run("第二轮提问")

    # Second request must include the full prior turn as history.
    second_messages = llm.chat_calls[1]
    roles = [m["role"] for m in second_messages]
    contents = [m["content"] for m in second_messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert contents[1] == "第一轮提问"
    assert contents[2] == "第一轮回复"
    assert contents[3] == "第二轮提问"


async def test_first_request_has_no_history(store: MemoryStore) -> None:
    llm = FakeLLM(reply="r")
    agent = _make_agent(store, llm)

    await agent.run("唯一提问")

    first = llm.chat_calls[0]
    # Only system + current user (current user msg not duplicated as history).
    assert [m["role"] for m in first] == ["system", "user"]
    assert first[-1]["content"] == "唯一提问"


# --------------------------------------------------------------------------- #
# Recall + skill injection reach the system message                            #
# --------------------------------------------------------------------------- #
async def test_recall_and_skills_injected_into_system(store: MemoryStore) -> None:
    store.add_fact("用户名叫 Jarvis", tags="profile", trust_score=0.9)
    skill = _skill(
        name="python-help",
        description="协助编写 Python 代码",
        tags=["python", "code"],
        body="遵循 PEP8。",
    )
    llm = FakeLLM(reply="ok")
    agent = _make_agent(store, llm, skills=[skill])

    # Query overlaps both the fact (via LIKE) and the skill (keyword overlap).
    await agent.run("python 用户 Jarvis 你好")

    system = llm.chat_calls[0][0]["content"]
    assert conversation.MEMORY_BLOCK_HEADER in system
    assert "用户名叫 Jarvis" in system
    assert "可用技能参考" in system
    assert "python-help" in system


# --------------------------------------------------------------------------- #
# Agent.run_stream (streaming)                                                 #
# --------------------------------------------------------------------------- #
async def test_run_stream_yields_deltas(store: MemoryStore) -> None:
    llm = FakeLLM(deltas=["Hel", "lo", "!"])
    agent = _make_agent(store, llm)

    collected = [chunk async for chunk in agent.run_stream("hi")]

    assert collected == ["Hel", "lo", "!"]


async def test_run_stream_persists_full_reply(store: MemoryStore) -> None:
    llm = FakeLLM(deltas=["A", "B", "C"])
    agent = _make_agent(store, llm)

    async for _ in agent.run_stream("提问"):
        pass

    history = store.get_history("s-test")
    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "提问"),
        ("assistant", "ABC"),
    ]


async def test_run_stream_persists_partial_on_error(store: MemoryStore) -> None:
    llm = FakeLLM(deltas=["部分1", "部分2"], stream_error=RuntimeError("boom"))
    agent = _make_agent(store, llm)

    collected: list[str] = []
    with pytest.raises(RuntimeError):
        async for chunk in agent.run_stream("提问"):
            collected.append(chunk)

    assert collected == ["部分1", "部分2"]
    history = store.get_history("s-test")
    # User message + whatever was generated before the failure.
    assert [(h["role"], h["content"]) for h in history] == [
        ("user", "提问"),
        ("assistant", "部分1部分2"),
    ]


async def test_run_stream_carries_history(store: MemoryStore) -> None:
    llm = FakeLLM(deltas=["一"])
    agent = _make_agent(store, llm)

    async for _ in agent.run_stream("第一轮"):
        pass
    llm.deltas = ["二"]
    async for _ in agent.run_stream("第二轮"):
        pass

    second = llm.stream_calls[1]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "user"]
    assert second[1]["content"] == "第一轮"
    assert second[2]["content"] == "一"
    assert second[3]["content"] == "第二轮"


# --------------------------------------------------------------------------- #
# Lifecycle                                                                    #
# --------------------------------------------------------------------------- #
async def test_aclose_releases_llm_and_memory() -> None:
    store = MemoryStore(":memory:")
    llm = FakeLLM()
    agent = _make_agent(store, llm)

    await agent.aclose()

    assert llm.closed is True
    # Memory connection is closed -> further use raises.
    with pytest.raises(Exception):
        store.add_message("s", "user", "x")


async def test_async_context_manager() -> None:
    llm = FakeLLM(reply="ctx")
    store = MemoryStore(":memory:")
    async with _make_agent(store, llm) as agent:
        assert await agent.run("hi") == "ctx"
    assert llm.closed is True


def test_default_session_id_is_uuid_hex() -> None:
    store = MemoryStore(":memory:")
    try:
        agent = Agent(
            llm=FakeLLM(),
            memory=store,
            skill_retriever=SkillRetriever([]),
        )
        assert isinstance(agent.session_id, str)
        assert len(agent.session_id) == 32  # uuid4().hex
    finally:
        store.close()
