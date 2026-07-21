"""Agent 工具调用循环的集成测试。

基于 test_agent.py 中 FakeLLM 的模式，扩展出 FakeLLMWithTools 类来模拟
模型返回 tool_calls 消息。测试覆盖了 Agent._generate() 的工具分派循环，
包括单/多工具调用、错误处理、轮次上限等场景。

所有测试完全自包含，不依赖外部服务。
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

from dataclasses import dataclass, field
from typing import Any

import pytest

from twinspark.core.agent import Agent
from twinspark.memory.store import MemoryStore
from twinspark.skills.retriever import SkillRetriever
from twinspark.tools import Tool, ToolRegistry


# --------------------------------------------------------------------------- #
# Fake 数据结构：模拟 OpenAI 的 tool_call 消息格式
# --------------------------------------------------------------------------- #
@dataclass
class FakeFunction:
    """模拟 ChatCompletionMessageToolCall 中的 function 字段。"""

    name: str
    arguments: str  # JSON 字符串


@dataclass
class FakeToolCall:
    """模拟 ChatCompletionMessageToolCall 结构。"""

    id: str
    type: str = "function"
    function: FakeFunction | None = None


@dataclass
class FakeMessage:
    """模拟 chat_raw() 返回的完整消息对象。"""

    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


# --------------------------------------------------------------------------- #
# FakeLLMWithTools：支持工具调用场景的模拟 LLM
# --------------------------------------------------------------------------- #
class FakeLLMWithTools:
    """模拟带工具调用的 LLM 客户端。

    responses 按调用顺序依次返回。每个响应可以是：
    - str: 纯文本回复（chat() 使用，或作为 chat_raw 返回无 tool_calls 的消息）
    - dict: {"content": "...", "tool_calls": [...]} 格式（chat_raw 返回带工具调用的消息）
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self._call_index = 0
        self.chat_calls: list[list[dict]] = []
        self.chat_raw_calls: list[list[dict]] = []
        self.closed = False

    async def chat(self, messages: list[dict], **kwargs: Any) -> str:
        """无工具场景使用的纯文本接口。"""
        self.chat_calls.append([dict(m) for m in messages])
        resp = self._responses[self._call_index]
        self._call_index += 1
        return resp if isinstance(resp, str) else resp.get("content", "")

    async def chat_raw(self, messages: list[dict], **kwargs: Any) -> FakeMessage:
        """有工具场景使用的原始消息接口，返回 FakeMessage 对象。"""
        self.chat_raw_calls.append([dict(m) for m in messages])
        resp = self._responses[self._call_index]
        self._call_index += 1

        if isinstance(resp, str):
            # 纯文本 → 无 tool_calls
            return FakeMessage(content=resp, tool_calls=None)

        # dict 格式 → 构建带 tool_calls 的 FakeMessage
        content = resp.get("content")
        raw_calls = resp.get("tool_calls", [])
        tool_calls = []
        for tc in raw_calls:
            tool_calls.append(
                FakeToolCall(
                    id=tc["id"],
                    function=FakeFunction(
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    ),
                )
            )
        return FakeMessage(content=content, tool_calls=tool_calls or None)

    async def stream(self, messages: list[dict], **kwargs: Any):
        """流式接口（本测试文件不使用，留作占位）。"""
        yield ""

    async def aclose(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# 测试用工具
# --------------------------------------------------------------------------- #
class AddTool(Tool):
    """一个简单的同步加法工具，用于测试。"""

    name = "add"
    description = "将两个数字相加"
    parameters = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }
    is_async = False

    def run(self, **kwargs):
        return str(kwargs["a"] + kwargs["b"])


class AsyncEchoTool(Tool):
    """一个异步 echo 工具，返回输入的消息。"""

    name = "echo"
    description = "回显消息"
    parameters = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }
    is_async = True

    async def run(self, **kwargs):
        return kwargs.get("message", "")


class ExplodingTool(Tool):
    """一个总是抛出异常的工具，用于测试错误处理。"""

    name = "explode"
    description = "爆炸"
    is_async = False

    def run(self, **kwargs):
        raise RuntimeError("工具爆炸了！")


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def store() -> MemoryStore:
    """临时内存 store。"""
    s = MemoryStore(":memory:")
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def tool_registry() -> ToolRegistry:
    """预注册了测试工具的 ToolRegistry。"""
    reg = ToolRegistry()
    reg.register(AddTool())
    reg.register(AsyncEchoTool())
    reg.register(ExplodingTool())
    return reg


@pytest.fixture()
def empty_registry() -> ToolRegistry:
    """空的 ToolRegistry，无任何工具注册。"""
    return ToolRegistry()


def _make_agent(
    store: MemoryStore,
    llm: FakeLLMWithTools,
    registry: ToolRegistry | None = None,
    max_tool_rounds: int = 5,
) -> Agent:
    """构建注入了 Fakes 的 Agent 实例。"""
    return Agent(
        llm=llm,
        memory=store,
        skill_retriever=SkillRetriever([]),
        session_id="s-tool-test",
        tool_registry=registry or ToolRegistry(),
        max_tool_rounds=max_tool_rounds,
    )


# --------------------------------------------------------------------------- #
# 测试用例
# --------------------------------------------------------------------------- #
class TestAgentNoTools:
    """无工具注册时的 Agent 行为测试。"""

    async def test_agent_no_tools_unchanged(self, store: MemoryStore, empty_registry: ToolRegistry):
        """无工具时行为不变：使用 chat() 而非 chat_raw()。"""
        llm = FakeLLMWithTools(responses=["你好，我是助手"])
        agent = _make_agent(store, llm, registry=empty_registry)

        result = await agent.run("你好")

        assert result == "你好，我是助手"
        # 无工具时应调用 chat() 而非 chat_raw()
        assert len(llm.chat_calls) == 1
        assert len(llm.chat_raw_calls) == 0


class TestAgentSingleToolCall:
    """单个工具调用场景测试。"""

    async def test_agent_single_tool_call(self, store: MemoryStore, tool_registry: ToolRegistry):
        """模型请求一个工具调用 → 执行 → 模型消费结果后返回最终文本。"""
        llm = FakeLLMWithTools(
            responses=[
                # 第一轮：模型请求调用 add 工具
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_001",
                            "function": {
                                "name": "add",
                                "arguments": json.dumps({"a": 3, "b": 5}),
                            },
                        }
                    ],
                },
                # 第二轮：模型看到工具结果后返回最终文本
                "3 加 5 等于 8",
            ]
        )
        agent = _make_agent(store, llm, registry=tool_registry)

        result = await agent.run("请计算 3+5")

        assert result == "3 加 5 等于 8"
        # chat_raw 被调用了两次（请求工具 + 最终回复）
        assert len(llm.chat_raw_calls) == 2

    async def test_agent_async_tool_call(self, store: MemoryStore, tool_registry: ToolRegistry):
        """异步工具（echo）也能被正确调用。"""
        llm = FakeLLMWithTools(
            responses=[
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_echo",
                            "function": {
                                "name": "echo",
                                "arguments": json.dumps({"message": "你好世界"}),
                            },
                        }
                    ],
                },
                "echo 返回: 你好世界",
            ]
        )
        agent = _make_agent(store, llm, registry=tool_registry)

        result = await agent.run("echo 你好世界")

        assert result == "echo 返回: 你好世界"


class TestAgentMultipleToolCalls:
    """多个并发工具调用场景测试。"""

    async def test_agent_multiple_tool_calls(self, store: MemoryStore, tool_registry: ToolRegistry):
        """模型在一次响应中请求多个工具调用，全部并发执行后返回结果。"""
        llm = FakeLLMWithTools(
            responses=[
                # 第一轮：模型同时请求 add 和 echo 两个工具
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_add",
                            "function": {
                                "name": "add",
                                "arguments": json.dumps({"a": 10, "b": 20}),
                            },
                        },
                        {
                            "id": "call_echo",
                            "function": {
                                "name": "echo",
                                "arguments": json.dumps({"message": "hello"}),
                            },
                        },
                    ],
                },
                # 第二轮：模型消费所有工具结果后返回
                "加法结果是 30，echo 返回了 hello",
            ]
        )
        agent = _make_agent(store, llm, registry=tool_registry)

        result = await agent.run("做两个计算")

        assert result == "加法结果是 30，echo 返回了 hello"

        # 验证第二轮消息中包含了两个 tool role 消息
        second_call_messages = llm.chat_raw_calls[1]
        tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
        assert len(tool_messages) == 2
        # 验证工具结果内容
        results_content = {m["content"] for m in tool_messages}
        assert "30" in results_content
        assert "hello" in results_content


class TestAgentToolErrors:
    """工具调用错误处理测试。"""

    async def test_agent_tool_not_found(self, store: MemoryStore, tool_registry: ToolRegistry):
        """请求不存在的工具：返回错误消息给模型，模型据此回复。"""
        llm = FakeLLMWithTools(
            responses=[
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_missing",
                            "function": {
                                "name": "nonexistent_tool",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "抱歉，该工具不存在",
            ]
        )
        agent = _make_agent(store, llm, registry=tool_registry)

        result = await agent.run("用不存在的工具")

        assert result == "抱歉，该工具不存在"
        # 验证工具结果消息中包含"未注册"的错误提示
        second_messages = llm.chat_raw_calls[1]
        tool_msg = next(m for m in second_messages if m.get("role") == "tool")
        assert "未注册" in tool_msg["content"]

    async def test_agent_tool_exception(self, store: MemoryStore, tool_registry: ToolRegistry):
        """工具执行抛出异常：异常被捕获，错误信息作为结果返回给模型。"""
        llm = FakeLLMWithTools(
            responses=[
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_boom",
                            "function": {
                                "name": "explode",
                                "arguments": "{}",
                            },
                        }
                    ],
                },
                "工具执行失败了",
            ]
        )
        agent = _make_agent(store, llm, registry=tool_registry)

        result = await agent.run("执行爆炸工具")

        assert result == "工具执行失败了"
        # 验证错误信息被包含在 tool 消息中
        second_messages = llm.chat_raw_calls[1]
        tool_msg = next(m for m in second_messages if m.get("role") == "tool")
        assert "异常" in tool_msg["content"] or "错误" in tool_msg["content"]
        assert "爆炸" in tool_msg["content"]

    async def test_agent_tool_invalid_json_args(self, store: MemoryStore, tool_registry: ToolRegistry):
        """工具参数 JSON 解析失败：返回参数解析错误信息给模型。"""
        llm = FakeLLMWithTools(
            responses=[
                {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_bad_json",
                            "function": {
                                "name": "add",
                                "arguments": "{invalid json!!!",  # 无效 JSON
                            },
                        }
                    ],
                },
                "参数格式不正确，我重新尝试",
            ]
        )
        agent = _make_agent(store, llm, registry=tool_registry)

        result = await agent.run("尝试错误参数")

        assert result == "参数格式不正确，我重新尝试"
        # 验证工具结果中包含"参数解析错误"
        second_messages = llm.chat_raw_calls[1]
        tool_msg = next(m for m in second_messages if m.get("role") == "tool")
        assert "参数解析错误" in tool_msg["content"]


class TestAgentToolRoundsLimit:
    """工具调用轮次限制测试。"""

    async def test_agent_max_rounds_limit(self, store: MemoryStore, tool_registry: ToolRegistry):
        """验证 max_tool_rounds 防止无限循环：到达上限后强制退出。"""
        # 模型每轮都请求工具调用，永不停止
        infinite_tool_calls = [
            {
                "content": None,
                "tool_calls": [
                    {
                        "id": f"call_{i}",
                        "function": {
                            "name": "add",
                            "arguments": json.dumps({"a": 1, "b": 1}),
                        },
                    }
                ],
            }
            for i in range(10)  # 准备足够多的响应
        ]

        llm = FakeLLMWithTools(responses=infinite_tool_calls)
        # 设置 max_tool_rounds=3，期望最多循环 3 次
        agent = _make_agent(store, llm, registry=tool_registry, max_tool_rounds=3)

        result = await agent.run("无限调用")

        # chat_raw 被调用的次数不超过 max_tool_rounds
        assert len(llm.chat_raw_calls) <= 3
        # 最后返回的 content 可能为空（因为模型持续返回 tool_calls）
        # 但不应死循环
