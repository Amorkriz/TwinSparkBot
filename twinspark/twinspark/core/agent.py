"""The TwinSpark agent core: orchestration of LLM, memory, skills and tools.

:class:`Agent` ties the four already-built subsystems together into a single
conversational loop:

* **LLM** (:class:`~twinspark.core.llm.LLMClient`) — text generation.
* **Memory** (:class:`~twinspark.memory.store.MemoryStore`) — session history
  and durable facts.
* **Skills** (:class:`~twinspark.skills.retriever.SkillRetriever`) — passive
  guidance injected into the system prompt.
* **Tools** (:data:`twinspark.tools.registry.registry`) — executable
  capabilities (currently none are registered; see the tool-loop TODOs below).

A turn flows as follows:

1. Ensure the session exists.
2. Recall relevant facts and retrieve relevant skills for the user message.
3. Read prior history and assemble the message list
   (:func:`twinspark.core.conversation.build_messages`).
4. Persist the user message.
5. Generate a reply (non-streaming :meth:`run` or streaming :meth:`run_stream`).
6. Persist the assistant reply (best-effort, even on interruption).

Two entry points are provided for clarity:

* :meth:`run` → ``str`` (full reply, non-streaming).
* :meth:`run_stream` → ``AsyncIterator[str]`` (text deltas, streaming).
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any, Optional

from twinspark.core import conversation
from twinspark.core.llm import LLMClient
from twinspark.memory.store import MemoryStore
from twinspark.skills.loader import SkillLoader
from twinspark.skills.retriever import SkillRetriever
from twinspark.tools.registry import registry as default_tool_registry

__all__ = ["Agent"]

logger = logging.getLogger(__name__)

#: Safety cap on the tool-dispatch loop so a misbehaving model cannot spin
#: forever. With no tools registered today only a single iteration ever runs.
DEFAULT_MAX_TOOL_ROUNDS = 5


class Agent:
    """Conversational agent wiring LLM + memory + skills (+ future tools).

    All collaborators are injectable, which keeps the agent fully unit-testable
    with fakes (see ``tests/test_agent.py``). When omitted they default to a
    freshly constructed real instance.

    Args:
        llm: The LLM client. Defaults to a new :class:`LLMClient`.
        memory: The memory store. Defaults to a new :class:`MemoryStore`.
        skill_retriever: The skill retriever. Defaults to a
            :class:`SkillRetriever` backed by a default :class:`SkillLoader`.
        session_id: Default session id for turns that do not pass one
            explicitly. Defaults to a fresh ``uuid4().hex``.
        system_persona: Optional base persona override. Defaults to
            :data:`twinspark.core.conversation.DEFAULT_SYSTEM_PERSONA`.
        tool_registry: The tool registry consulted for function-calling
            schemas. Defaults to the module-level singleton.
        max_tool_rounds: Upper bound on tool-dispatch iterations per turn.
    """

    def __init__(
        self,
        *,
        llm: Optional[LLMClient] = None,
        memory: Optional[MemoryStore] = None,
        skill_retriever: Optional[SkillRetriever] = None,
        session_id: Optional[str] = None,
        system_persona: Optional[str] = None,
        tool_registry: Any = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
    ) -> None:
        self.llm = llm if llm is not None else LLMClient()
        self.memory = memory if memory is not None else MemoryStore()
        self.skill_retriever = (
            skill_retriever
            if skill_retriever is not None
            else SkillRetriever(SkillLoader())
        )
        self.session_id = session_id or uuid.uuid4().hex
        self.system_persona = system_persona
        self.tools = (
            tool_registry if tool_registry is not None else default_tool_registry
        )
        self.max_tool_rounds = max(1, int(max_tool_rounds))

    # ------------------------------------------------------------------ #
    # Turn preparation
    # ------------------------------------------------------------------ #
    def _resolve_session(self, session_id: Optional[str]) -> str:
        """Return the effective session id and ensure the row exists."""
        sid = session_id or self.session_id
        self.memory.ensure_session(sid)
        return sid

    def _build_turn_messages(self, user_msg: str, sid: str) -> list[dict[str, str]]:
        """Gather memory + skills + history and assemble the chat messages.

        History is read *before* the current user message is persisted, so the
        assembled list contains each turn exactly once (``build_messages``
        appends the current ``user_msg`` itself).
        """
        memory_facts = self.memory.recall(user_msg)
        skill_hits = self.skill_retriever.retrieve(user_msg)
        skill_injection = self.skill_retriever.build_injection_text(skill_hits)
        history = self.memory.get_history(sid)

        return conversation.build_messages(
            user_msg,
            history=history,
            memory_facts=memory_facts,
            skill_injection=skill_injection,
            system_persona=self.system_persona,
        )

    # ------------------------------------------------------------------ #
    # Non-streaming turn
    # ------------------------------------------------------------------ #
    async def run(
        self,
        user_msg: str,
        *,
        session_id: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> str:
        """Run a single non-streaming turn and return the full reply.

        The user message is persisted before the request; the assistant reply
        is persisted afterwards. If generation raises, whatever text was
        produced so far is still persisted (see the ``finally`` block).

        Args:
            user_msg: The user's input for this turn.
            session_id: Session to use; falls back to ``self.session_id``.
            **llm_kwargs: Extra parameters forwarded to
                :meth:`LLMClient.chat` (e.g. ``temperature``, ``model``).

        Returns:
            The assistant's full reply text (possibly empty).
        """
        sid = self._resolve_session(session_id)
        messages = self._build_turn_messages(user_msg, sid)
        self.memory.add_message(sid, "user", user_msg)

        reply = ""
        try:
            reply = await self._generate(messages, **llm_kwargs)
        finally:
            if reply:
                self.memory.add_message(sid, "assistant", reply)
        return reply

    async def _generate(
        self, messages: list[dict[str, str]], **llm_kwargs: Any
    ) -> str:
        """文本生成循环，支持工具调用分派。

        当 ToolRegistry 中有工具注册时，将工具 schema 传递给模型，
        并在模型请求工具调用时执行工具、收集结果、继续对话。
        无工具注册时行为与此前完全一致（单轮文本生成）。
        """
        schemas = self.tools.get_openai_schemas()
        call_kwargs = dict(llm_kwargs)
        if schemas:
            # 存在工具时才向模型广告 schema，保持无工具场景的调用参数不变
            call_kwargs.setdefault("tools", schemas)

        reply = ""
        for _round in range(self.max_tool_rounds):
            # 无工具注册时，直接使用 chat() 获取纯文本回复，保持向后兼容
            if not schemas:
                reply = await self.llm.chat(messages, **call_kwargs)
                break

            # 有工具时使用 chat_raw() 获取完整消息对象（含 tool_calls）
            raw_msg = await self.llm.chat_raw(messages, **call_kwargs)
            if raw_msg is None:
                break

            reply = raw_msg.content or ""
            tool_calls = raw_msg.tool_calls

            # 模型未请求工具调用 → 返回文本结束循环
            if not tool_calls:
                break

            # 追加 assistant 消息（含 tool_calls 信息）到对话历史
            # 严格遵循 OpenAI 的 function-calling 消息格式
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if raw_msg.content:
                assistant_msg["content"] = raw_msg.content
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
            messages.append(assistant_msg)

            # 并发执行所有工具调用；return_exceptions=True 保证单个工具失败
            # 不会中断整个循环，异常将作为结果单独处理
            import asyncio

            results = await asyncio.gather(
                *[self._execute_tool_call(tc) for tc in tool_calls],
                return_exceptions=True,
            )

            # 逐个追加工具执行结果到对话历史，供模型下一轮消费
            for tc, result in zip(tool_calls, results):
                if isinstance(result, Exception):
                    content = f"执行错误: {result}"
                else:
                    content = str(result)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": content,
                    }
                )
            # 继续 for 循环，让模型对工具结果做出反应

        return reply

    async def _execute_tool_call(self, tool_call: Any) -> str:
        """执行单个工具调用，返回结果字符串。

        该方法负责：
        1. 从 ToolRegistry 中查找工具
        2. 解析 JSON 参数
        3. 根据工具的 is_async 标志选择同步或异步调用
        4. 捕获所有异常，确保不会因单个工具失败而中断整个循环

        Args:
            tool_call: OpenAI 格式的 tool_call 对象，
                       含 ``.function.name`` 和 ``.function.arguments``。

        Returns:
            工具执行结果的字符串表示；失败时返回可读的错误说明。
        """
        import json

        name = tool_call.function.name
        tool = self.tools.get(name)
        if tool is None:
            return f"错误：工具 '{name}' 未注册"

        # 解析工具调用参数（模型输出的 arguments 是 JSON 字符串）
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError as e:
            return f"参数解析错误: {e}"

        # 执行工具，区分同步/异步实现
        try:
            if tool.is_async:
                result = await tool.run(**args)
            else:
                result = tool.run(**args)
            return str(result) if result is not None else ""
        except Exception as e:  # noqa: BLE001 - 工具异常不应打断整体循环
            logger.warning(
                "工具 '%s' 执行异常: %s", name, e, exc_info=True
            )
            return f"工具执行异常: {type(e).__name__}: {e}"

    # ------------------------------------------------------------------ #
    # Streaming turn
    # ------------------------------------------------------------------ #
    async def run_stream(
        self,
        user_msg: str,
        *,
        session_id: Optional[str] = None,
        **llm_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Run a single streaming turn, yielding text deltas as they arrive.

        Deltas are accumulated and the full reply is persisted once streaming
        finishes. If the consumer stops early or the stream errors mid-way, the
        portion generated so far is still persisted (``try/finally``).

        Note:
            The tool-dispatch skeleton in :meth:`_generate` is intentionally
            *not* mirrored here: streaming tool-calls require accumulating and
            parsing ``tool_calls`` deltas, which is deferred until tools exist.
            With no tools registered a streaming turn is always a single pass.

        Args:
            user_msg: The user's input for this turn.
            session_id: Session to use; falls back to ``self.session_id``.
            **llm_kwargs: Extra parameters forwarded to
                :meth:`LLMClient.stream`.

        Yields:
            Non-empty text increments from the model.
        """
        sid = self._resolve_session(session_id)
        messages = self._build_turn_messages(user_msg, sid)
        self.memory.add_message(sid, "user", user_msg)

        parts: list[str] = []
        try:
            async for delta in self.llm.stream(messages, **llm_kwargs):
                if delta:
                    parts.append(delta)
                    yield delta
        finally:
            full = "".join(parts)
            if full:
                self.memory.add_message(sid, "assistant", full)

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Release the memory store (synchronous resources).

        The LLM client owns async resources; prefer :meth:`aclose` when running
        inside an event loop. This synchronous variant is convenient for CLI
        teardown where only the DB connection needs closing.
        """
        try:
            self.memory.close()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.debug("Error while closing memory store", exc_info=True)

    async def aclose(self) -> None:
        """Release both the LLM client and the memory store."""
        try:
            await self.llm.aclose()
        except Exception:  # noqa: BLE001 - best-effort teardown
            logger.debug("Error while closing LLM client", exc_info=True)
        self.close()

    async def __aenter__(self) -> "Agent":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()
