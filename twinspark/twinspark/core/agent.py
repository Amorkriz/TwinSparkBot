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
        """Text-generation loop with a placeholder tool-dispatch skeleton.

        Today ``registry.get_openai_schemas()`` returns ``[]``, so no tool
        schemas are advertised, the model never emits ``tool_calls``, and the
        loop makes exactly one text-generation call before returning.

        The loop below is structured so tool support can be dropped in without
        reshaping the control flow — see the TODOs.
        """
        schemas = self.tools.get_openai_schemas()
        call_kwargs = dict(llm_kwargs)
        if schemas:
            # When tools exist, advertise them to the model.
            call_kwargs.setdefault("tools", schemas)

        reply = ""
        for _round in range(self.max_tool_rounds):
            reply = await self.llm.chat(messages, **call_kwargs)

            # TODO(tools): The current LLMClient.chat() returns only the reply
            # text, not the raw message (which would carry ``tool_calls``).
            # When tools are introduced:
            #   1. Surface the assistant message incl. ``tool_calls``.
            #   2. If tool_calls present: append the assistant message, then for
            #      each call resolve ``self.tools.get(name)``, invoke it
            #      (``await tool.run(**args)`` when ``tool.is_async`` else
            #      ``tool.run(**args)``), append a {"role": "tool", ...} result
            #      message, and ``continue`` the loop to let the model react.
            #   3. Otherwise ``break`` with the final text (current behaviour).
            #
            # No tools registered -> no tool_calls -> single text turn.
            break

        return reply

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
