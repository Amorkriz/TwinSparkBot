"""Message assembly for the TwinSpark agent core.

This module is the single place that decides *how* a chat request is laid out
before it is handed to :class:`~twinspark.core.llm.LLMClient`. Keeping the
assembly logic here (rather than inside :class:`~twinspark.core.agent.Agent`)
makes it trivially unit-testable and keeps the agent focused on orchestration.

The resulting message list follows the classic OpenAI chat shape::

    [
        {"role": "system",    "content": <persona + memory + skills>},
        {"role": "user",      "content": <older turn>},
        {"role": "assistant", "content": <older turn>},
        ...
        {"role": "user",      "content": <current user message>},
    ]

The ``system`` message is built from three optional sections, in order:

1. the base persona (always present),
2. a *relevant memory* block rendered from :meth:`MemoryStore.recall` results,
3. a *skill injection* block produced by
   :meth:`~twinspark.skills.retriever.SkillRetriever.build_injection_text`.

Empty sections are omitted so the model never sees dangling headers.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

__all__ = [
    "DEFAULT_SYSTEM_PERSONA",
    "MEMORY_BLOCK_HEADER",
    "format_memory_block",
    "build_system_content",
    "build_messages",
]

#: Simple, reliable default persona used when the caller does not supply one.
DEFAULT_SYSTEM_PERSONA = "你是 TwinSpark,一个简洁、可靠的 AI 助手。"

#: Heading for the injected "relevant memory" block.
MEMORY_BLOCK_HEADER = "## 相关记忆"


def format_memory_block(
    facts: Optional[Sequence[Mapping[str, Any]]],
) -> str:
    """Render recalled facts into a Markdown block for the system prompt.

    Args:
        facts: The list of fact dicts returned by
            :meth:`~twinspark.memory.store.MemoryStore.recall`. Each item is
            expected to carry a ``"content"`` key; other keys are ignored.
            ``None`` or an empty sequence yields an empty string.

    Returns:
        A block like::

            ## 相关记忆
            - fact one
            - fact two

        or an empty string when there is nothing worth injecting.
    """
    if not facts:
        return ""

    lines = [MEMORY_BLOCK_HEADER]
    appended = False
    for fact in facts:
        content = str(fact.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"- {content}")
        appended = True

    # If every fact was blank, there is nothing meaningful to inject.
    if not appended:
        return ""
    return "\n".join(lines)


def build_system_content(
    *,
    system_persona: Optional[str] = None,
    memory_facts: Optional[Sequence[Mapping[str, Any]]] = None,
    skill_injection: Optional[str] = None,
) -> str:
    """Assemble the full ``system`` message content.

    Sections are joined with a blank line and empty sections are skipped, so
    the model never receives an empty memory/skill header.

    Args:
        system_persona: Base persona text. Falls back to
            :data:`DEFAULT_SYSTEM_PERSONA` when ``None`` or blank.
        memory_facts: Facts from ``MemoryStore.recall`` (rendered via
            :func:`format_memory_block`).
        skill_injection: Pre-rendered skill block from
            :meth:`SkillRetriever.build_injection_text` (already Markdown, or
            an empty string).

    Returns:
        The concatenated system-prompt string.
    """
    persona = (system_persona or "").strip() or DEFAULT_SYSTEM_PERSONA
    sections: list[str] = [persona]

    memory_block = format_memory_block(memory_facts)
    if memory_block:
        sections.append(memory_block)

    if skill_injection and skill_injection.strip():
        sections.append(skill_injection.strip())

    return "\n\n".join(sections)


def _normalize_history(
    history: Optional[Sequence[Mapping[str, Any]]],
) -> list[dict[str, str]]:
    """Reduce stored history rows to ``{"role", "content"}`` chat messages.

    ``MemoryStore.get_history`` returns rows carrying an extra ``created_at``
    field; the chat API only wants ``role`` and ``content``. Rows missing
    either field, or carrying a non-conversational role, are skipped
    defensively so a malformed row can never break a request.
    """
    if not history:
        return []

    messages: list[dict[str, str]] = []
    for row in history:
        role = row.get("role")
        content = row.get("content")
        if not role or content is None:
            continue
        messages.append({"role": str(role), "content": str(content)})
    return messages


def build_messages(
    user_msg: str,
    *,
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    memory_facts: Optional[Sequence[Mapping[str, Any]]] = None,
    skill_injection: Optional[str] = None,
    system_persona: Optional[str] = None,
) -> list[dict[str, str]]:
    """Assemble the full chat message list for a single turn.

    Order: ``system`` (persona + memory + skills) → prior ``history`` → the
    current ``user`` message.

    Args:
        user_msg: The current user input. Appended last as a ``user`` message.
        history: Prior conversation turns (oldest first) as returned by
            :meth:`~twinspark.memory.store.MemoryStore.get_history`. This must
            *not* already include ``user_msg``.
        memory_facts: Relevant facts from ``MemoryStore.recall``.
        skill_injection: Rendered skill block (may be empty).
        system_persona: Optional persona override.

    Returns:
        A list of ``{"role", "content"}`` dicts ready to send to the LLM.
    """
    system_content = build_system_content(
        system_persona=system_persona,
        memory_facts=memory_facts,
        skill_injection=skill_injection,
    )

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content}
    ]
    messages.extend(_normalize_history(history))
    messages.append({"role": "user", "content": user_msg})
    return messages
