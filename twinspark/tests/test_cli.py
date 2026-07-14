"""Unit tests for the TwinSpark CLI's pure helpers and command dispatch.

The interactive REPL itself depends on a TTY and is not exercised here.
Instead we cover:

* :func:`parse_command` — slash-command parsing edge cases,
* the formatting helpers (``format_skills`` / ``format_memory_results`` /
  ``format_history_results``) for empty and non-empty inputs,
* ``help_text`` / ``welcome_text`` / ``config_error_message`` content,
* :func:`_handle_command` dispatch against a lightweight fake agent (no
  network, no real Agent construction for most branches).
"""

from __future__ import annotations

import os

# Provide a fake key so importing config-dependent modules never fails.
os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-not-real")

from typing import Any

import pytest

from twinspark import cli


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class FakeMemory:
    """Records calls and returns scripted recall/history results."""

    def __init__(
        self,
        *,
        recall_result: list[dict] | None = None,
        history_result: list[dict] | None = None,
    ) -> None:
        self.recall_result = recall_result if recall_result is not None else []
        self.history_result = history_result if history_result is not None else []
        self.recall_calls: list[str] = []
        self.history_calls: list[str] = []

    def recall(self, query: str, limit: int = 5) -> list[dict]:
        self.recall_calls.append(query)
        return self.recall_result

    def search_history(self, query: str, limit: int = 10) -> list[dict]:
        self.history_calls.append(query)
        return self.history_result


class FakeAgent:
    """Minimal stand-in for :class:`Agent` used by command-dispatch tests."""

    def __init__(self, session_id: str = "abcdef1234567890") -> None:
        self.session_id = session_id
        self.memory = FakeMemory()
        self.closed = False

    async def aclose(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# parse_command                                                                #
# --------------------------------------------------------------------------- #
def test_parse_command_returns_none_for_plain_text() -> None:
    assert cli.parse_command("hello world") is None
    assert cli.parse_command("  just a message  ") is None


def test_parse_command_simple() -> None:
    assert cli.parse_command("/help") == ("help", "")
    assert cli.parse_command("/quit") == ("quit", "")


def test_parse_command_case_insensitive_command() -> None:
    assert cli.parse_command("/HELP") == ("help", "")


def test_parse_command_with_argument() -> None:
    assert cli.parse_command("/memory python tips") == ("memory", "python tips")
    # Extra surrounding whitespace is trimmed from the argument.
    assert cli.parse_command("/history   foo bar  ") == ("history", "foo bar")


def test_parse_command_leading_whitespace() -> None:
    assert cli.parse_command("   /skills") == ("skills", "")


def test_parse_command_lone_slash() -> None:
    assert cli.parse_command("/") == ("", "")


# --------------------------------------------------------------------------- #
# Formatting helpers                                                           #
# --------------------------------------------------------------------------- #
def test_format_skills_empty() -> None:
    text = cli.format_skills([])
    assert "No skills" in text


def test_format_skills_non_empty() -> None:
    skills = [
        {
            "name": "example",
            "description": "does a thing",
            "category": "general",
            "tags": ["a", "b"],
        }
    ]
    text = cli.format_skills(skills)
    assert "general/example" in text
    assert "does a thing" in text
    assert "a, b" in text


def test_format_memory_results_empty() -> None:
    assert "No relevant memories" in cli.format_memory_results([])


def test_format_memory_results_non_empty() -> None:
    facts = [{"content": "user likes tea", "trust_score": 0.9, "tags": "pref"}]
    text = cli.format_memory_results(facts)
    assert "user likes tea" in text
    assert "trust=0.9" in text


def test_format_history_results_empty() -> None:
    assert "No matching history" in cli.format_history_results([])


def test_format_history_results_non_empty() -> None:
    msgs = [{"role": "user", "content": "hi", "created_at": "2026-01-01"}]
    text = cli.format_history_results(msgs)
    assert "user: hi" in text
    assert "2026-01-01" in text


def test_help_text_lists_commands() -> None:
    text = cli.help_text()
    for token in ["/new", "/reset", "/skills", "/memory", "/history", "/help", "/quit"]:
        assert token in text


def test_welcome_text_shows_model_and_session() -> None:
    text = cli.welcome_text("qwen-plus", "abcdef1234567890")
    assert "qwen-plus" in text
    assert "abcdef12" in text  # short session prefix


def test_config_error_message_mentions_key() -> None:
    text = cli.config_error_message(ValueError("boom"))
    assert "DASHSCOPE_API_KEY" in text
    assert ".env" in text


# --------------------------------------------------------------------------- #
# Command dispatch                                                             #
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_handle_help_returns_none(capsys: Any) -> None:
    agent = FakeAgent()
    result = await cli._handle_command(agent, "help", "")
    assert result is None
    assert "Commands:" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_handle_quit_raises_exit() -> None:
    agent = FakeAgent()
    with pytest.raises(cli._ExitRepl):
        await cli._handle_command(agent, "quit", "")


@pytest.mark.asyncio
async def test_handle_memory_calls_recall(capsys: Any) -> None:
    agent = FakeAgent()
    agent.memory.recall_result = [{"content": "fact one"}]
    result = await cli._handle_command(agent, "memory", "topic")
    assert result is None
    assert agent.memory.recall_calls == ["topic"]
    assert "fact one" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_handle_memory_without_argument(capsys: Any) -> None:
    agent = FakeAgent()
    await cli._handle_command(agent, "memory", "")
    assert agent.memory.recall_calls == []
    assert "Usage: /memory" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_handle_history_calls_search(capsys: Any) -> None:
    agent = FakeAgent()
    agent.memory.history_result = [{"role": "assistant", "content": "prev"}]
    await cli._handle_command(agent, "history", "prev")
    assert agent.memory.history_calls == ["prev"]
    assert "prev" in capsys.readouterr().out


@pytest.mark.asyncio
async def test_handle_unknown_command(capsys: Any) -> None:
    agent = FakeAgent()
    result = await cli._handle_command(agent, "bogus", "")
    assert result is None
    assert "Unknown command" in capsys.readouterr().out
