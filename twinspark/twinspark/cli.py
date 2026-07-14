"""Interactive command-line / TUI entry point for TwinSpark.

This module implements an asynchronous REPL built on
:mod:`prompt_toolkit`. It drives a single :class:`~twinspark.core.agent.Agent`
instance, streams replies with a typewriter effect, and exposes a small set of
slash commands for session management and introspection.

Design highlights
------------------
* **Async REPL** — :class:`prompt_toolkit.PromptSession` with
  :meth:`prompt_async` reads user input without blocking the event loop.
* **Streaming render** — replies from :meth:`Agent.run_stream` are printed
  incrementally (chunk-by-chunk) for a live typewriter feel.
* **Slash commands** — ``/new`` ``/reset`` ``/skills`` ``/memory`` ``/history``
  ``/help`` ``/quit`` ``/exit``.
* **Persistent history** — command history is saved to
  ``~/.twinspark/history`` via :class:`~prompt_toolkit.history.FileHistory`.
* **Graceful signals** — ``Ctrl+C`` interrupts the current generation (or a
  blank prompt) and returns to the prompt; ``Ctrl+D`` / EOF exits cleanly.
* **Friendly config errors** — a missing ``DASHSCOPE_API_KEY`` yields an
  actionable message instead of a raw traceback.

The pure formatting/parsing helpers (``parse_command``, ``help_text``,
``format_skills``, ``format_memory_results``, ``format_history_results``) carry
no I/O so they can be unit-tested without a TTY or a live agent.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout

from twinspark.core.agent import Agent

__all__ = ["main", "main_async"]

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
APP_NAME = "TwinSpark"

#: Where prompt_toolkit persists command history across sessions.
HISTORY_PATH = Path("~/.twinspark/history").expanduser()

#: Prefix that marks a line as a slash command.
COMMAND_PREFIX = "/"

#: Commands that terminate the REPL.
QUIT_COMMANDS = {"quit", "exit"}

#: Commands that start a fresh session.
RESET_COMMANDS = {"new", "reset"}

_PROMPT = "you › "
_ASSISTANT_LABEL = "twinspark › "


# --------------------------------------------------------------------------- #
# Pure helpers (no I/O — unit-testable)                                        #
# --------------------------------------------------------------------------- #
def parse_command(text: str) -> Optional[tuple[str, str]]:
    """Parse a slash command line into ``(command, argument)``.

    Args:
        text: The raw user input line.

    Returns:
        ``None`` when ``text`` is not a slash command (i.e. does not start with
        ``/`` after stripping). Otherwise a ``(command, argument)`` tuple where
        ``command`` is lowercased and stripped of its leading slash and
        ``argument`` is the remainder of the line (may be empty).
    """
    stripped = text.strip()
    if not stripped.startswith(COMMAND_PREFIX):
        return None
    # Drop the leading slash, then split into command + rest on first space.
    body = stripped[len(COMMAND_PREFIX):]
    parts = body.split(None, 1)
    if not parts:
        # A lone "/" — treat as an empty (unknown) command.
        return "", ""
    command = parts[0].lower()
    argument = parts[1].strip() if len(parts) > 1 else ""
    return command, argument


def help_text() -> str:
    """Return the multi-line help / command reference string."""
    return (
        "Commands:\n"
        "  /new, /reset          Start a fresh conversation (new session)\n"
        "  /skills               List available skills\n"
        "  /memory <query>       Recall relevant memories for <query>\n"
        "  /history <query>      Search past messages for <query>\n"
        "  /help                 Show this help\n"
        "  /quit, /exit          Exit TwinSpark\n"
        "\n"
        "Tips:\n"
        "  • Ctrl+C interrupts the current reply and returns to the prompt.\n"
        "  • Ctrl+D (EOF) exits TwinSpark.\n"
        "  • Anything not starting with '/' is sent to the agent."
    )


def welcome_text(model: str, session_id: str) -> str:
    """Return the banner printed at startup.

    Args:
        model: The configured model name to advertise.
        session_id: The active agent session id (a short prefix is shown).
    """
    short_sid = session_id[:8] if session_id else "-"
    return (
        f"{APP_NAME} — minimal agent on Alibaba Bailian (DashScope)\n"
        f"  model:   {model}\n"
        f"  session: {short_sid}\n"
        "  Type /help for commands, /quit to exit."
    )


def format_skills(skills: list[dict]) -> str:
    """Format ``SkillLoader.list_skills()`` output for display.

    Args:
        skills: A list of ``{name, description, category, tags}`` dicts.

    Returns:
        A human-readable, multi-line string. When empty, an informative note
        pointing at the skills directory convention is returned instead.
    """
    if not skills:
        return (
            "No skills found. Add skills under your skills directory "
            "(~/.twinspark/skills/<category>/<name>/SKILL.md)."
        )
    lines = [f"Available skills ({len(skills)}):"]
    for skill in skills:
        name = skill.get("name", "?")
        category = skill.get("category", "")
        description = skill.get("description", "")
        tags = skill.get("tags") or []
        prefix = f"  • {category}/{name}" if category else f"  • {name}"
        if description:
            prefix += f" — {description}"
        lines.append(prefix)
        if tags:
            lines.append(f"      tags: {', '.join(str(t) for t in tags)}")
    return "\n".join(lines)


def format_memory_results(results: list[dict]) -> str:
    """Format ``memory.recall()`` fact dicts for display.

    Args:
        results: Fact dicts with at least ``content`` (and optionally
            ``trust_score``, ``tags``).

    Returns:
        A human-readable, multi-line string; a friendly note when empty.
    """
    if not results:
        return "No relevant memories found."
    lines = [f"Relevant memories ({len(results)}):"]
    for fact in results:
        content = str(fact.get("content", "")).strip()
        trust = fact.get("trust_score")
        tags = fact.get("tags")
        line = f"  • {content}"
        meta: list[str] = []
        if trust is not None:
            meta.append(f"trust={trust}")
        if tags:
            meta.append(f"tags={tags}")
        if meta:
            line += f"  ({', '.join(meta)})"
        lines.append(line)
    return "\n".join(lines)


def format_history_results(results: list[dict]) -> str:
    """Format ``memory.search_history()`` message dicts for display.

    Args:
        results: Message dicts with ``role`` / ``content`` (and optionally
            ``created_at``).

    Returns:
        A human-readable, multi-line string; a friendly note when empty.
    """
    if not results:
        return "No matching history found."
    lines = [f"Matching history ({len(results)}):"]
    for msg in results:
        role = str(msg.get("role", "?"))
        content = str(msg.get("content", "")).strip()
        created = msg.get("created_at")
        stamp = f"[{created}] " if created else ""
        lines.append(f"  • {stamp}{role}: {content}")
    return "\n".join(lines)


def config_error_message(exc: Exception) -> str:
    """Return a friendly message for a configuration/validation failure.

    Args:
        exc: The exception raised while loading configuration (typically a
            :class:`pydantic.ValidationError` for a missing key).
    """
    return (
        f"{APP_NAME} could not start: configuration is incomplete.\n"
        "\n"
        "The DASHSCOPE_API_KEY environment variable is required.\n"
        "Set it in your shell, e.g.:\n"
        "    export DASHSCOPE_API_KEY=sk-...\n"
        "or add it to a local .env file (see .env.example):\n"
        "    DASHSCOPE_API_KEY=sk-...\n"
        "\n"
        f"(details: {exc.__class__.__name__})"
    )


# --------------------------------------------------------------------------- #
# I/O helpers                                                                  #
# --------------------------------------------------------------------------- #
def _emit(text: str = "", *, end: str = "\n") -> None:
    """Write ``text`` to stdout with a flush (used for streaming output)."""
    sys.stdout.write(text + end)
    sys.stdout.flush()


def _resolve_model_name() -> str:
    """Return the configured model name, or a placeholder on failure.

    The banner should never crash the app; if config is somehow unavailable
    after the initial check, fall back to a neutral label.
    """
    try:
        from twinspark.config import get_config

        return get_config().model
    except Exception:  # noqa: BLE001 - banner must not raise
        return "unknown"


# --------------------------------------------------------------------------- #
# Command handling                                                             #
# --------------------------------------------------------------------------- #
async def _handle_command(agent: Agent, command: str, argument: str) -> Optional[Agent]:
    """Dispatch a slash command.

    Args:
        agent: The currently active agent.
        command: The lowercased command name (without leading slash).
        argument: The remainder of the command line.

    Returns:
        * ``None`` to keep the current agent and continue the loop.
        * A **new** :class:`Agent` when the session was reset (``/new``).

    Raises:
        _ExitRepl: When a quit command is issued (handled by the caller).
    """
    if command in QUIT_COMMANDS:
        raise _ExitRepl()

    if command in RESET_COMMANDS:
        await agent.aclose()
        new_agent = Agent()
        _emit(f"Started a new session: {new_agent.session_id[:8]}")
        return new_agent

    if command == "help":
        _emit(help_text())
        return None

    if command == "skills":
        from twinspark.skills.loader import SkillLoader

        try:
            skills = SkillLoader().list_skills()
        except Exception as exc:  # noqa: BLE001 - never crash the REPL
            _emit(f"Could not list skills: {exc}")
            return None
        _emit(format_skills(skills))
        return None

    if command == "memory":
        if not argument:
            _emit("Usage: /memory <query>")
            return None
        try:
            results = agent.memory.recall(argument)
        except Exception as exc:  # noqa: BLE001
            _emit(f"Could not recall memories: {exc}")
            return None
        _emit(format_memory_results(results))
        return None

    if command == "history":
        if not argument:
            _emit("Usage: /history <query>")
            return None
        try:
            results = agent.memory.search_history(argument)
        except Exception as exc:  # noqa: BLE001
            _emit(f"Could not search history: {exc}")
            return None
        _emit(format_history_results(results))
        return None

    _emit(f"Unknown command: /{command}. Type /help for the command list.")
    return None


async def _stream_reply(agent: Agent, user_msg: str) -> None:
    """Consume ``agent.run_stream`` and print deltas with a typewriter effect.

    Chunks are written without a trailing newline so they concatenate into a
    single flowing reply; a final newline is emitted once the stream ends.

    A ``KeyboardInterrupt`` raised mid-stream (Ctrl+C) propagates to the
    caller after ensuring the partial line is terminated; the async generator
    is closed so the agent can persist whatever text was produced.
    """
    _emit(_ASSISTANT_LABEL, end="")
    stream = agent.run_stream(user_msg)
    produced = False
    try:
        async for chunk in stream:
            if chunk:
                produced = True
                _emit(chunk, end="")
    except KeyboardInterrupt:
        # Terminate the partial line and re-raise so the loop can report it.
        _emit("", end="\n")
        await stream.aclose()
        raise
    finally:
        # Ensure the generator's own finally (partial persistence) runs.
        await stream.aclose()
    # Normal completion: close the line.
    _emit("" if produced else "(no output)")


class _ExitRepl(Exception):
    """Internal control-flow signal to leave the REPL loop."""


# --------------------------------------------------------------------------- #
# REPL                                                                         #
# --------------------------------------------------------------------------- #
async def _run_repl(agent: Agent, session: PromptSession) -> None:
    """The core read-eval-print loop over an already-constructed agent."""
    _emit(welcome_text(_resolve_model_name(), agent.session_id))
    _emit("")

    while True:
        try:
            with patch_stdout():
                text = await session.prompt_async(_PROMPT)
        except KeyboardInterrupt:
            # Ctrl+C at an (empty) prompt: ignore and re-prompt.
            continue
        except EOFError:
            # Ctrl+D: exit gracefully.
            _emit("\nGoodbye.")
            break

        if text is None:
            continue
        stripped = text.strip()
        if not stripped:
            continue

        parsed = parse_command(stripped)
        if parsed is not None:
            command, argument = parsed
            try:
                new_agent = await _handle_command(agent, command, argument)
            except _ExitRepl:
                _emit("Goodbye.")
                break
            if new_agent is not None:
                agent = new_agent
            continue

        # Regular message → stream the agent's reply.
        try:
            await _stream_reply(agent, stripped)
        except KeyboardInterrupt:
            _emit("[interrupted]")
            continue
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive
            _emit(f"\n[error] {exc.__class__.__name__}: {exc}")
            continue


async def main_async() -> int:
    """Async entry point: validate config, build the agent, run the REPL.

    Returns:
        A process exit code (``0`` on normal exit, ``1`` on a config error).
    """
    # Validate configuration up front so a missing key produces a friendly
    # message rather than a traceback deep inside the first LLM call.
    try:
        from twinspark.config import get_config

        get_config()
    except Exception as exc:  # noqa: BLE001 - surface a friendly message
        _emit(config_error_message(exc), end="\n")
        return 1

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    session: PromptSession = PromptSession(history=FileHistory(str(HISTORY_PATH)))

    agent = Agent()
    try:
        await _run_repl(agent, session)
    finally:
        await agent.aclose()
    return 0


def main() -> None:
    """Synchronous console-script entry point (``twinspark = ...:main``).

    Drives :func:`main_async` with :func:`asyncio.run` and exits with its
    return code. A stray ``KeyboardInterrupt`` at the very top level (e.g.
    before the loop is ready) is treated as a clean exit.
    """
    try:
        exit_code = asyncio.run(main_async())
    except KeyboardInterrupt:
        exit_code = 130
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
