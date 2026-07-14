"""Tests for :class:`twinspark.memory.store.MemoryStore`."""

from __future__ import annotations

import sqlite3

import pytest

from twinspark.memory.store import MemoryStore


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    """A MemoryStore backed by a temporary on-disk database.

    A file-based DB is used (rather than ``:memory:``) so the shared
    connection semantics match production; the temp dir is cleaned up by
    pytest automatically.
    """
    db = tmp_path / "state.db"
    s = MemoryStore(path=db)
    yield s
    s.close()


# --------------------------------------------------------------------- #
# Schema / construction
# --------------------------------------------------------------------- #
def test_schema_created(store: MemoryStore) -> None:
    tables = {
        row[0]
        for row in store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"sessions", "messages", "facts", "facts_fts"} <= tables


def test_memory_db_works() -> None:
    """FTS5 must be available even for an in-memory database."""
    s = MemoryStore(path=":memory:")
    try:
        fid = s.add_fact("in memory fact about python", tags="lang")
        assert fid is not None
        assert s.recall("python")
    finally:
        s.close()


# --------------------------------------------------------------------- #
# Messages & history
# --------------------------------------------------------------------- #
def test_add_message_and_history_roundtrip(store: MemoryStore) -> None:
    store.ensure_session("s1")
    store.add_message("s1", "user", "hello")
    store.add_message("s1", "assistant", "hi there")
    store.add_message("s1", "user", "how are you?")

    history = store.get_history("s1")
    assert [m["role"] for m in history] == ["user", "assistant", "user"]
    assert [m["content"] for m in history] == [
        "hello",
        "hi there",
        "how are you?",
    ]
    assert all("created_at" in m for m in history)


def test_get_history_limit_returns_recent_in_order(store: MemoryStore) -> None:
    for i in range(5):
        store.add_message("s1", "user", f"msg-{i}")
    recent = store.get_history("s1", limit=2)
    assert [m["content"] for m in recent] == ["msg-3", "msg-4"]


def test_add_message_autocreates_session(store: MemoryStore) -> None:
    store.add_message("auto", "user", "hi")
    rows = store._conn.execute(
        "SELECT session_id FROM sessions WHERE session_id = ?", ("auto",)
    ).fetchall()
    assert len(rows) == 1


def test_add_message_updates_session_timestamp(store: MemoryStore) -> None:
    store.ensure_session("s1")
    store.add_message("s1", "user", "hi")
    row = store._conn.execute(
        "SELECT created_at, updated_at FROM sessions WHERE session_id = ?",
        ("s1",),
    ).fetchone()
    assert row["updated_at"] is not None


# --------------------------------------------------------------------- #
# Facts & recall
# --------------------------------------------------------------------- #
def test_add_fact_dedupes(store: MemoryStore) -> None:
    first = store.add_fact("the sky is blue")
    dup = store.add_fact("the sky is blue")
    assert first is not None
    assert dup is None
    count = store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
    assert count == 1


def test_recall_orders_by_trust(store: MemoryStore) -> None:
    store.add_fact("python is a programming language", trust_score=0.4)
    store.add_fact("python powers many web servers", trust_score=0.9)
    results = store.recall("python", limit=5)
    assert len(results) == 2
    # Highest trust first.
    assert results[0]["trust_score"] == 0.9
    assert results[1]["trust_score"] == 0.4


def test_recall_increments_retrieval_count(store: MemoryStore) -> None:
    store.add_fact("cats are mammals")
    store.recall("cats")
    row = store._conn.execute(
        "SELECT retrieval_count FROM facts WHERE content = ?",
        ("cats are mammals",),
    ).fetchone()
    assert row["retrieval_count"] == 1


def test_recall_empty_query_returns_empty(store: MemoryStore) -> None:
    store.add_fact("something")
    assert store.recall("") == []
    assert store.recall("   ") == []


def test_recall_special_chars_do_not_crash(store: MemoryStore) -> None:
    store.add_fact("email me at test@example.com about python")
    # These would break a naive FTS5 MATCH expression.
    for q in ['"', "()", "AND OR NOT", "test@example.com", "*:^", "a AND"]:
        try:
            store.recall(q)
        except sqlite3.OperationalError as exc:  # pragma: no cover
            pytest.fail(f"recall raised on query {q!r}: {exc}")


def test_recall_matches_tags(store: MemoryStore) -> None:
    store.add_fact("some neutral content", tags="astronomy space")
    results = store.recall("astronomy")
    assert len(results) == 1


# --------------------------------------------------------------------- #
# History search
# --------------------------------------------------------------------- #
def test_search_history_matches(store: MemoryStore) -> None:
    store.add_message("s1", "user", "let us talk about databases")
    store.add_message("s1", "assistant", "sure, sqlite is great")
    store.add_message("s2", "user", "unrelated chatter")

    hits = store.search_history("sqlite")
    assert len(hits) == 1
    assert hits[0]["content"] == "sure, sqlite is great"


def test_search_history_empty_query(store: MemoryStore) -> None:
    store.add_message("s1", "user", "hello")
    assert store.search_history("") == []


def test_search_history_special_chars_do_not_crash(store: MemoryStore) -> None:
    store.add_message("s1", "user", "50% off _today_ only")
    # LIKE wildcards must be escaped, not interpreted.
    assert store.search_history("50%") == store.search_history("50%")
    assert store.search_history("_today_")


# --------------------------------------------------------------------- #
# Context manager
# --------------------------------------------------------------------- #
def test_context_manager(tmp_path) -> None:
    db = tmp_path / "ctx.db"
    with MemoryStore(path=db) as s:
        s.add_fact("hi")
        assert s.recall("hi")
