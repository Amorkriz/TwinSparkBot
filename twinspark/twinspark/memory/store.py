"""SQLite + FTS5 backed memory store for TwinSpark.

This module provides :class:`MemoryStore`, a thread-safe persistence layer
that keeps track of:

* **sessions** – conversation containers identified by a ``session_id``.
* **messages** – individual chat turns belonging to a session.
* **facts** – durable pieces of knowledge with a trust score, full-text
  searchable via an FTS5 virtual table kept in sync by triggers.

It is intentionally small: no entity resolution, no vector banks – just the
essentials required by the agent core (Task 6), the HTTP API (Task 8) and
skill retrieval (Task 5).
"""

from __future__ import annotations

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

__all__ = ["MemoryStore"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata    TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS messages (
    msg_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);

CREATE TABLE IF NOT EXISTS facts (
    fact_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    content         TEXT NOT NULL UNIQUE,
    tags            TEXT DEFAULT '',
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_facts_trust ON facts(trust_score DESC);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;
"""

# Matches runs of alphanumerics / CJK / underscore. Used to tokenize a
# user-supplied query into safe terms before building an FTS5 MATCH string.
_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


class MemoryStore:
    """Thread-safe SQLite + FTS5 memory store.

    A single connection is shared across threads (``check_same_thread=False``)
    and all write operations are guarded by a :class:`threading.RLock` so the
    store can be used safely from a multi-request API server.

    Args:
        path: Path to the SQLite database file. When ``None`` the value is
            taken from :func:`twinspark.config.get_config` (``cfg.db_path``).
            Pass ``":memory:"`` for an ephemeral in-memory database (useful in
            tests). A custom path is created together with its parent
            directory if needed.
    """

    def __init__(self, path: Optional[str | Path] = None) -> None:
        if path is None:
            # Import lazily so tests that pass an explicit path (or ``:memory:``)
            # do not require the DASHSCOPE_API_KEY that get_config() demands.
            from twinspark.config import get_config

            resolved: str | Path = get_config().db_path
        else:
            resolved = path

        self._path: str = str(resolved)
        if self._path != ":memory:":
            parent = Path(self._path).expanduser().parent
            parent.mkdir(parents=True, exist_ok=True)

        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_db()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def _init_db(self) -> None:
        """Create tables, indexes, the FTS5 table and sync triggers."""
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Sessions
    # ------------------------------------------------------------------ #
    def ensure_session(
        self, session_id: str, metadata: Optional[str] = None
    ) -> None:
        """Create the session if it does not already exist.

        Existing sessions are left untouched (their ``metadata`` and
        timestamps are preserved).

        Args:
            session_id: Unique session identifier.
            metadata: Optional free-form metadata string (e.g. JSON).
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id, metadata) "
                "VALUES (?, ?)",
                (session_id, metadata or ""),
            )
            self._conn.commit()

    # ------------------------------------------------------------------ #
    # Messages
    # ------------------------------------------------------------------ #
    def add_message(self, session_id: str, role: str, content: str) -> int:
        """Append a message to a session and bump the session's ``updated_at``.

        The parent session is created on demand if it does not exist yet.

        Args:
            session_id: Session the message belongs to.
            role: Message role, e.g. ``"user"``, ``"assistant"`` or ``"system"``.
            content: Message text.

        Returns:
            The auto-incremented ``msg_id`` of the inserted row.
        """
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO sessions(session_id) VALUES (?)",
                (session_id,),
            )
            cur = self._conn.execute(
                "INSERT INTO messages(session_id, role, content) "
                "VALUES (?, ?, ?)",
                (session_id, role, content),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = CURRENT_TIMESTAMP "
                "WHERE session_id = ?",
                (session_id,),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_history(
        self, session_id: str, limit: Optional[int] = None
    ) -> list[dict[str, Any]]:
        """Return a session's messages ordered by ``created_at`` ascending.

        Args:
            session_id: Session to read.
            limit: When given, return only the most recent ``limit`` messages
                (still ordered oldest-to-newest in the result).

        Returns:
            A list of ``{"role", "content", "created_at"}`` dicts.
        """
        with self._lock:
            if limit is None:
                rows = self._conn.execute(
                    "SELECT role, content, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY msg_id ASC",
                    (session_id,),
                ).fetchall()
            else:
                # Grab the newest ``limit`` rows, then re-sort ascending.
                rows = self._conn.execute(
                    "SELECT role, content, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY msg_id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
                rows = list(reversed(rows))
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Facts
    # ------------------------------------------------------------------ #
    def add_fact(
        self, content: str, tags: str = "", trust_score: float = 0.5
    ) -> Optional[int]:
        """Insert a fact, ignoring duplicates (``content`` is UNIQUE).

        Args:
            content: The fact text. Must be unique across the store.
            tags: Optional space/comma separated tags (also full-text indexed).
            trust_score: Initial trust in ``[0, 1]``; defaults to ``0.5``.

        Returns:
            The new ``fact_id`` when inserted, or ``None`` if a fact with the
            same ``content`` already existed.
        """
        with self._lock:
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO facts(content, tags, trust_score) "
                "VALUES (?, ?, ?)",
                (content, tags, trust_score),
            )
            self._conn.commit()
            if cur.rowcount == 0:
                return None
            return int(cur.lastrowid)

    def recall(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Full-text search facts, ordered by trust then FTS rank.

        The user query is sanitized into a safe FTS5 MATCH expression (tokens
        OR-joined, each double-quoted). If FTS5 raises or the sanitized query
        is empty, the method falls back to a ``LIKE`` scan so that arbitrary
        user input never triggers ``sqlite3.OperationalError``.

        Every returned fact has its ``retrieval_count`` incremented by one.

        Args:
            query: Natural-language search string.
            limit: Maximum number of facts to return.

        Returns:
            A list of fact dicts ordered by ``trust_score`` descending
            (FTS rank as a tie-breaker).
        """
        with self._lock:
            match_query = self._build_fts_query(query)
            rows: list[sqlite3.Row] = []

            if match_query:
                sql = """
                    SELECT f.fact_id, f.content, f.tags, f.trust_score,
                           f.retrieval_count, f.created_at, f.updated_at
                    FROM facts f
                    JOIN facts_fts fts ON fts.rowid = f.fact_id
                    WHERE facts_fts MATCH ?
                    ORDER BY f.trust_score DESC, fts.rank
                    LIMIT ?
                """
                try:
                    rows = self._conn.execute(
                        sql, (match_query, limit)
                    ).fetchall()
                except sqlite3.OperationalError:
                    rows = []

            if not rows:
                rows = self._recall_like(query, limit)

            results = [dict(r) for r in rows]
            if results:
                ids = [r["fact_id"] for r in results]
                placeholders = ",".join("?" * len(ids))
                self._conn.execute(
                    "UPDATE facts SET retrieval_count = retrieval_count + 1 "
                    f"WHERE fact_id IN ({placeholders})",
                    ids,
                )
                self._conn.commit()
            return results

    def _recall_like(self, query: str, limit: int) -> list[sqlite3.Row]:
        """Fallback fact search using ``LIKE`` on content and tags."""
        term = query.strip()
        if not term:
            return []
        pattern = f"%{self._escape_like(term)}%"
        return self._conn.execute(
            """
            SELECT fact_id, content, tags, trust_score,
                   retrieval_count, created_at, updated_at
            FROM facts
            WHERE content LIKE ? ESCAPE '\\' OR tags LIKE ? ESCAPE '\\'
            ORDER BY trust_score DESC, fact_id DESC
            LIMIT ?
            """,
            (pattern, pattern, limit),
        ).fetchall()

    # ------------------------------------------------------------------ #
    # Message search
    # ------------------------------------------------------------------ #
    def search_history(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Search message contents with a ``LIKE`` scan.

        Args:
            query: Substring to look for in message content.
            limit: Maximum number of messages to return.

        Returns:
            A list of ``{"msg_id", "session_id", "role", "content",
            "created_at"}`` dicts ordered newest first.
        """
        with self._lock:
            term = query.strip()
            if not term:
                return []
            pattern = f"%{self._escape_like(term)}%"
            rows = self._conn.execute(
                """
                SELECT msg_id, session_id, role, content, created_at
                FROM messages
                WHERE content LIKE ? ESCAPE '\\'
                ORDER BY msg_id DESC
                LIMIT ?
                """,
                (pattern, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    # ------------------------------------------------------------------ #
    # Query sanitization helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_fts_query(query: str) -> str:
        """Turn arbitrary user text into a safe FTS5 MATCH expression.

        Tokens are extracted with a Unicode word regex, each wrapped in double
        quotes (with internal quotes doubled) and OR-joined. This avoids FTS5
        syntax errors from characters like ``"``, ``*``, ``(`` or ``:`` and
        broadens recall (default FTS5 behaviour is AND).

        Returns an empty string when the query has no usable tokens.
        """
        if not query or not query.strip():
            return ""
        tokens = _TOKEN_RE.findall(query)
        if not tokens:
            return ""
        quoted = [f'"{tok.replace(chr(34), chr(34) * 2)}"' for tok in tokens]
        return " OR ".join(quoted)

    @staticmethod
    def _escape_like(term: str) -> str:
        """Escape ``%``, ``_`` and ``\\`` for a LIKE pattern (ESCAPE '\\')."""
        return (
            term.replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "MemoryStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
