"""SQLite session storage (stdlib sqlite3, one short-lived connection per operation)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    system_prompt TEXT,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    metadata   TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id, id);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sqlite_path_from_url(url: str) -> str:
    """Translate a SQLAlchemy-style URL into a filesystem path.

    sqlite:////data/app.db -> /data/app.db   (absolute)
    sqlite:///./app.db     -> ./app.db       (relative)
    """
    prefix = "sqlite:///"
    if not url.startswith(prefix):
        raise ValueError("DATABASE_URL must start with sqlite:///")
    path = url[len(prefix):]
    if not path:
        raise ValueError("DATABASE_URL has no file path")
    return path


class Database:
    def __init__(self, url: str) -> None:
        self.path = sqlite_path_from_url(url)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        if self.path != ":memory:":
            Path(self.path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)

    def ping(self) -> bool:
        try:
            with self._connect() as conn:
                conn.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT session_id, system_prompt, created_at FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Messages of one session, oldest first. `limit` keeps only the newest N."""
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    "SELECT role, content, metadata, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY id ASC",
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM (SELECT id, role, content, metadata, created_at FROM messages "
                    "WHERE session_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id ASC",
                    (session_id, limit),
                ).fetchall()
        out = []
        for row in rows:
            item: dict[str, Any] = {"role": row["role"], "content": row["content"], "created_at": row["created_at"]}
            if row["metadata"]:
                item["metadata"] = json.loads(row["metadata"])
            out.append(item)
        return out

    def save_exchange(
        self,
        session_id: str,
        system_prompt: str | None,
        user_message: str,
        reply: str,
        metadata: dict[str, Any],
    ) -> None:
        """Persist one successful turn atomically (session row + user + assistant message)."""
        ts = now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions (session_id, system_prompt, created_at, updated_at) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(session_id) DO UPDATE SET "
                "system_prompt = COALESCE(excluded.system_prompt, sessions.system_prompt), "
                "updated_at = excluded.updated_at",
                (session_id, system_prompt, ts, ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, 'user', ?, NULL, ?)",
                (session_id, user_message, ts),
            )
            conn.execute(
                "INSERT INTO messages (session_id, role, content, metadata, created_at) VALUES (?, 'assistant', ?, ?, ?)",
                (session_id, reply, json.dumps(metadata), now_iso()),
            )
