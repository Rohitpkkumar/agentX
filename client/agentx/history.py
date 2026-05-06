"""SQLite-backed conversation history for local agent mode."""
from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any


class History:
    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                extra TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            )
        """)
        self._conn.commit()

    def create_session(self, title: str = "") -> str:
        sid = str(uuid.uuid4())
        self._conn.execute(
            "INSERT OR IGNORE INTO sessions (id, title) VALUES (?, ?)", (sid, title)
        )
        self._conn.commit()
        return sid

    def touch(self, session_id: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,)
        )
        self._conn.commit()

    def append(self, session_id: str, role: str, content: str, extra: dict | None = None) -> None:
        self._conn.execute(
            "INSERT INTO messages (session_id, role, content, extra) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(extra or {})),
        )
        self.touch(session_id)

    def load(self, session_id: str) -> list[dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT role, content, extra FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        )
        return [{"role": r, "content": c, "extra": json.loads(e)} for r, c, e in cur.fetchall()]

    def list_sessions(self, limit: int = 20) -> list[dict]:
        cur = self._conn.execute(
            "SELECT id, title, created_at, updated_at FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        return [{"id": r[0], "title": r[1], "created_at": r[2], "updated_at": r[3]} for r in cur.fetchall()]

    def message_count(self, session_id: str) -> int:
        cur = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        )
        return cur.fetchone()[0]

    def update_title(self, session_id: str, title: str) -> None:
        self._conn.execute(
            "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
