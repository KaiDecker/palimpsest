"""Small SQLite persistence layer for the MVP.

The schema is deliberately boring and extensible: JSON columns preserve context
that can later be used to build SFT/DPO/evaluation datasets.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: str | Path = "data/palimpsest.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connection(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _initialize(self) -> None:
        with self.connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    stability REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL,
                    evidence_count INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    last_updated TEXT NOT NULL,
                    valid_until TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_memories_content ON memories(content);
                CREATE TABLE IF NOT EXISTS profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.5,
                    source TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS experiences (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
                    prompt TEXT NOT NULL,
                    response TEXT NOT NULL,
                    context_json TEXT NOT NULL,
                    retrieved_memories_json TEXT NOT NULL,
                    feedback_json TEXT NOT NULL DEFAULT '{}',
                    model_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY,
                    experience_id TEXT NOT NULL REFERENCES experiences(id) ON DELETE CASCADE,
                    rating INTEGER,
                    edited_response TEXT,
                    chosen_response TEXT,
                    rejected_response TEXT,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS variants (
                    id TEXT PRIMARY KEY,
                    experience_id TEXT NOT NULL REFERENCES experiences(id) ON DELETE CASCADE,
                    label TEXT NOT NULL CHECK(label IN ('A', 'B')),
                    content TEXT NOT NULL,
                    model_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_variants_experience ON variants(experience_id, label);
                """
            )

    @staticmethod
    def new_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def decode(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return dict(row)

    @staticmethod
    def json_value(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False)

    @staticmethod
    def parse_json(value: str) -> Any:
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return value
