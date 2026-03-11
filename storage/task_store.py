"""SQLite-backed storage for scheduled prompts and execution records."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class ScheduledTask:
    id: int
    goal: str
    prompt: str
    timestamp: str
    status: str


class TaskStore:
    def __init__(self, db_path: str = "data/kaal.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scheduled_prompts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    goal TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scheduled_prompt_id INTEGER,
                    status TEXT NOT NULL,
                    result TEXT,
                    payload TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(scheduled_prompt_id) REFERENCES scheduled_prompts(id)
                )
                """
            )

    def add_plans(self, plans: list[dict]) -> int:
        if not plans:
            return 0
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO scheduled_prompts(goal, prompt, timestamp, status) VALUES (?, ?, ?, 'pending')",
                [(p["goal"], p["prompt"], p["timestamp"]) for p in plans],
            )
            return len(plans)

    def due_tasks(self) -> list[ScheduledTask]:
        now = datetime.now().isoformat(sep=" ", timespec="microseconds")
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, goal, prompt, timestamp, status FROM scheduled_prompts WHERE status='pending' AND timestamp <= ? ORDER BY timestamp ASC",
                (now,),
            ).fetchall()
        return [ScheduledTask(**dict(row)) for row in rows]

    def mark_executed(self, task_id: int, status: str, result: str, payload: dict | list | str | None):
        payload_text = json.dumps(payload, ensure_ascii=False) if payload is not None and not isinstance(payload, str) else payload
        with self._connect() as conn:
            conn.execute("UPDATE scheduled_prompts SET status='done' WHERE id=?", (task_id,))
            conn.execute(
                "INSERT INTO executions(scheduled_prompt_id, status, result, payload) VALUES (?, ?, ?, ?)",
                (task_id, status, result, payload_text),
            )

    def pending_count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM scheduled_prompts WHERE status='pending'").fetchone()[0])
