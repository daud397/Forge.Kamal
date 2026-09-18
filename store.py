"""Job state in SQLite. One row per job, state as JSON.

A prototype does not need migrations, but it does need to survive a restart
mid-render, which an in-memory dict does not.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid

import config

_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                created REAL NOT NULL,
                updated REAL NOT NULL,
                stage TEXT NOT NULL,
                state TEXT NOT NULL
            )
        """)


def create(initial: dict) -> str:
    job_id = uuid.uuid4().hex[:12]
    now = time.time()
    initial = {**initial, "id": job_id, "log": [], "stage": "created"}
    with _lock, _conn() as c:
        c.execute("INSERT INTO jobs (id, created, updated, stage, state) VALUES (?,?,?,?,?)",
                  (job_id, now, now, "created", json.dumps(initial)))
    return job_id


def get(job_id: str) -> dict | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
    return json.loads(row["state"]) if row else None


def update(job_id: str, **fields) -> dict:
    with _lock, _conn() as c:
        row = c.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        state = json.loads(row["state"])
        state.update(fields)
        c.execute("UPDATE jobs SET updated=?, stage=?, state=? WHERE id=?",
                  (time.time(), state.get("stage", "?"), json.dumps(state), job_id))
    return state


def log(job_id: str, message: str, level: str = "info"):
    with _lock, _conn() as c:
        row = c.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            return
        state = json.loads(row["state"])
        state.setdefault("log", []).append(
            {"t": round(time.time(), 2), "level": level, "message": message}
        )
        c.execute("UPDATE jobs SET updated=?, state=? WHERE id=?",
                  (time.time(), json.dumps(state), job_id))


def recent(limit: int = 30) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT state FROM jobs ORDER BY created DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        s = json.loads(r["state"])
        out.append({
            "id": s["id"],
            "stage": s.get("stage"),
            "product": (s.get("brief") or {}).get("product_name"),
            "created": s.get("created_at"),
        })
    return out
