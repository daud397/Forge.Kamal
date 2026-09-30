"""Job state in SQLite. One row per job, state as JSON.

A prototype does not need migrations, but it does need to survive a restart
mid-render, which an in-memory dict does not.
"""
from __future__ import annotations

import json
import re
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


# The sidebar used to show the first sixty characters of the instruction, so
# every row read "Would you please apply Whole Blue Steel duvet…" and nothing
# could be told apart. A title is the subject of the sentence, not its opening.
BOILERPLATE = (
    "would you please", "could you please", "please", "i attached", "i have attached",
    "can you", "kindly", "i want you to", "i need you to", "make sure",
    "apply as it is", "as it is", "use this image as a reference",
)


def short_title(state: dict) -> str:
    brief = state.get("brief") or {}
    if brief.get("product_name"):
        return brief["product_name"][:48]

    t = " ".join((state.get("note") or "").split())
    low = t.lower()
    for phrase in BOILERPLATE:
        while low.startswith(phrase):
            t = t[len(phrase):].lstrip(" ,.:-")
            low = t.lower()

    # A filename in the instruction is the most recognisable thing in it, and
    # the mill's filenames carry spaces: "Witches Brew (Bed) YELLOW.jpg".
    m = re.search(r"([\w][\w()'&\- ]{1,40})\.(?:jpe?g|png|webp|tiff?)\b", t, re.I)
    if m:
        stem = m.group(1).strip(" ,.-")
        if len(stem) > 2:
            return stem[:48]

    words = t.split()
    return " ".join(words[:7])[:48] or "Untitled run"


def recent(limit: int = 60, project: str | None = None) -> list[dict]:
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT state FROM jobs ORDER BY created DESC LIMIT ?", (limit * 3,)
        ).fetchall()
    out = []
    for r in rows:
        s = json.loads(r["state"])
        brief = s.get("brief") or {}
        qa = s.get("qa") or {}
        if project is not None and (s.get("project") or "") != project:
            continue
        out.append({
            "id": s["id"],
            "thread": s.get("thread") or s["id"],
            "title": short_title(s),
            "rating": s.get("rating"),
            "error": s.get("error"),
            "stage": s.get("stage"),
            "mode": s.get("mode", "edit"),
            "project": s.get("project") or "",
            "product": brief.get("product_name") or s.get("note") or None,
            "created": s.get("created_at"),
            "thumb": s.get("thumb"),
            "verdict": qa.get("level"),
            "exported": len(s.get("exports") or []),
        })
    return out[:limit]


def projects() -> list[dict]:
    """Every project name in use, with how many runs sit under each."""
    with _lock, _conn() as c:
        rows = c.execute("SELECT state FROM jobs").fetchall()

    counts: dict[str, int] = {}
    for r in rows:
        name = (json.loads(r["state"]).get("project") or "").strip()
        if name:
            counts[name] = counts.get(name, 0) + 1

    return [{"name": n, "runs": counts[n]} for n in sorted(counts)]
