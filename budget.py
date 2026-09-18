"""Spend tracking and a hard daily ceiling.

The failure mode this exists to prevent: a loop, a bad prompt, or someone who
found the URL quietly running up a bill you discover at the end of the month.
Generative image models are expensive per call, so the cap refuses work rather
than warning about it.

Costs are ESTIMATES from config, not billed amounts. OpenAI is the only source of
truth for what you actually owe. Check the dashboard and correct the per-call
figures in .env before you rely on the cap being accurate.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone

import config

_lock = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


def _conn():
    c = sqlite3.connect(config.DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    return c


def init():
    with _lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS spend (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                day TEXT NOT NULL,
                kind TEXT NOT NULL,
                model TEXT NOT NULL,
                units INTEGER NOT NULL DEFAULT 1,
                estimate REAL NOT NULL,
                job_id TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS spend_day ON spend(day)")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record(kind: str, model: str, units: int = 1, job_id: str | None = None) -> float:
    """Log an estimated cost. kind is 'image' or 'text'."""
    per_unit = config.COST_PER_IMAGE if kind == "image" else config.COST_PER_TEXT_CALL
    estimate = per_unit * units

    with _lock, _conn() as c:
        c.execute(
            "INSERT INTO spend (ts, day, kind, model, units, estimate, job_id) "
            "VALUES (?,?,?,?,?,?,?)",
            (time.time(), _today(), kind, model, units, estimate, job_id),
        )
    return estimate


def spent_today() -> float:
    with _lock, _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(estimate), 0) AS total FROM spend WHERE day = ?",
            (_today(),),
        ).fetchone()
    return float(row["total"])


def summary() -> dict:
    with _lock, _conn() as c:
        today = c.execute(
            "SELECT kind, COUNT(*) AS calls, COALESCE(SUM(estimate),0) AS total "
            "FROM spend WHERE day = ? GROUP BY kind", (_today(),)
        ).fetchall()
        lifetime = c.execute(
            "SELECT COALESCE(SUM(estimate),0) AS total FROM spend"
        ).fetchone()

    breakdown = {r["kind"]: {"calls": r["calls"], "estimate": round(r["total"], 3)}
                 for r in today}
    total = round(sum(v["estimate"] for v in breakdown.values()), 3)

    return {
        "today": total,
        "cap": config.DAILY_CAP,
        "remaining": round(max(0.0, config.DAILY_CAP - total), 3),
        "breakdown": breakdown,
        "lifetime": round(float(lifetime["total"]), 3),
        "enforced": config.DAILY_CAP > 0,
        "note": "Estimated from configured per-call costs, not billed amounts.",
    }


def history(days: int = 14) -> list[dict]:
    """Spend per day, newest first, for the usage view."""
    with _lock, _conn() as c:
        rows = c.execute(
            "SELECT day, kind, COUNT(*) AS calls, COALESCE(SUM(estimate),0) AS total "
            "FROM spend GROUP BY day, kind ORDER BY day DESC"
        ).fetchall()

    byday: dict[str, dict] = {}
    for r in rows:
        d = byday.setdefault(r["day"], {"day": r["day"], "images": 0,
                                        "text": 0, "estimate": 0.0})
        d["images" if r["kind"] == "image" else "text"] += r["calls"]
        d["estimate"] += float(r["total"])

    out = sorted(byday.values(), key=lambda d: d["day"], reverse=True)[:days]
    for d in out:
        d["estimate"] = round(d["estimate"], 3)
    return out


def check(about_to_spend: float = 0.0):
    """Raise before making a call that would breach the cap."""
    if config.DAILY_CAP <= 0:
        return                                  # 0 means no ceiling
    if spent_today() + about_to_spend > config.DAILY_CAP:
        raise BudgetExceeded(
            f"Daily cap of ${config.DAILY_CAP:.2f} reached "
            f"(${spent_today():.2f} estimated so far). It resets at 00:00 UTC. "
            f"Raise DAILY_CAP in .env if this is deliberate."
        )


def check_images(count: int):
    check(config.COST_PER_IMAGE * count)


def check_text():
    check(config.COST_PER_TEXT_CALL)
