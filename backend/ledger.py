"""The Ledger: single source of truth for every test any agent runs.

Ten agents searching means ten times the lucky results, so *everything* that is a forking path is
recorded here: hypotheses tried, parameter variants, and discretionary cleaning choices.
The effective search size shown in the UI and used in every confession line is derived from this table.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from typing import Any, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY, name TEXT, config TEXT, created REAL
);
CREATE TABLE IF NOT EXISTS ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    kind TEXT NOT NULL,              -- 'test' | 'cleaning_choice'
    agent TEXT,
    family TEXT,
    hypothesis_id TEXT,
    spec TEXT,                       -- JSON hypothesis spec / choice description
    params TEXT,                     -- JSON params tried
    cleaning TEXT,                   -- JSON cleaning choices in force
    statistic REAL,
    p_raw REAL,
    n_options INTEGER,               -- cleaning_choice: how many alternatives were available
    rows_affected INTEGER,
    ts REAL,
    seed INTEGER,
    tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    status TEXT DEFAULT 'tested',    -- tested | discovery | luck | cant_tell | killed:<cause>
    detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_ledger_run ON ledger(run_id, kind);
"""


def _j(x: Any) -> str | None:
    return None if x is None else json.dumps(x, sort_keys=True, default=str)


class Ledger:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(SCHEMA)

    # ------------------------------------------------------------------ writes
    def start_run(self, name: str = "run", config: dict | None = None) -> str:
        run_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._db.execute("INSERT INTO runs VALUES (?,?,?,?)", (run_id, name, _j(config), time.time()))
            self._db.commit()
        return run_id

    def record(self, run_id: str, agent: str, family: str | None = None, hypothesis_id: str | None = None,
               spec: dict | None = None, params: dict | None = None, cleaning: dict | None = None,
               statistic: float | None = None, p_raw: float | None = None, seed: int | None = None,
               tokens: int = 0, cost_usd: float = 0.0, status: str = "tested") -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO ledger (run_id,kind,agent,family,hypothesis_id,spec,params,cleaning,statistic,p_raw,"
                "ts,seed,tokens,cost_usd,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, "test", agent, family, hypothesis_id, _j(spec), _j(params), _j(cleaning), statistic, p_raw,
                 time.time(), seed, tokens, cost_usd, status))
            self._db.commit()
            return cur.lastrowid

    def record_many(self, run_id: str, rows: Iterable[dict]) -> int:
        n = 0
        with self._lock:
            for r in rows:
                self._db.execute(
                    "INSERT INTO ledger (run_id,kind,agent,family,hypothesis_id,spec,params,cleaning,statistic,p_raw,"
                    "ts,seed,tokens,cost_usd,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, "test", r.get("agent"), r.get("family"), r.get("hypothesis_id"), _j(r.get("spec")),
                     _j(r.get("params")), _j(r.get("cleaning")), r.get("statistic"), r.get("p_raw"), time.time(),
                     r.get("seed"), r.get("tokens", 0), r.get("cost_usd", 0.0), r.get("status", "tested")))
                n += 1
            self._db.commit()
        return n

    def record_cleaning(self, run_id: str, agent: str, name: str, choice: Any, n_options: int,
                        rows_affected: int = 0, seed: int | None = None) -> int:
        """A discretionary data-cleaning choice. Each one multiplies the forking paths."""
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO ledger (run_id,kind,agent,spec,n_options,rows_affected,ts,seed,status) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, "cleaning_choice", agent, _j({"name": name, "choice": choice}), n_options, rows_affected,
                 time.time(), seed, "logged"))
            self._db.commit()
            return cur.lastrowid

    def set_status(self, row_id: int, status: str, detail: dict | None = None):
        with self._lock:
            self._db.execute("UPDATE ledger SET status=?, detail=? WHERE id=?", (status, _j(detail), row_id))
            self._db.commit()

    def set_status_by_hypothesis(self, run_id: str, hypothesis_id: str, status: str, detail: dict | None = None):
        with self._lock:
            self._db.execute("UPDATE ledger SET status=?, detail=? WHERE run_id=? AND hypothesis_id=?",
                             (status, _j(detail), run_id, hypothesis_id))
            self._db.commit()

    # ------------------------------------------------------------------ reads
    def effective_search_size(self, run_id: str) -> dict:
        """Everything the Court must account for: tests + discretionary cleaning forks."""
        with self._lock:
            q = lambda sql, *a: self._db.execute(sql, a).fetchone()[0]
            tests = q("SELECT COUNT(*) FROM ledger WHERE run_id=? AND kind='test'", run_id)
            cleaning = q("SELECT COUNT(*) FROM ledger WHERE run_id=? AND kind='cleaning_choice'", run_id)
            agents = q("SELECT COUNT(DISTINCT agent) FROM ledger WHERE run_id=? AND kind='test'", run_id)
            fork = self._db.execute(
                "SELECT n_options FROM ledger WHERE run_id=? AND kind='cleaning_choice'", (run_id,)).fetchall()
        multiplier = 1
        for r in fork:
            multiplier *= max(1, r[0] or 1)
        return {"tests": tests, "cleaning_choices": cleaning, "agents": agents,
                "cleaning_paths": multiplier,
                # every test that actually ran counts; cleaning forks that were *not* all executed still widen the
                # space the analyst could have explored, reported separately so we never hide them
                "effective": tests, "potential_paths": tests * multiplier}

    def tests(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM ledger WHERE run_id=? AND kind='test' ORDER BY id", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def by_agent(self, run_id: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT agent, COUNT(*) n, SUM(tokens) tokens, SUM(cost_usd) cost FROM ledger "
                "WHERE run_id=? GROUP BY agent", (run_id,)).fetchall()
        return [dict(r) for r in rows]

    def status_counts(self, run_id: str) -> dict:
        with self._lock:
            rows = self._db.execute(
                "SELECT status, COUNT(*) n FROM ledger WHERE run_id=? AND kind='test' GROUP BY status", (run_id,)).fetchall()
        return {r["status"]: r["n"] for r in rows}
