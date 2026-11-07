"""Holdout commitment: SHA-256 hash of the full spec + predicted direction + timestamp, then out-of-sample scoring.

The held-out slice is never touched during search. The hash proves the spec was frozen BEFORE the holdout was scored.
It does NOT prove the spec was chosen before the search (it is registered after the verdicts), so this is a holdout
commitment, not a classical pre-registration. Anyone can recompute the hash from the published spec, direction and `ts`.
"""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from scipy import stats

from ..ledger import Ledger

_SCHEMA = """
CREATE TABLE IF NOT EXISTS prereg (
    hash TEXT PRIMARY KEY, run_id TEXT, hypothesis_id TEXT, spec TEXT, direction INTEGER, ts REAL,
    holdout_info TEXT, result TEXT
);
"""


def spec_hash(spec: dict, direction: int, ts: float) -> str:
    blob = json.dumps({"spec": spec, "direction": direction, "ts": ts}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


def register(ledger: Ledger, run_id: str, hypothesis_id: str, spec: dict, direction: int,
             holdout_info: dict | None = None, ts: float | None = None) -> dict:
    ts = ts if ts is not None else time.time()
    h = spec_hash(spec, direction, ts)
    with ledger._lock:
        ledger._db.executescript(_SCHEMA)
        ledger._db.execute("INSERT OR REPLACE INTO prereg VALUES (?,?,?,?,?,?,?,?)",
                           (h, run_id, hypothesis_id, json.dumps(spec, sort_keys=True, default=str), int(direction), ts,
                            json.dumps(holdout_info, default=str), None))
        ledger._db.commit()
    return {"hash": h, "hypothesis_id": hypothesis_id, "direction": int(direction), "ts": ts}


def verify(record: dict, spec: dict) -> bool:
    return spec_hash(spec, record["direction"], record["ts"]) == record["hash"]


def score_out_of_sample(ledger: Ledger, record: dict, pnl_holdout: np.ndarray, alpha: float = 0.05) -> dict:
    """Score a pre-registered claim on data that was never touched. One-sided test in the predicted direction."""
    x = np.asarray(pnl_holdout, float) * record["direction"]
    n = len(x)
    sd = x.std(ddof=1)
    t = x.mean() / sd * np.sqrt(n) if sd > 0 else 0.0
    p = float(stats.t.sf(t, n - 1))
    res = {"n_days": n, "oos_sharpe": float(x.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0, "t": float(t),
           "p_one_sided": p, "confirmed": bool(p < alpha and x.mean() > 0)}
    with ledger._lock:
        ledger._db.executescript(_SCHEMA)
        ledger._db.execute("UPDATE prereg SET result=? WHERE hash=?", (json.dumps(res), record["hash"]))
        ledger._db.commit()
    return res
