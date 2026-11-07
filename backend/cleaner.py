"""Cleaner: deterministic data cleaning, with the *choices* made by an agent and ALL of them logged.

Every discretionary choice is a forking path. It is written to the ledger as a `cleaning_choice` with the number of
alternatives that were available, so it widens the effective search size shown in every confession line.
"""
from __future__ import annotations

import copy

import numpy as np

from .adapters.market_panel import Panel, known_factors
from .ledger import Ledger
from .llm import LLM

OPTIONS = {
    "winsorize_pct": [1.0, 5.0],            # clip pooled daily returns at these tail percentiles
    "gap_fill": ["fill_zero", "drop"],       # missing return -> 0 (flat) or leave missing (name skipped that day)
    "fundamental_lag_days": [0, 63],         # point-in-time alignment of slowly reported fundamentals
}
DEFAULT = {"winsorize_pct": 1.0, "gap_fill": "fill_zero", "fundamental_lag_days": 63}

CHOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "winsorize_pct": {"enum": OPTIONS["winsorize_pct"]},
        "gap_fill": {"enum": OPTIONS["gap_fill"]},
        "fundamental_lag_days": {"enum": OPTIONS["fundamental_lag_days"]},
        "rationale": {"type": "string"},
    },
    "required": ["winsorize_pct", "gap_fill", "fundamental_lag_days"],
}


def profile(panel: Panel) -> dict:
    R = panel.fields["ret"]
    ok = np.isfinite(R) & panel.alive
    return {"shape": list(R.shape), "missing_frac": float((~np.isfinite(R) & panel.alive).mean()),
            "extreme_frac_gt_25pct": float((np.abs(R[ok]) > 0.25).mean()),
            "delisted_names": int((~panel.listed_today).sum()), "has_fundamentals": "btm" in panel.fields}


def register_mock(llm: LLM):
    llm.register_mock("cleaner", lambda hint: {**DEFAULT, "rationale": "Default: 1% winsorisation, flat fill for gaps, "
                                                "fundamentals lagged one quarter to avoid lookahead."})


def clean(panel: Panel, llm: LLM, ledger: Ledger, run_id: str, search_end: int | None = None, seed: int = 0) -> tuple[Panel, dict]:
    """Returns (cleaned panel, data-quality report). Thresholds are computed on the search window only."""
    prof = profile(panel)
    prompt = ("You are the Cleaner. Choose data-cleaning options for a daily equity panel. Each is a forking path and will "
              f"be counted against the search.\nProfile: {prof}\nOptions: {OPTIONS}\nReturn JSON with your choices and a rationale.")
    choice, usage = llm.complete_json("cleaner", "You clean financial data conservatively.", prompt, CHOICE_SCHEMA,
                                      mock_hint={"profile": prof})
    out = copy.copy(panel)
    out.fields = {k: v.copy() for k, v in panel.fields.items()}
    R = out.fields["ret"]
    n_end = search_end or R.shape[0]
    rep = {"profile": prof, "choices": {k: choice[k] for k in OPTIONS}, "rationale": choice.get("rationale", ""), "steps": []}

    # 1. winsorize (tails estimated on the search window only: never peek at the held-out slice)
    lo, hi = np.nanpercentile(R[:n_end][panel.alive[:n_end]], [choice["winsorize_pct"], 100 - choice["winsorize_pct"]])
    clipped = int((np.isfinite(R) & ((R < lo) | (R > hi))).sum())
    R = np.where(np.isfinite(R), np.clip(R, lo, hi), np.nan)
    rep["steps"].append({"step": "winsorize", "pct": choice["winsorize_pct"], "bounds": [float(lo), float(hi)], "rows_affected": clipped})
    ledger.record_cleaning(run_id, "cleaner", "winsorize_pct", choice["winsorize_pct"], len(OPTIONS["winsorize_pct"]), clipped, seed)

    # 2. gaps (missing returns on days the name is tradable)
    gaps = int((~np.isfinite(R) & panel.alive).sum())
    if choice["gap_fill"] == "fill_zero":
        R = np.where(~np.isfinite(R) & panel.alive, 0.0, R)
    rep["steps"].append({"step": "gap_fill", "mode": choice["gap_fill"], "rows_affected": gaps})
    ledger.record_cleaning(run_id, "cleaner", "gap_fill", choice["gap_fill"], len(OPTIONS["gap_fill"]), gaps, seed)
    out.fields["ret"] = R

    # 3. point-in-time fundamentals
    L = int(choice["fundamental_lag_days"])
    has_btm = "btm" in out.fields and np.isfinite(out.fields["btm"]).any()
    if has_btm and L > 0:
        b = out.fields["btm"]
        out.fields["btm"] = np.vstack([np.full((L, b.shape[1]), np.nan), b[:-L]])
    if has_btm:
        rep["steps"].append({"step": "fundamental_lag", "days": L, "rows_affected": int(out.fields["btm"].size) if L else 0})
        ledger.record_cleaning(run_id, "cleaner", "fundamental_lag_days", L, len(OPTIONS["fundamental_lag_days"]), 0, seed)
    if has_btm and L == 0:
        rep["warning"] = "Fundamentals used without a reporting lag: value-type signals may contain lookahead."

    out.factors = known_factors(out)          # also sets out.factor_names to the factors that exist for this data
    out.meta = {**panel.meta, "cleaning": rep["choices"]}
    return out, rep
