"""Sports-betting adapter (Kaggle 'Sports Betting Predictive Analysis Dataset 2025').

Claim under audit: "betting the model's Predicted_Winner has an edge." Each hypothesis = a segment rule (sport, side,
odds bucket, weekday, month, ...). Daily pnl = flat 1-unit stakes at the listed odds.
NOTE: this file looks machine-generated (synthetic team names, uniformly drawn odds). Everything built on it is
flagged as a simulated stand-in: it says nothing about real betting markets.
The defining Prosecutor here is the PLACEBO: does betting a random side earn the same? If yes, the 'edge' is mispriced
odds, not predictive skill.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..ledger import Ledger
from ..court import null_worlds as NW, stats as CS
from ..prosecutors.attacks import concentration, regime
from ..search import Search

DATA = Path(__file__).resolve().parents[2] / "data" / "sports_betting" / "sports_betting_predictive_analysis.csv"


def load(path: Path | str = DATA, ledger: Ledger | None = None, run_id: str | None = None) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, parse_dates=["Date"])
    n0 = len(df)
    df = df.drop_duplicates("Match_ID")
    pred_home = df.Predicted_Winner == df.Home_Team
    df["pred_odds"] = np.where(pred_home, df.Home_Team_Odds, df.Away_Team_Odds)
    df["other_odds"] = np.where(pred_home, df.Away_Team_Odds, df.Home_Team_Odds)
    n_missing = int(df.pred_odds.isna().sum())
    df = df.dropna(subset=["pred_odds"]).copy()           # cleaning choice: drop bets we cannot price (vs impute)
    pred_home = df.Predicted_Winner == df.Home_Team
    if ledger is not None:
        ledger.record_cleaning(run_id, "cleaner", "missing_odds", "drop_rows", 2, n_missing)
        ledger.record_cleaning(run_id, "cleaner", "draw_handling", "draw_loses_bet", 2, int((df.Actual_Winner == "Draw").sum()))
    df["win"] = df.Predicted_Winner == df.Actual_Winner
    df["profit"] = np.where(df.win, df.pred_odds - 1.0, -1.0)
    # placebo: expected profit of a coin-flip between the two sides (needs both odds; else fall back to own-side)
    other_win = (~df.win) & (df.Actual_Winner != "Draw")
    other_profit = np.where(other_win, df.other_odds - 1.0, -1.0)
    df["placebo"] = np.where(df.other_odds.notna(), 0.5 * (df.profit + other_profit), df.profit)
    df["side"] = np.where(pred_home, "home", "away")
    df["odds_bucket"] = pd.cut(df.pred_odds, [0, 2.0, 3.0, 4.0, 10], labels=["<2", "2-3", "3-4", "4+"]).astype(str)
    df["fav"] = np.where(df.other_odds.isna(), "unknown", np.where(df.pred_odds < df.other_odds, "favourite", "underdog"))
    df["dow"] = df.Date.dt.day_name()
    df["month"] = df.Date.dt.strftime("%b")
    df["half"] = df.Date.dt.year.astype(str) + "H" + ((df.Date.dt.month > 6) + 1).astype(str)
    info = {"rows": n0, "bets": len(df), "dropped_missing_odds": n_missing, "date_min": str(df.Date.min().date()),
            "date_max": str(df.Date.max().date()), "is_standin": True,
            "note": "Dataset appears synthetic (machine-generated team names, odds spread uniformly 1.2-5.0, "
                    "favourites win 43%). Treat all results as a demonstration, not evidence about real markets.",
            "naive_roi": float(df.profit.mean()), "placebo_roi": float(df.placebo.mean())}
    return df, info


def rules(df: pd.DataFrame) -> list[tuple[str, np.ndarray, str]]:
    out = [("all predicted bets", np.ones(len(df), bool), "overall")]
    for col, fam in (("Sport", "sport"), ("side", "side"), ("odds_bucket", "odds"), ("fav", "price"),
                     ("dow", "calendar"), ("month", "calendar"), ("half", "calendar")):
        for v in sorted(df[col].unique()):
            m = (df[col] == v).to_numpy()
            if m.sum() >= 25:
                out.append((f"{col}={v}", m, fam))
    for sp in sorted(df.Sport.unique()):
        for col in ("side", "odds_bucket", "fav"):
            for v in sorted(df[col].unique()):
                m = ((df.Sport == sp) & (df[col] == v)).to_numpy()
                if m.sum() >= 25:
                    out.append((f"Sport={sp} & {col}={v}", m, "sport"))
    return out


def build_search(df: pd.DataFrame, info: dict, holdout_days: int = 120) -> Search:
    days = pd.date_range(df.Date.min(), df.Date.max(), freq="D")
    T_all = len(days)
    didx = (df.Date - days[0]).dt.days.to_numpy()
    rs = rules(df)
    K = len(rs)
    pnl, plc = np.zeros((K, T_all)), np.zeros((K, T_all))
    for k, (name, m, fam) in enumerate(rs):
        np.add.at(pnl[k], didx[m], df.profit.to_numpy()[m])
        np.add.at(plc[k], didx[m], df.placebo.to_numpy()[m])
    Ts = T_all - holdout_days
    hyps = [{"id": f"bet|{name}", "family": fam, "agent": f"{fam}-researcher", "description": f"Bet the predicted winner when {name}",
             "prior": 0.5 if k == 0 else 0.25,
             "spec": {"formula": None, "rule": name, "universe": "point_in_time", "uses_future_bar": False, "n_bets": int(m.sum())}}
            for k, (name, m, fam) in enumerate(rs)]
    placebo = plc[:, :Ts]

    _skill = {}

    def placebo_attack(search, k, cfg):
        a, b = search.pnl[k], placebo[k]
        if "p" not in _skill:        # audit the SKILL statistic (predicted - random side) across the whole search too
            r = NW.run_null_worlds(search.pnl - placebo, np.zeros((search.pnl.shape[1], 1)), CS.raw_t, "stationary_bootstrap", 300, seed=7)
            _skill["p"] = r.p_stepdown
        p_skill = float(_skill["p"][k])
        n_bets = max(1, hyps[k]["spec"]["n_bets"])
        diff = a - b
        t_skill = float(diff.mean() / (diff.std(ddof=1) + 1e-12) * np.sqrt(len(diff)))
        ok = t_skill >= 2.0 and p_skill <= 0.05 and b.sum() < 0.6 * a.sum()
        return {"attack": "placebo", "severity": "high", "survived": bool(ok), "evidence": {
            "roi_predicted_pct": float(100 * a.sum() / n_bets), "roi_random_side_pct": float(100 * b.sum() / n_bets),
            "skill_t": t_skill, "skill_p_adj_search": p_skill,
            "note": "Predictions beat a random side." if ok else
                    "A coin-flip between the two sides earns about the same: the profit comes from how the odds are set, "
                    "not from the model's predictions."}}
    placebo_attack.__name__ = "placebo"

    def oos(k):
        x = pnl[k, Ts:]
        n = len(x)
        sd = x.std(ddof=1)
        t = float(x.mean() / sd * np.sqrt(n)) if sd > 0 else 0.0
        p = float(stats.t.sf(t, n - 1))
        return {"n_days": n, "oos_sharpe": float(x.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0, "t": t,
                "p_one_sided": p, "confirmed": bool(p < 0.05 and x.mean() > 0)}

    return Search(hyps=hyps, pnl=pnl[:, :Ts], pnl_hold=pnl[:, Ts:], factors=np.zeros((Ts, 1)),
                  factors_hold=np.zeros((T_all - Ts, 1)), factor_names=["none"], turnover=np.zeros(K), dates=days[:Ts],
                  domain="betting", attack_fns=[regime, concentration, placebo_attack], oos_fn=oos,
                  meta={**info, "claim": "Betting the model's predicted winner has an edge.", "unit": "days"})
