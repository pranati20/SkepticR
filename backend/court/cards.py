"""Strategy cards + portfolio view for surviving market findings (zero tokens)."""
from __future__ import annotations

import numpy as np
from scipy import stats

ANN = 252


def _sr(x):
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def max_drawdown(cum):
    peak = np.maximum.accumulate(cum)
    return float((cum - peak).min())


def _downsample(a, n=160):
    idx = np.unique(np.linspace(0, len(a) - 1, min(n, len(a))).astype(int))
    return [float(a[i]) for i in idx]


def deflated_sharpe(pnl, n_trials, sr_var):
    T = len(pnl)
    sr = np.mean(pnl) / np.std(pnl, ddof=1)
    gamma = 0.5772156649
    sr0 = np.sqrt(sr_var) * ((1 - gamma) * stats.norm.ppf(1 - 1 / max(n_trials, 2)) + gamma * stats.norm.ppf(1 - 1 / (max(n_trials, 2) * np.e)))
    sk, ku = stats.skew(pnl), stats.kurtosis(pnl, fisher=False)
    den = np.sqrt(max(1 - sk * sr + (ku - 1) / 4 * sr ** 2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / den))


def strategy_card(search, k, verdict: dict, n_trials: int, sr_var: float, cost_bps: float = 5.0) -> dict:
    pnl = search.pnl[k]
    turn = float(search.turnover[k])
    net = pnl - turn * cost_bps / 1e4
    cum_g, cum_n = np.cumsum(pnl), np.cumsum(net)
    ev = {a["attack"]: a for a in verdict.get("attacks", [])}
    reg = ev.get("regime", {}).get("evidence", {})
    return {
        "hypothesis_id": search.hyps[k]["id"],
        "equity_gross": _downsample(cum_g), "equity_net": _downsample(cum_n),
        "sharpe_gross": _sr(pnl), "sharpe_net": _sr(net), "deflated_sharpe_prob": deflated_sharpe(pnl, n_trials, sr_var),
        "max_drawdown_net": max_drawdown(cum_n), "annual_return_net": float(net.mean() * ANN),
        "annual_vol": float(net.std(ddof=1) * np.sqrt(ANN)), "turnover_per_day": turn,
        "capacity_note": ("Low turnover: capacity is mostly limited by liquidity of the traded names." if turn < 0.2 else
                          "High turnover: edge per trade is thin, so capacity is limited by costs and market impact."),
        "regime": reg.get("period_sharpes", {}), "high_vol_sharpe": reg.get("high_vol_sharpe"), "low_vol_sharpe": reg.get("low_vol_sharpe"),
        "disclaimer": "Backtest, not a forecast. No trading advice.",
    }


def portfolio_view(search, idx: list[int], cost_bps: float = 5.0) -> dict | None:
    if len(idx) < 1:
        return None
    net = np.array([search.pnl[k] - search.turnover[k] * cost_bps / 1e4 for k in idx])
    corr = np.corrcoef(net) if len(idx) > 1 else np.array([[1.0]])
    port = net.mean(axis=0)
    return {"ids": [search.hyps[k]["id"] for k in idx], "correlation": np.round(corr, 3).tolist(),
            "combined_sharpe": _sr(port), "mean_individual_sharpe": float(np.mean([_sr(x) for x in net])),
            "equity": _downsample(np.cumsum(port))}
