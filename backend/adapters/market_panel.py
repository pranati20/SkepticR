"""Daily equity panel: container + simulated STAND-IN world with planted ground truth + backtester.

The simulated panel is a stand-in (flagged `is_standin=True` everywhere in the UI). It is used by the live swarm demo
because it has KNOWN truth:
  planted real alpha :  reversal10 (short-term reversal), lowvol60 (low-volatility), volshock (volume shock fade)
  planted factor premia (beta, NOT alpha): value (btm), momentum (12-1)  -> the confounder attack must kill these
  nothing: sentiment, calendar effects, most momentum windows -> any "discovery" there is luck
It also has dirty data (gaps, bad ticks), delisted names (survivorship), and slowly reported fundamentals.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features_dsl import evaluate, parse

warnings.filterwarnings('ignore', category=RuntimeWarning)


@dataclass
class Panel:
    fields: dict                    # close, ret, volume, btm, sentiment : (T,N)
    dates: pd.DatetimeIndex
    tickers: list
    alive: np.ndarray               # (T,N) bool: tradable point-in-time (listed and not delisted)
    listed_today: np.ndarray        # (N,) bool: still listed at the end (the survivor-biased universe)
    factors: np.ndarray | None = None          # (T,4) known factor returns, aligned to ret dates
    factor_names: list = field(default_factory=lambda: ["MKT", "SIZE", "VAL", "MOM"])
    truth: dict | None = None       # planted drivers {name: (T,N)} -- stand-in only
    meta: dict = field(default_factory=dict)

    def as_dsl(self) -> dict:
        d = dict(self.fields)
        d["dates"] = self.dates
        return d

    @property
    def shape(self):
        return self.fields["close"].shape


# --------------------------------------------------------------------------------------------- backtester
def _ls_weights(F: np.ndarray, mask: np.ndarray, q: float = 0.2) -> np.ndarray:
    ok = mask & np.isfinite(F)
    X = np.where(ok, F, np.nan)
    pct = pd.DataFrame(X).rank(axis=1, pct=True).to_numpy()
    long = (pct >= 1 - q) & ok
    short = (pct <= q) & ok
    nl, ns = long.sum(1, keepdims=True), short.sum(1, keepdims=True)
    w = np.where(long, 0.5 / np.maximum(nl, 1), 0.0) - np.where(short, 0.5 / np.maximum(ns, 1), 0.0)
    w[(nl[:, 0] < 3) | (ns[:, 0] < 3)] = 0.0
    return w


def backtest(panel: Panel, formula: str, universe: str = "point_in_time", q: float = 0.2, hold: int = 1,
             ret: np.ndarray | None = None, extra_lag: int = 0) -> dict:
    """Dollar-neutral top/bottom-quantile backtest. Feature at close of t earns the return of t+1 (causal), unless the
    formula itself peeks (`lead`). Returns daily pnl aligned to the return date."""
    F = evaluate(formula, panel.as_dsl())
    return backtest_feature(panel, F, universe, q, hold, ret, extra_lag)


def backtest_feature(panel: Panel, F: np.ndarray, universe: str = "point_in_time", q: float = 0.2, hold: int = 1,
                     ret: np.ndarray | None = None, extra_lag: int = 0) -> dict:
    R = panel.fields["ret"] if ret is None else ret
    if universe == "point_in_time":
        mask = panel.alive
    elif universe == "trim_winners":          # stress: drop the 20% of names with the best full-sample return (hindsight winners)
        c = panel.fields["close"]
        tot = np.array([(c[np.isfinite(c[:, j]), j][-1] / c[np.isfinite(c[:, j]), j][0]) if np.isfinite(c[:, j]).sum() > 2 else np.nan for j in range(c.shape[1])])
        keep = tot < np.nanquantile(tot, 0.8)
        mask = panel.alive & keep[None, :]
    else:
        mask = panel.alive & panel.listed_today[None, :]
    w = _ls_weights(F, mask, q)
    if hold > 1:
        idx = (np.arange(w.shape[0]) // hold) * hold
        w = w[idx]
    if extra_lag:
        w = np.vstack([np.zeros((extra_lag, w.shape[1])), w[:-extra_lag]])
    Rn = np.vstack([R[1:], np.full((1, R.shape[1]), np.nan)])        # return earned after the weight is set at t
    pnl_t = np.nansum(w * np.where(np.isfinite(Rn), Rn, 0.0), axis=1)  # aligned to decision date t
    turn = np.abs(np.diff(w, axis=0)).sum(1)
    pnl = np.concatenate([[0.0], pnl_t[:-1]])                         # shift to the realisation date
    pnl[-1] = pnl[-1]
    return {"pnl": pnl, "turnover": float(turn[np.any(w[1:] != 0, axis=1)].mean()) if turn.size else 0.0,
            "feature": F, "w": w}


def known_factors(panel: Panel) -> np.ndarray:
    """MKT (equal-weight), SIZE (small-minus-big proxy), VAL (btm, if fundamentals exist), MOM (12-1): built from the
    panel itself, causal. Sets panel.factor_names to the factors actually available."""
    R = panel.fields["ret"]
    mkt = np.nan_to_num(np.nanmean(np.where(panel.alive, R, np.nan), axis=1))
    cols, names = [mkt], ["MKT"]
    spec = [("SIZE", "-log(rolling_mean(close * volume, 60))"), ("VAL", "btm"), ("MOM", "ts_return(lag(close, 21), 231)")]
    for nm, f in spec:
        if nm == "VAL" and not np.isfinite(panel.fields.get("btm", np.full(1, np.nan))).any():
            continue
        cols.append(backtest(panel, f, universe="point_in_time")["pnl"]); names.append(nm)
    panel.factor_names = names
    return np.column_stack(cols)


# --------------------------------------------------------------------------------------------- simulator
def simulate_market_panel(seed: int = 0, n_days: int = 1500, n_assets: int = 150, planted: bool = True,
                          decay_after: int | None = None, decay_to: float = 0.25) -> Panel:
    rng = np.random.default_rng(seed)
    T, N = n_days, n_assets
    dates = pd.bdate_range("2018-01-01", periods=T)
    g = np.zeros(T)
    sh = rng.normal(size=T)
    for t in range(1, T):
        g[t] = 0.98 * g[t - 1] + 0.06 * sh[t]
    h = np.exp(g)
    tn = lambda shape: rng.standard_t(4, size=shape) / np.sqrt(2)
    beta = rng.normal(1.0, 0.3, N)
    sig = 0.014 * np.exp(0.3 * rng.normal(size=N))
    sector = rng.integers(0, 10, N)
    mkt = 0.0003 + 0.009 * h * tn(T)
    sec = 0.004 * h[:, None] * tn((T, 10))
    eps = sig * h[:, None] * tn((T, N))

    # slow fundamentals: book-to-market latent (persistent), reported with a lag in the raw data
    btm_lat = np.zeros((T, N)); x = rng.normal(size=N)
    for t in range(T):
        x = 0.998 * x + np.sqrt(1 - 0.998 ** 2) * rng.normal(size=N)
        btm_lat[t] = x
    # delistings: 15% of names crash then vanish
    delist_day = np.full(N, 10 ** 9)
    dl = rng.random(N) < 0.22
    delist_day[dl] = rng.integers(int(T * 0.4), int(T * 0.95), dl.sum())

    # planted effect strengths expressed as target annualised Sharpe -> IC, a = SR / sqrt(N*252)
    ic = lambda sr: sr / np.sqrt(N * 252)
    strength = {"reversal10": ic(4.2), "lowvol60": ic(4.5), "volshock": ic(1.8), "value": ic(1.6), "momentum": ic(1.6)} \
        if planted else {}
    zs = lambda v: (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)

    ret = np.zeros((T, N))
    close = np.zeros((T, N)); vol = np.zeros((T, N))
    close[0] = 100 * np.exp(rng.normal(0, 0.3, N))
    volbase = np.exp(rng.normal(12, 0.8, N))
    volume = np.zeros((T, N)); volume[0] = volbase
    drivers = {k: np.full((T, N), np.nan) for k in strength}
    alive = np.ones((T, N), bool)
    for t in range(1, T):
        drift = np.zeros(N)
        if planted:
            if t > 11:
                d = zs(close[t - 1] / close[t - 11] - 1); drivers["reversal10"][t - 1] = -d; drift += strength["reversal10"] * (decay_to if (decay_after is not None and t >= decay_after) else 1.0) * -d
            if t > 62:
                rs = ret[t - 60:t].std(axis=0); d = zs(rs); drivers["lowvol60"][t - 1] = -d; drift += strength["lowvol60"] * -d
            if t > 22:
                vs = volume[t - 1] / volume[t - 21:t - 1].mean(axis=0); d = zs(vs); drivers["volshock"][t - 1] = -d; drift += strength["volshock"] * -d
            d = zs(btm_lat[t - 1]); drivers["value"][t - 1] = d; drift += strength["value"] * d
            if t > 252:
                d = zs(close[t - 22] / close[t - 252] - 1); drivers["momentum"][t - 1] = d; drift += strength["momentum"] * d
        r = beta * mkt[t] + sec[t, sector] + eps[t] + drift * sig * h[t]
        # distress crash + delisting
        crash = (t >= delist_day - 25) & (t < delist_day)
        r = np.where(crash, -0.016 + 0.5 * r, r)
        gone = t >= delist_day
        r = np.where(gone, np.nan, r)
        alive[t] = ~gone
        ret[t] = r
        prev = np.where(np.isnan(close[t - 1]), np.nan, close[t - 1])
        close[t] = prev * (1 + np.nan_to_num(r, nan=0.0))
        close[t][gone] = np.nan
        volume[t] = np.where(gone, np.nan, volbase * np.exp(rng.normal(0, 0.35, N) + 6 * np.abs(np.nan_to_num(r))))
    alive[0] = True
    ret[0] = np.nan
    listed_today = delist_day > T

    raw = {"close": close.copy(), "ret": ret.copy(), "volume": volume,
           "btm": btm_lat + 0.0, "sentiment": rng.normal(size=(T, N))}
    # ---- dirty data injection (the Cleaner has to deal with this; each choice is a forking path) ----
    n_gap = int(0.004 * T * N)
    gi = (rng.integers(1, T, n_gap), rng.integers(0, N, n_gap))
    raw["ret"][gi] = np.nan
    n_bad = int(0.0015 * T * N)
    bi = (rng.integers(1, T, n_bad), rng.integers(0, N, n_bad))
    raw["ret"][bi] = np.where(np.isnan(raw["ret"][bi]), np.nan, raw["ret"][bi] * rng.choice([-8.0, 8.0], n_bad))
    p = Panel(fields=raw, dates=dates, tickers=[f"S{i:03d}" for i in range(N)], alive=alive, listed_today=listed_today,
              truth={k: v for k, v in drivers.items()} if planted else None,
              meta={"source": "simulated stand-in", "is_standin": True, "seed": seed,
                    "note": "Simulated prices with planted effects and dirty data. Not real market data.",
                    "planted_alpha": ["reversal10", "lowvol60", "volshock"],
                    "planted_beta": ["value", "momentum"]})
    return p


def driver_pnls(panel: Panel) -> dict:
    """Daily pnl of each planted driver traded perfectly (the 'true effect' portfolios)."""
    if panel.truth is None:
        return {}
    return {k: backtest_feature(panel, D)["pnl"] for k, D in panel.truth.items()}


def truth_label(pnl: np.ndarray, drivers: dict, meta: dict, corr_alpha: float = 0.4, corr_beta: float = 0.5) -> tuple[str, str | None]:
    """Ground-truth label from how closely a hypothesis' pnl tracks the planted effects' pnl.
    'alpha' if it contains a planted alpha effect, 'beta' if it is (mostly) a planted factor premium, else 'null'."""
    if not drivers:
        return "null", None
    best = {}
    for k, d in drivers.items():
        if np.std(pnl) == 0 or np.std(d) == 0:
            continue
        best[k] = float(np.corrcoef(pnl[1:], d[1:])[0, 1])
    a = max(((c, k) for k, c in best.items() if k in meta.get("planted_alpha", [])), default=(0, None))
    b = max(((c, k) for k, c in best.items() if k in meta.get("planted_beta", [])), default=(0, None))
    if a[0] >= corr_alpha:
        return "alpha", a[1]
    if b[0] >= corr_beta:
        return "beta", b[1]
    return "null", None
