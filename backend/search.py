"""Search: the container the Court audits. One row per hypothesis the swarm tried.

A Search is domain-agnostic: pnl-like per-hypothesis series (or per-event outcomes laid on a timeline), factors for
the confounder attack, the held-out slice, and optional callbacks that let Prosecutors RE-RUN a hypothesis
(different universe, extra lag, perturbed window) instead of trusting labels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .adapters.market_panel import Panel, backtest, driver_pnls, truth_label
from .adapters.synthetic import SynthData
from .features_dsl import parse, DSLError


@dataclass
class Search:
    hyps: list                       # [{id, family, description, prior, agent, spec:{...}}]
    pnl: np.ndarray | None           # (K,T) search window (None for tabular domains)
    pnl_hold: np.ndarray | None      # (K,H) never touched by search
    factors: np.ndarray | None       # (T,f)
    factors_hold: np.ndarray | None
    factor_names: list
    turnover: np.ndarray | None      # (K,) mean daily one-way turnover
    dates: object = None             # DatetimeIndex for the search window (regime/year splits)
    truth: np.ndarray | None = None  # per-hypothesis ground truth label if known: 'alpha','beta','trap','null'
    rerun: Callable | None = None    # rerun(k, universe=None, formula=None, extra_lag=0) -> pnl (T,)
    feature_fn: Callable | None = None   # feature_fn(k) -> (T,N) causal feature, for IC diagnostics
    realized_ret: np.ndarray | None = None
    alpha_adj_p: np.ndarray | None = None   # search-adjusted p of the factor-adjusted ALPHA t (filled by the Court)
    # ---- domain hooks (tabular / betting / A-B / health / upload). All optional; defaults are the pnl pipeline.
    observed: np.ndarray | None = None      # (K,) signed statistics when not derivable from pnl
    n_obs: int | None = None                # observations behind the statistics (power calc)
    null_fn: Callable | None = None         # null_fn(cum_idx, n_worlds, seed) -> NullResult over hypotheses cum_idx
    attack_fns: list | None = None          # replaces the default attack list: [fn(search,k,cfg)->result]
    extra_attack_fns: list | None = None    # appended to the default list
    oos_fn: Callable | None = None          # oos_fn(k) -> {n_days, oos_sharpe, t, p_one_sided, confirmed}
    unit: str = "days"                      # unit of "more data needed"
    literature: dict | None = None          # Librarian evidence per hypothesis id (fed to the literature attack)
    domain: str = "market"
    meta: dict = field(default_factory=dict)

    @property
    def K(self):
        return len(self.hyps)

    @property
    def T(self):
        return self.n_obs if self.n_obs is not None else self.pnl.shape[1]

    def obs(self) -> np.ndarray:
        if self.observed is not None:
            return self.observed
        from .court import stats as _S
        return _S.tstat(self.pnl)

    def is_true(self) -> np.ndarray | None:
        return None if self.truth is None else self.truth == "alpha"


def from_synthetic(d: SynthData, seed: int = 0) -> Search:
    """Wrap the synthetic benchmark. Priors are assigned INDEPENDENTLY of truth (no peeking):
    ~20% of ideas are 'literature-motivated' (prior 0.8), the rest random (0.3)."""
    rng = np.random.default_rng(seed + 777)
    hyps = []
    truth = []
    for i, m in enumerate(d.meta):
        lit = rng.random() < 0.2
        hyps.append({"id": m["id"], "family": m["family"], "description": f"{m['family']} signal #{i}",
                     "prior": 0.8 if lit else 0.3, "agent": f"{m['family']}-hunter",
                     "source_paper": "literature" if lit else None, "spec": dict(m["spec"])})
        truth.append("alpha" if m["kind"] == "planted" else "trap" if m["kind"].startswith("trap")
                     else "beta" if m["kind"] == "decoy" else "null")
    return Search(hyps=hyps, pnl=d.pnl, pnl_hold=d.pnl_holdout, factors=d.factors, factors_hold=d.factors_holdout,
                  factor_names=d.factor_names, turnover=d.turnover, truth=np.array(truth), domain="synthetic",
                  meta={"synthetic": True})


def build_market_search(panel: Panel, hyps: list, holdout: int = 252, progress: Callable | None = None) -> Search:
    """Backtest every hypothesis (zero tokens) and assemble the Search. The last `holdout` days are never used by search."""
    Tn = panel.shape[0]
    Ts = Tn - holdout
    pnl = np.zeros((len(hyps), Tn))
    turn = np.zeros(len(hyps))
    truth = []
    drivers = driver_pnls(panel)
    for i, h in enumerate(hyps):
        sp = h["spec"]
        try:
            b = backtest(panel, sp["formula"], universe=sp.get("universe", "point_in_time"), hold=sp.get("hold", 1))
        except Exception:
            b = {"pnl": np.zeros(Tn), "turnover": 0.0, "feature": np.full(panel.shape, np.nan)}
        pnl[i] = b["pnl"]
        turn[i] = b["turnover"]
        lab, _ = truth_label(b["pnl"], drivers, panel.meta)
        truth.append(lab)
        if progress:
            progress(i + 1, len(hyps))
    cache_formula = {}

    def rerun(k, universe=None, formula=None, extra_lag=0):
        sp = hyps[k]["spec"]
        r = backtest(panel, formula or sp["formula"], universe=universe or sp.get("universe", "point_in_time"),
                     hold=sp.get("hold", 1), extra_lag=extra_lag)
        return r["pnl"][:Ts]

    def feature_fn(k):
        from .features_dsl import evaluate
        return evaluate(hyps[k]["spec"]["formula"], panel.as_dsl())[:Ts]

    F = panel.factors
    return Search(hyps=hyps, pnl=pnl[:, 1:Ts], pnl_hold=pnl[:, Ts:], factors=F[1:Ts], factors_hold=F[Ts:],
                  factor_names=panel.factor_names, turnover=turn, dates=panel.dates[1:Ts], truth=np.array(truth),
                  rerun=lambda k, **kw: rerun(k, **kw)[1:], feature_fn=feature_fn,
                  realized_ret=panel.fields["ret"][:Ts], domain="market", meta=dict(panel.meta))
