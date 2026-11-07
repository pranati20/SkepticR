"""Prosecutor swarm. Each attack targets ONE failure mode and returns
    {attack, severity: low|medium|high, evidence: {...}, survived: bool}
Deterministic attacks run on pnl/series with zero tokens and RE-RUN the hypothesis when the Search allows it
(different universe, extra lag, perturbed windows) instead of trusting labels. An LLM-assisted spec review adds notes.
Only HIGH-severity failed attacks can kill a finding; the rest are reported as warnings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..court import stats as S
from ..features_dsl import parse, scale_windows

ANN = 252


@dataclass
class AttackConfig:
    cost_bps: float = 5.0            # one-way cost + slippage in basis points of traded notional
    min_net_sr_frac: float = 0.4     # net Sharpe must keep at least this share of gross
    alpha_t_min: float = 2.0         # factor-adjusted alpha t-stat floor (confounder attack)
    alpha_p_max: float = 0.05        # ...and its search-adjusted p-value (null worlds on the alpha statistic)
    top_day_frac: float = 0.01       # remove the best 1% of days
    conc_keep: float = 0.35          # mean pnl must keep at least this share without them
    ic_next_max: float = 0.08        # implausible daily IC of a feature with next-day return => leakage
    frag_keep: float = 0.4           # median neighbour Sharpe must keep this share of base
    surv_keep: float = 0.85          # PIT-universe Sharpe must keep this share of the reported one (>15% inflation = bias)
    year_pos_frac: float = 0.5       # >= this share of years must be positive


def _sr(x):
    x = np.asarray(x, float)
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(ANN)) if sd > 0 else 0.0


def result(attack, severity, survived, **evidence):
    return {"attack": attack, "severity": severity, "survived": bool(survived), "evidence": evidence}


# ------------------------------------------------------------------------------------------------ attacks
def leakage(search, k, cfg: AttackConfig):
    h = search.hyps[k]
    spec = h["spec"]
    reasons = []
    if spec.get("uses_future_bar"):
        reasons.append("spec reads future data (lead()/next-bar value)")
    ic = None
    if search.feature_fn is not None and search.realized_ret is not None and not reasons:
        F = search.feature_fn(k)
        R = search.realized_ret
        Rn = np.vstack([R[1:], np.full((1, R.shape[1]), np.nan)])
        cs = []
        for t in range(0, F.shape[0] - 1, 3):
            m = np.isfinite(F[t]) & np.isfinite(Rn[t])
            if m.sum() > 20 and np.std(F[t][m]) > 0:
                cs.append(stats.spearmanr(F[t][m], Rn[t][m])[0])
        ic = float(np.nanmean(cs)) if cs else 0.0
        if abs(ic) > cfg.ic_next_max:
            reasons.append(f"implausible rank-IC with next-day return ({ic:.3f}); real daily equity IC is ~0.01-0.03")
    return result("leakage", "high", not reasons, reasons=reasons, ic_next_day=ic,
                  test="AST lookahead scan + next-day IC plausibility")


def survivorship(search, k, cfg: AttackConfig):
    spec = search.hyps[k]["spec"]
    uni = spec.get("universe", "point_in_time")
    if uni == "point_in_time" and search.meta.get("survivorship_bias") and search.rerun is not None:
        # the DATA itself only contains today's survivors; we cannot re-run on the true universe, so stress it:
        # remove the 20% of names with the best hindsight return and see how much of the edge was riding on them
        base = _sr(search.pnl[k])
        trimmed = _sr(search.rerun(k, universe="trim_winners"))
        keep = trimmed / base if base > 0 else 0.0
        return result("survivorship", "medium", keep >= 0.5, universe="today's constituents (survivors only)",
                      reported_sharpe=base, sharpe_without_hindsight_winners=trimmed, edge_retained=float(keep),
                      note="Dataset contains only companies that are still listed. Treat the edge as overstated; this is a stress test, "
                           "not a correction, because delisted names are unavailable.")
    if uni == "point_in_time":
        return result("survivorship", "high", True, universe=uni, note="uses point-in-time constituents")
    ev = {"universe": uni, "reported_sharpe": _sr(search.pnl[k])}
    if search.rerun is not None:
        pit = search.rerun(k, universe="point_in_time")
        ev["point_in_time_sharpe"] = _sr(pit)
        keep = ev["point_in_time_sharpe"] / ev["reported_sharpe"] if ev["reported_sharpe"] > 0 else 0.0
        ev["edge_retained"] = float(keep)
        return result("survivorship", "high", keep >= cfg.surv_keep, **ev,
                      test="re-ran on the full point-in-time universe including delisted names")
    return result("survivorship", "high", False, **ev, note="universe is today's constituents: survivorship bias by construction")


def costs(search, k, cfg: AttackConfig):
    pnl = search.pnl[k]
    turn = float(search.turnover[k])
    cost_day = turn * cfg.cost_bps / 1e4
    net = pnl - cost_day
    g, n = _sr(pnl), _sr(net)
    t_net = float(S.tstat(net[None, :])[0])
    ok = n > 0 and g > 0 and n >= cfg.min_net_sr_frac * g
    return result("costs", "high", ok, gross_sharpe=g, net_sharpe=n, net_t=t_net, turnover_per_day=turn,
                  cost_bps=cfg.cost_bps, edge_per_day_bps=float(pnl.mean() * 1e4), cost_per_day_bps=float(cost_day * 1e4))


def regime(search, k, cfg: AttackConfig):
    pnl = search.pnl[k]
    T = len(pnl)
    ev = {}
    # calendar years (or equal chunks when no dates)
    if search.dates is not None and len(search.dates) == T:
        yrs = np.asarray(search.dates.year)
        labels = sorted(set(yrs))
        groups = {str(y): pnl[yrs == y] for y in labels if (yrs == y).sum() > 60}
    else:
        groups = {f"part{i+1}": c for i, c in enumerate(np.array_split(pnl, 5))}
    srs = {y: _sr(x) for y, x in groups.items()}
    ev["period_sharpes"] = srs
    # volatility regimes from the market factor
    if search.factors is not None and search.factors.shape[0] == T and search.factors.shape[1] > 0 and np.ptp(search.factors[:, 0]) > 0:
        mv = np.abs(search.factors[:, 0])
        thr = np.median(mv)
        ev["high_vol_sharpe"] = _sr(pnl[mv > thr])
        ev["low_vol_sharpe"] = _sr(pnl[mv <= thr])
    pos = np.mean([v > 0 for v in srs.values()]) if srs else 1.0
    worst = min(srs, key=srs.get) if srs else None
    tot = pnl.sum()
    top_share = max((x.sum() / tot for x in groups.values()), default=0.0) if tot > 0 else 1.0
    ok = pos >= cfg.year_pos_frac and top_share < 0.8
    ev.update(positive_period_frac=float(pos), worst_period=worst, worst_sharpe=srs.get(worst) if worst else None,
              max_single_period_share_of_pnl=float(top_share))
    return result("regime", "high", ok, **ev,
                  note=(f"Worked in {pos:.0%} of periods; worst: {worst}" if worst else ""))


def concentration(search, k, cfg: AttackConfig):
    pnl = search.pnl[k]
    n_drop = max(1, int(len(pnl) * cfg.top_day_frac))
    order = np.argsort(pnl)
    kept = pnl[order[:-n_drop]]
    base, after = pnl.mean(), kept.mean()
    keep = after / base if base > 0 else 0.0
    return result("concentration", "high", keep >= cfg.conc_keep, days_removed=n_drop,
                  mean_edge_retained=float(keep), sharpe_after=_sr(kept), sharpe_before=_sr(pnl),
                  top_days_share_of_pnl=float(pnl[order[-n_drop:]].sum() / pnl.sum()) if pnl.sum() > 0 else None)


def confounder(search, k, cfg: AttackConfig):
    pnl = search.pnl[k][None, :]
    F = search.factors
    t_alpha = float(S.alpha_t(pnl, F)[0])
    t_raw = float(S.tstat(pnl)[0])
    betas = S.factor_betas(pnl, F)[0]
    X = np.column_stack([np.ones(F.shape[0]), F])
    coef = np.linalg.lstsq(X, pnl[0], rcond=None)[0]
    r2 = 1 - ((pnl[0] - X @ coef) ** 2).sum() / ((pnl[0] - pnl[0].mean()) ** 2).sum()
    p_adj = None
    if getattr(search, "alpha_adj_p", None) is not None and np.isfinite(search.alpha_adj_p[k]):
        p_adj = float(search.alpha_adj_p[k])            # alpha t re-audited across the WHOLE search (null worlds)
    ok = t_alpha >= cfg.alpha_t_min and (p_adj is None or p_adj <= cfg.alpha_p_max)
    return result("confounder", "high", ok, raw_t=t_raw, alpha_t=t_alpha, alpha_p_adj_search=p_adj, r2_factors=float(r2),
                  betas={n: float(b) for n, b in zip(search.factor_names, betas)},
                  note="alpha vs MKT/SIZE/VAL/MOM; if alpha vanishes it was beta")


def fragility(search, k, cfg: AttackConfig):
    h = search.hyps[k]
    f = h["spec"].get("formula")
    if f is None or search.rerun is None:
        return result("fragility", "low", True, note="not applicable (no re-runnable parameters)")
    base = _sr(search.pnl[k])
    srs = {}
    for fac in (0.7, 0.85, 1.2, 1.4):
        try:
            g = scale_windows(f, fac)
            srs[f"x{fac}"] = _sr(search.rerun(k, formula=g))
        except Exception:
            continue
    if not srs:
        return result("fragility", "low", True, note="no tunable windows")
    med = float(np.median(list(srs.values())))
    ok = base <= 0 or (med >= cfg.frag_keep * base and np.mean([v > 0 for v in srs.values()]) >= 0.75)
    return result("fragility", "high", ok, base_sharpe=base, neighbour_sharpes=srs, median_neighbour=med)


ATTACKS = {"leakage": leakage, "survivorship": survivorship, "costs": costs, "regime": regime,
           "concentration": concentration, "confounder": confounder, "fragility": fragility}


def prosecute(search, k, cfg: AttackConfig | None = None, only: list | None = None, llm_review=None) -> list[dict]:
    cfg = cfg or AttackConfig()
    out = []
    fns = dict(ATTACKS)
    if getattr(search, "attack_fns", None) is not None:
        fns = {f.__name__: f for f in search.attack_fns}
    for f in (getattr(search, "extra_attack_fns", None) or []):
        fns[f.__name__] = f
    for name, fn in fns.items():
        if only and name not in only:
            continue
        try:
            out.append(fn(search, k, cfg))
        except Exception as e:                         # an attack that crashes must never silently pass a finding
            out.append(result(name, "high", False, error=str(e)))
    if llm_review is not None:
        out.append(llm_review(search, k))
    return out
