"""Tabular claims: a treatment/outcome table with many candidate subgroup effects (A/B tests, health claims, uploads).

The search = 'overall' + every subgroup (and 2-way intersections) x every metric the analyst looked at.
Statistic  : direction * (Welch t - its value under the null), direction = sign of the HEADLINE (raw) effect.
Null worlds: permute the treatment labels, WITHIN confounder strata when strata are given (so confounding is preserved
             and only the treatment->outcome link is destroyed); randomised experiments use a plain permutation.
Prosecutors: Simpson's paradox (does the effect reverse once stratified?), peeking (optional stopping), outlier
             concentration, stability across time halves.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..court import null_worlds as NW
from ..search import Search

MAX_SEG = 220


def _bin(col: pd.Series, k: int = 3) -> pd.Series:
    if col.dtype.kind in "if" and col.nunique() > 6:
        return pd.qcut(col, k, labels=[f"{col.name}:Q{i+1}" for i in range(k)], duplicates="drop").astype(str)
    return col.astype(str).map(lambda v: f"{col.name}={v}")


def segments(df: pd.DataFrame, cols: list[str], pairs: bool = True, min_n: int = 60) -> list[tuple[str, np.ndarray]]:
    out = [("overall", np.ones(len(df), bool))]
    lv = {c: _bin(df[c]) for c in cols}
    singles = []
    for c in cols:
        for v in sorted(lv[c].unique()):
            m = (lv[c] == v).to_numpy()
            if m.sum() >= min_n:
                singles.append((v, m)); out.append((v, m))
    if pairs:
        for (a, ma), (b, mb) in itertools.combinations(singles, 2):
            if a.split("=")[0].split(":")[0] == b.split("=")[0].split(":")[0]:
                continue
            m = ma & mb
            if m.sum() >= min_n * 2:
                out.append((f"{a} & {b}", m))
    return out[:MAX_SEG]


def _welch(M, w, Y):
    """t of (treated - control) within each mask row of M (K,n) for outcome matrix Y (n,) ; w treatment (n,)."""
    w = w.astype(float); c = 1 - w
    n1, n0 = M @ w, M @ c
    s1, s0 = M @ (w * Y), M @ (c * Y)
    q1, q0 = M @ (w * Y * Y), M @ (c * Y * Y)
    with np.errstate(all="ignore"):
        m1, m0 = s1 / n1, s0 / n0
        v1 = np.maximum(q1 / n1 - m1 ** 2, 1e-12) * n1 / np.maximum(n1 - 1, 1)
        v0 = np.maximum(q0 / n0 - m0 ** 2, 1e-12) * n0 / np.maximum(n0 - 1, 1)
        t = (m1 - m0) / np.sqrt(v1 / n1 + v0 / n0)
    t[(n1 < 5) | (n0 < 5)] = 0.0
    return t, m1 - m0


@dataclass
class TabularSpec:
    df: pd.DataFrame
    treat: str
    outcomes: list[str]
    subgroup_cols: list[str]
    strata_cols: list[str] | None = None
    time_col: str | None = None
    holdout_frac: float = 0.25
    claim: str = ""
    peek_looks: int = 0
    peekers: dict | None = None        # {(outcome, segment): look_index} claims the analyst reported at an interim look
    pairs: bool = True


def _strata_ids(df, cols):
    if not cols:
        return None
    return pd.factorize(df[cols].astype(str).agg("|".join, axis=1))[0]


def _perm_within(rng, w, strata):
    if strata is None:
        return rng.permutation(w)
    out = w.copy()
    for s in np.unique(strata):
        ix = np.where(strata == s)[0]
        out[ix] = rng.permutation(w[ix])
    return out


def build_tabular_search(spec: TabularSpec, seed: int = 0) -> Search:
    df = spec.df.reset_index(drop=True)
    n_all = len(df)
    if spec.time_col:
        df = df.sort_values(spec.time_col).reset_index(drop=True)
    n_search = int(n_all * (1 - spec.holdout_frac))
    S_df, H_df = df.iloc[:n_search].reset_index(drop=True), df.iloc[n_search:].reset_index(drop=True)
    segs = segments(S_df, spec.subgroup_cols, spec.pairs)
    seg_names = [s[0] for s in segs]
    M_all = np.array([m for _, m in segs], float)                      # (S,n_search)
    H_masks = {name: m for name, m in segments(H_df, spec.subgroup_cols, spec.pairs, min_n=1)}
    w = S_df[spec.treat].to_numpy().astype(float)
    strata = _strata_ids(S_df, spec.strata_cols)
    Yraw = np.column_stack([S_df[o].to_numpy().astype(float) for o in spec.outcomes])
    mu_y, sd_y = Yraw.mean(0), Yraw.std(0) + 1e-12
    Ys = (Yraw - mu_y) / sd_y
    Yh = (np.column_stack([H_df[o].to_numpy().astype(float) for o in spec.outcomes]) - mu_y) / sd_y
    wh = H_df[spec.treat].to_numpy().astype(float)

    hyps, idx_o, idx_s = [], [], []
    for oi, o in enumerate(spec.outcomes):
        for si, name in enumerate(seg_names):
            hyps.append({"id": f"{o}|{name}", "family": "overall" if name == "overall" else "subgroup", "agent": "segment-hunter",
                         "description": f"effect of {spec.treat} on {o} in {name}", "prior": 0.6 if name == "overall" else 0.25,
                         "spec": {"outcome": o, "segment": name, "universe": "point_in_time", "uses_future_bar": False}})
            idx_o.append(oi); idx_s.append(si)
    idx_o, idx_s = np.array(idx_o), np.array(idx_s)

    def stats_for(wv: np.ndarray, sel: np.ndarray) -> np.ndarray:
        """raw Welch t for hypotheses in `sel`, treatment vector wv."""
        t = np.zeros(len(sel))
        for oi in np.unique(idx_o[sel]):
            pick = np.where(idx_o[sel] == oi)[0]
            t[pick], _ = _welch(M_all[idx_s[sel][pick]], wv, Ys[:, oi])
        return t

    allsel = np.arange(len(hyps))
    t_raw, eff_raw = np.zeros(len(hyps)), np.zeros(len(hyps))
    for oi in range(len(spec.outcomes)):
        pk = np.where(idx_o == oi)[0]
        t_raw[pk], eff_raw[pk] = _welch(M_all[idx_s[pk]], w, Ys[:, oi])
    direction = np.where(t_raw >= 0, 1.0, -1.0)
    for k, h in enumerate(hyps):
        h["spec"]["direction"] = int(direction[k]); h["spec"]["headline_t"] = float(t_raw[k])

    # optional-stopping analyst: first interim look (global arrival order) at which |t| crossed 1.96
    cuts = [int(n_search * (j + 1) / spec.peek_looks) for j in range(spec.peek_looks)] if spec.peek_looks else []
    pre = np.arange(n_search)

    def t_prefix(wv, k, c):
        mm = (M_all[idx_s[k]] * (pre < c))[None, :]
        return float(_welch(mm, wv, Ys[:, idx_o[k]])[0][0])
    if cuts:
        for k, h in enumerate(hyps):
            for j, c in enumerate(cuts[:-1]):
                if abs(t_prefix(w, k, c)) > 1.96:
                    h["spec"]["peeked_at_look"] = j
                    break

    # null centre (expected raw t under the null world, e.g. the confounded association)
    _mu_cache = {}

    def null_centre(B=60, seed_=12345):
        if "mu" in _mu_cache:
            return _mu_cache["mu"]
        rng = np.random.default_rng(seed_)
        acc = np.zeros(len(hyps))
        for _ in range(B):
            acc += stats_for(_perm_within(rng, w, strata), allsel)
        _mu_cache["mu"] = acc / B if strata is not None else np.zeros(len(hyps))
        return _mu_cache["mu"]

    mu = null_centre()
    observed = direction * (t_raw - mu)

    def null_fn(cum, B, seed_):
        rng = np.random.default_rng(seed_)
        cum = np.asarray(cum)
        nul = np.empty((B, len(cum)))
        for b in range(B):
            nul[b] = np.abs(direction[cum] * (stats_for(_perm_within(rng, w, strata), cum) - mu[cum]))
        obs_abs = np.abs(observed[cum])
        p_single, p_step, p_unadj, nmax = NW.adjusted_pvalues(obs_abs, nul)
        return NW.NullResult(method="label_permutation" + ("_within_strata" if strata is not None else ""), n_worlds=B,
                             observed=observed[cum], null_stats=nul, p_single=p_single, p_stepdown=p_step,
                             p_unadjusted=p_unadj, null_max=nmax, search_size=len(cum))

    # ------------------------------------------------------------------ attacks (domain-specific)
    def _res(name, sev, ok, **ev):
        return {"attack": name, "severity": sev, "survived": bool(ok), "evidence": ev}

    def simpson(search, k, cfg):
        if strata is None:
            return _res("simpson", "high", True, note="randomised / no confounder strata supplied")
        m = M_all[idx_s[k]] > 0
        y = Ys[:, idx_o[k]]
        effs, wts, per = [], [], {}
        for s in np.unique(strata[m]):
            ix = m & (strata == s)
            a, b = ix & (w == 1), ix & (w == 0)
            if a.sum() >= 15 and b.sum() >= 15:
                e = y[a].mean() - y[b].mean()
                effs.append(e); wts.append(min(a.sum(), b.sum())); per[str(s)] = round(float(e), 3)
        if not effs:
            return _res("simpson", "high", True, note="too few cells to stratify")
        adj = float(np.average(effs, weights=wts))
        pooled = float(eff_raw[k])
        d = direction[k]
        reversed_ = (np.sign(adj) != np.sign(pooled)) and abs(adj) > 0.02
        shrunk = abs(adj) < 0.4 * abs(pooled) if np.sign(adj) == np.sign(pooled) else False
        return _res("simpson", "high", not (reversed_ or shrunk), pooled_effect=pooled, stratum_adjusted_effect=adj,
                    per_stratum=per, reversed=bool(reversed_), claim_direction=int(d),
                    note="pooled association flips sign once the confounder is held fixed" if reversed_ else "")

    def concentration(search, k, cfg):
        m = M_all[idx_s[k]] > 0
        y = Ys[m, idx_o[k]]; ww = w[m]
        if ww.sum() < 10 or (1 - ww).sum() < 10:
            return _res("concentration", "high", True, note="n/a")
        lo, hi = np.percentile(y, [1, 99])
        yc = np.clip(y, lo, hi)
        e0 = y[ww == 1].mean() - y[ww == 0].mean()
        e1 = yc[ww == 1].mean() - yc[ww == 0].mean()
        keep = e1 / e0 if e0 != 0 else 1.0
        return _res("concentration", "high", keep >= 0.5 or np.sign(e0) != direction[k], effect_before=float(e0), effect_winsorised=float(e1), retained=float(keep))

    def stability(search, k, cfg):
        m = np.where(M_all[idx_s[k]] > 0)[0]
        halves = np.array_split(m, 2)
        ts = []
        for h in halves:
            mm = np.zeros((1, len(w))); mm[0, h] = 1
            ts.append(float(_welch(mm, w, Ys[:, idx_o[k]])[0][0]) * direction[k])
        ok = ts[0] > 0 and ts[1] > 0 or np.sign(observed[k]) < 0
        return _res("regime", "high", ok, first_half_t=ts[0], second_half_t=ts[1], note="same direction in both halves of the data")

    def peeking(search, k, cfg):
        look = search.hyps[k]["spec"].get("peeked_at_look")
        if look is None or not cuts:
            return _res("peeking", "high", True, note="no interim-look claim")
        t_stop = abs(t_prefix(w, k, cuts[look]))
        t_final = abs(t_prefix(w, k, cuts[-1]))
        rng = np.random.default_rng(777 + k)
        null_max = np.empty(250)
        for b in range(250):                              # best |t| at ANY of the looks, under the null
            wp = _perm_within(rng, w, strata)
            null_max[b] = max(abs(t_prefix(wp, k, c)) for c in cuts)
        p_peek = float((1 + (null_max >= t_final).sum()) / 251)
        return _res("peeking", "high", p_peek <= 0.05, looks_taken=int(look + 1), t_at_first_significant_look=float(t_stop),
                    t_at_final_look=float(t_final), p_final_accounting_for_peeking=p_peek,
                    note="an interim 'win' only counts if the final data clears the peeking-adjusted bar")

    def oos_fn(k):
        m = H_masks.get(search.hyps[k]["spec"]["segment"])
        if m is None or m.sum() < 10:
            return {"n_days": 0, "oos_sharpe": 0.0, "t": 0.0, "p_one_sided": 1.0, "confirmed": False}
        mm = m.astype(float)[None, :]
        t = float(_welch(mm, wh, Yh[:, idx_o[k]])[0][0]) * direction[k]
        from scipy import stats as st
        p = float(st.norm.sf(t))
        return {"n_days": int(m.sum()), "oos_sharpe": t, "t": t, "p_one_sided": p, "confirmed": bool(p < 0.05 and t > 0)}

    search = Search(hyps=hyps, pnl=None, pnl_hold=None, factors=None, factors_hold=None, factor_names=[], turnover=None,
                    domain="tabular", observed=observed, n_obs=n_search, null_fn=null_fn,
                    attack_fns=[simpson, concentration, stability, peeking], oos_fn=oos_fn, unit="participants",
                    meta={"claim": spec.claim, "treat": spec.treat, "outcomes": spec.outcomes,
                          "strata": spec.strata_cols or [], "n": n_all, "n_search": n_search, "n_holdout": n_all - n_search,
                          "headline": {h["id"]: float(t_raw[i]) for i, h in enumerate(hyps) if h["family"] == "overall"}})
    search.truth = None
    return search
