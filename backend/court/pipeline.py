"""The Court: orchestrates ledger -> null-world audit -> alpha-investing -> prosecutors -> judge -> pre-registration.

Online: the search is audited round by round. In each round the null worlds re-run the ENTIRE cumulative search, the
alpha-investing wealth sets the level the round must beat, candidates are prosecuted, then judged. Zero-token stats do
the heavy lifting; the only LLM involvement is the (advisory) spec review.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..ledger import Ledger
from ..llm import TokenMeter
from ..prosecutors.attacks import AttackConfig, prosecute
from ..search import Search
from . import cards, judge as J, null_worlds as NW, prereg, stats as S
from .alpha_investing import AlphaInvesting


@dataclass
class CourtConfig:
    n_worlds: int = 300
    method: str = "stationary_bootstrap"
    n_rounds: int = 4
    p_luck: float = 0.5
    seed: int = 0
    attack: AttackConfig = field(default_factory=AttackConfig)
    wealth0: float = 0.09
    payout: float = 0.04
    use_alpha_investing: bool = True
    flat_alpha: float = 0.05            # used when alpha-investing is off
    run_prosecutors: bool = True
    preregister: bool = True


def _hist(null_max: np.ndarray, observed: list[float], bins: int = 30) -> dict:
    hi = max(float(null_max.max()), max(observed + [0.0])) * 1.05
    counts, edges = np.histogram(null_max, bins=bins, range=(0, hi))
    return {"counts": counts.tolist(), "edges": [float(e) for e in edges], "p95": float(np.quantile(null_max, 0.95)),
            "median": float(np.median(null_max))}


class Court:
    def __init__(self, ledger: Ledger, meter: TokenMeter | None = None, llm_review: Callable | None = None,
                 emit: Callable[[dict], None] | None = None, cfg: CourtConfig | None = None):
        self.ledger, self.meter, self.llm_review = ledger, meter or TokenMeter(), llm_review
        self.emit = emit or (lambda e: None)
        self.cfg = cfg or CourtConfig()

    def _e(self, type_: str, **kw):
        self.emit({"type": type_, "ts": time.time(), **kw})

    def run(self, search: Search, run_id: str, log_tests: bool = True) -> dict:
        cfg, K = self.cfg, search.K
        ai = AlphaInvesting(wealth0=cfg.wealth0, payout=cfg.payout)
        rng = np.random.default_rng(cfg.seed)
        rounds = np.array([h.get("round", -1) for h in search.hyps])
        if (rounds < 0).any():
            rounds = np.concatenate([np.full(len(c), i) for i, c in enumerate(np.array_split(np.arange(K), cfg.n_rounds))])
        n_rounds = int(rounds.max()) + 1
        tstat_all = search.obs()
        if log_tests:
            self.ledger.record_many(run_id, [{
                "agent": h.get("agent", "researcher"), "family": h.get("family"), "hypothesis_id": h["id"], "spec": h.get("spec"),
                "params": h.get("params"), "statistic": float(tstat_all[i]), "seed": cfg.seed,
                "tokens": 0} for i, h in enumerate(search.hyps)])
        verdicts: dict[int, J.Verdict] = {}
        round_log = []
        self._e("phase", name="court_start", total_hypotheses=K, rounds=n_rounds)
        for r in range(n_rounds):
            batch = np.where(rounds == r)[0]
            cum = np.where(rounds <= r)[0]
            self._e("agent", agent="Court", text=f"Round {r+1}/{n_rounds}: re-running the whole search ({len(cum)} ideas) in "
                    f"{cfg.n_worlds} null worlds", round=r)
            if search.null_fn is not None:
                null = search.null_fn(cum, cfg.n_worlds, cfg.seed + 101 * r)
            else:
                null = NW.run_null_worlds(search.pnl[cum], search.factors, S.raw_t, cfg.method, cfg.n_worlds, seed=cfg.seed + 101 * r)
            self.meter.stat("null_world_search_runs", cfg.n_worlds * len(cum))
            pos = {int(i): j for j, i in enumerate(cum)}
            conf = J.confidence_scores(null.p_unadjusted)
            prior = float(np.mean([search.hyps[i].get("prior", 0.5) for i in batch])) if len(batch) else 0.5
            level = ai.propose_level(prior, min_level=1.0 / (cfg.n_worlds + 1)) if cfg.use_alpha_investing else cfg.flat_alpha
            es = self.ledger.effective_search_size(run_id)
            es = {**es, "effective": max(es["effective"], len(cum))}
            obs_abs = np.abs(null.observed)
            top = np.argsort(-obs_abs)[:5]
            self._e("null_audit", round=r, search_size=len(cum), level=level, wealth=ai.wealth,
                    histogram=_hist(null.null_max, [float(obs_abs[t]) for t in top]),
                    top_observed=[{"id": search.hyps[int(cum[t])]["id"], "stat": float(obs_abs[t])} for t in top])
            cands = [int(i) for i in batch if null.observed[pos[int(i)]] > 0 and null.p_stepdown[pos[int(i)]] <= level]
            # prosecute every plausible candidate, not just winners: a borderline "CAN'T TELL" that is really a
            # decoy must not carry a high stated confidence
            tab = search.null_fn is not None          # tabular claims: a sign reversal or an interim-look claim must be prosecuted too
            to_pros = [int(i) for i in batch if ((null.observed[pos[int(i)]] > 0 or tab) and null.p_stepdown[pos[int(i)]] < cfg.p_luck)
                       or search.hyps[int(i)].get("spec", {}).get("peeked_at_look") is not None]
            self._e("agent", agent="Court", text=f"{len(cands)} candidate(s) beat the null at level {level:.3f}; {len(to_pros)} borderline-or-better sent to the Skeptics", round=r)
            attacks: dict[int, list] = {}
            if to_pros and cfg.run_prosecutors:
                # confounder attack evidence: audit the factor-ADJUSTED alpha across the same whole search
                if search.pnl is not None and search.domain in ("market", "synthetic") and search.null_fn is None:
                    if search.alpha_adj_p is None:
                        search.alpha_adj_p = np.full(K, np.nan)
                    try:
                        nullA = NW.run_null_worlds(search.pnl[cum], search.factors, S.alpha_t, cfg.method, cfg.n_worlds, seed=cfg.seed + 303 * (r + 1))
                        self.meter.stat("null_world_search_runs", cfg.n_worlds * len(cum))
                        search.alpha_adj_p[cum] = nullA.p_stepdown
                    except Exception as e:  # noqa: BLE001  one bad factor set must not abort a run that already has verdicts
                        self._e("agent", agent="Court", text=f"Factor-adjusted audit skipped ({type(e).__name__}: {e}); confounder attack falls back to raw evidence", round=r)
                with ThreadPoolExecutor(max_workers=4) as ex:
                    futs = {k: ex.submit(prosecute, search, k, cfg.attack, None, self.llm_review) for k in to_pros}
                    for k, f in futs.items():
                        attacks[k] = f.result()
                        self.meter.stat("prosecutor_attacks", len(attacks[k]))
                        self._e("prosecution", hypothesis_id=search.hyps[k]["id"], attacks=attacks[k])
            n_disc = 0
            for i in batch:
                i = int(i)
                j = pos[i]
                v = J.judge_one(search.hyps[i]["id"], null, j, level, attacks.get(i), es, float(conf[j]), search.T, cfg.p_luck, alpha_crit=level)
                if null.observed[j] < 0 and v.verdict == J.DISCOVERY:          # wrong sign is not a discovery of what was claimed
                    v.verdict, v.cause_of_death = J.LUCK, "Died of the wrong sign (sign-flipped variant is a separate idea)"
                verdicts[i] = v
                if v.verdict == J.DISCOVERY:
                    n_disc += 1
                self.ledger.set_status_by_hypothesis(run_id, v.hypothesis_id, {J.DISCOVERY: "discovery", J.LUCK: "luck",
                                                      J.CANT_TELL: "cant_tell"}[v.verdict], {"p_adj": v.p_adj, "cause": v.cause_of_death})
                if v.verdict == J.LUCK and v.cause_of_death and (attacks.get(i) or v.p_adj >= cfg.p_luck) and (null.p_unadjusted[j] < 0.05 or attacks.get(i)):
                    self._e("graveyard", hypothesis_id=v.hypothesis_id, cause=v.cause_of_death, p_adj=v.p_adj,
                            p_naive=float(null.p_unadjusted[j]), formula=search.hyps[i].get("spec", {}).get("formula"),
                            description=search.hyps[i].get("description"), family=search.hyps[i].get("family"),
                            attacks=attacks.get(i, []), statistic=float(null.observed[j]), round=r)
                self._e("verdict", **v.to_dict(), family=search.hyps[i].get("family"), round=r,
                        formula=search.hyps[i].get("spec", {}).get("formula"), description=search.hyps[i].get("description"))
            if cfg.use_alpha_investing:
                if n_disc:        # one payout per validated round: near-duplicate discoveries must not print money
                    ai.update(level, True, label=f"round{r}: {n_disc} discoveries")
                else:
                    ai.update(level, False, label=f"round{r}")
            self._e("credibility", wealth=ai.wealth, wealth0=ai.wealth0, history=ai.history[-1], round=r)
            self._e("ledger", **es, round=r)
            round_log.append({"round": r, "level": level, "candidates": len(cands), "discoveries": n_disc,
                              "wealth_after": ai.wealth, "search_size": len(cum),
                              "null_hist": _hist(null.null_max, [float(obs_abs[t]) for t in top])})
        # ---------------- pre-registration + out-of-sample scoring + strategy cards
        order = sorted(verdicts)
        discoveries = [i for i in order if verdicts[i].verdict == J.DISCOVERY]
        has_pnl = search.pnl is not None and search.null_fn is None
        sr_var = float(np.nanvar(S.sharpe(search.pnl) / np.sqrt(252))) if has_pnl else 0.0
        out_cards, prer = {}, {}
        for i in discoveries:
            if cfg.preregister:
                rec = prereg.register(self.ledger, run_id, search.hyps[i]["id"], search.hyps[i].get("spec", {}), +1,
                                      {"holdout_obs": int(search.pnl_hold.shape[1]) if search.pnl_hold is not None else search.meta.get("n_holdout")})
                res = search.oos_fn(i) if search.oos_fn else prereg.score_out_of_sample(self.ledger, rec, search.pnl_hold[i])
                if search.oos_fn:
                    with self.ledger._lock:
                        self.ledger._db.executescript(prereg._SCHEMA)
                        self.ledger._db.execute("UPDATE prereg SET result=? WHERE hash=?", (__import__("json").dumps(res), rec["hash"]))
                        self.ledger._db.commit()
                prer[i] = {**rec, "oos": res}
                self._e("prereg", **prer[i])
            if has_pnl:
                out_cards[i] = cards.strategy_card(search, i, verdicts[i].to_dict(), K, sr_var, cfg.attack.cost_bps)
                self._e("strategy_card", **out_cards[i])
        port = cards.portfolio_view(search, discoveries, cfg.attack.cost_bps) if has_pnl else None
        if port:
            self._e("portfolio", **port)
        self._emit_illusions(search, verdicts, tstat_all)
        result = {"verdicts": {search.hyps[i]["id"]: verdicts[i].to_dict() for i in order},
                  "discoveries": [search.hyps[i]["id"] for i in discoveries], "rounds": round_log,
                  "alpha_history": ai.history, "ledger": self.ledger.effective_search_size(run_id),
                  "prereg": {search.hyps[i]["id"]: prer[i] for i in prer},
                  "cards": {search.hyps[i]["id"]: out_cards[i] for i in out_cards}, "portfolio": port,
                  "tokens": self.meter.snapshot()}
        self._e("done", summary={"discoveries": len(discoveries), "tested": K})
        return result


def _curve(x: np.ndarray, n: int) -> list[float]:
    """Cumulative series, downsampled to at most n points (for the UI)."""
    c = np.cumsum(np.nan_to_num(x))
    idx = np.unique(np.linspace(0, len(c) - 1, min(n, len(c))).astype(int))
    return [float(c[i]) for i in idx]


def _reality(self, search: Search, i: int, v, cfg_attack) -> tuple[np.ndarray, str]:
    """What the strategy really delivers once the Judge's finding is taken into account (same period as the backtest)."""
    pnl = search.pnl[i]
    killers = sorted([a for a in v.attacks if a.get("severity") == "high" and not a.get("survived", True)],
                     key=lambda a: J.KILL_PRIORITY.index(a["attack"]) if a["attack"] in J.KILL_PRIORITY else 99)
    cause = killers[0]["attack"] if killers else None
    ev = killers[0].get("evidence", {}) if killers else {}
    try:
        if cause == "costs" and search.turnover is not None:
            return pnl - float(search.turnover[i]) * cfg_attack.cost_bps / 1e4, "the same trades after trading costs"
        if cause == "leakage" and search.rerun is not None:
            return np.asarray(search.rerun(i, extra_lag=1), float), "re-run using only information available at the time"
        if cause == "survivorship" and search.rerun is not None:
            return np.asarray(search.rerun(i, universe="point_in_time"), float), "re-run including companies that later disappeared"
        if cause == "placebo" and "roi_random_side_pct" in ev:
            n_bets = max(1, int(search.hyps[i].get("spec", {}).get("n_bets", 1)))
            return np.full(len(pnl), ev["roi_random_side_pct"] / 100 * n_bets / len(pnl)), "the same number of bets placed on a random side"
    except Exception:  # noqa: BLE001  a failed re-run falls back to the held-out data
        pass
    rate = float(search.pnl_hold[i].mean())
    return np.full(len(pnl), rate), "its average result on fresh data it was never tuned on"


def _emit_illusions(self, search: Search, verdicts: dict, tstat_all: np.ndarray, top: int = 8):
    """'Cost of illusion': strategies the Judge threw out as LUCK that nevertheless looked profitable in the backtest.
    Compares what the backtest promised with what the strategy really delivers over the same period (see _reality)."""
    if search.pnl is None or search.pnl_hold is None or search.null_fn is not None:
        return
    lucky = [i for i, v in verdicts.items() if v.verdict == J.LUCK and search.pnl[i].sum() > 0 and tstat_all[i] >= 2.0]
    lucky.sort(key=lambda i: -float(search.pnl[i].sum()))
    kind = "bets" if search.domain == "betting" else "returns"
    for i in lucky[:top]:
        v, h = verdicts[i], search.hyps[i]
        real, basis = _reality(self, search, i, v, self.cfg.attack)
        self._e("illusion", hypothesis_id=h["id"], description=h.get("description"), formula=h.get("spec", {}).get("formula"),
                cause=v.cause_of_death, p_adj=v.p_adj, kind=kind, n_lucky_total=len(lucky), basis=basis,
                in_curve=_curve(search.pnl[i], 120), real_curve=_curve(real, 120),
                in_total=float(search.pnl[i].sum()), real_total=float(np.nansum(real)), hold_total=float(search.pnl_hold[i].sum()),
                n_in=int(search.pnl.shape[1]), n_hold=int(search.pnl_hold.shape[1]), t_in=float(tstat_all[i]))


Court._emit_illusions = _emit_illusions


def score_against_truth(search: Search, result: dict) -> dict:
    if search.truth is None:
        return {}
    ids = [h["id"] for h in search.hyps]
    disc = np.array([i in set(result["discoveries"]) for i in ids])
    truth = search.truth == "alpha"
    n, tp = int(disc.sum()), int((disc & truth).sum())
    return {"claimed": n, "tp": tp, "fp": n - tp, "precision": tp / n if n else float("nan"), "recall": tp / max(1, truth.sum()),
            "fdr": (n - tp) / n if n else 0.0, "fp_beta": int((disc & (search.truth == "beta")).sum()),
            "fp_trap": int((disc & (search.truth == "trap")).sum()), "fp_null": int((disc & (search.truth == "null")).sum())}
