"""Scenario runner: drives the whole flow and emits a live event stream.

Used by the FastAPI server (WebSocket), the offline demo CLI and the demo recorder. Everything works with no API key
(MockLLM) and no network (pre-downloaded data).

Event types: meta, agent, proposal, data_quality, backtest_progress, naive, null_audit, prosecution, verdict, graveyard,
credibility, ledger, tokens, prereg, strategy_card, portfolio, truth_check, replication, done, error
"""
from __future__ import annotations

import io
import os
import time
from typing import Callable

import numpy as np
import pandas as pd

from . import cleaner as CL
from .adapters import betting as BT, claims_data as CD, equities as EQ, tabular as TB
from .adapters.market_panel import simulate_market_panel
from .adapters.synthetic import SynthConfig, generate
from .court import stats as S
from .court.pipeline import Court, CourtConfig, score_against_truth
from . import literature as LIT
from .hunters import base as H
from .ledger import Ledger
from .llm import LLM
from .prosecutors import review
from .search import Search, build_market_search, from_synthetic

HONESTY = [
    "Skeptic reduces false discoveries and quantifies uncertainty. It does not prove truth.",
    "Null-world validity depends on the null generator preserving the data's real structure (assumptions are listed per generator).",
    "Backtests are not forecasts. Nothing here is trading advice.",
    "Built on known methods: White's Reality Check, Hansen's SPA, Deflated Sharpe, Benjamini-Hochberg, alpha-investing (Foster & Stine). "
    "Our contribution is applying them to an autonomous agent's own search as one integrated, measurable system.",
]

TITLES = {
    "market": ("Market swarm (simulated stand-in)", "Six Hunters search for equity patterns; the Court tries to prove each one is luck."),
    "equities": ("Real equities (2010-2025)", "The same swarm on real prices of 117 large caps. The universe is survivors only, which is a flaw."),
    "synthetic": ("Planted-signal benchmark", "500 candidate signals; 5 are real, 10 are confounded decoys, 5 are traps. Ground truth is known."),
    "betting": ("Sports betting edge", "Does betting the model's predicted winner make money, and is it skill?"),
    "ab_test": ("A/B test with peeking", "8 metrics x many segments, analyst peeked 10 times. One effect is real."),
    "health": ("Supplement headline (Simpson's paradox)", "Supplement users look LESS healthy. Is that true once age is held fixed?"),
    "upload": ("Your claim", "Upload a CSV and describe the claim; it goes through the same Court."),
    "replication": ("Replication mode", "Paste a paper's claim; it is rebuilt on our data and judged, including decay after publication."),
}


class Ctx:
    def __init__(self, emit: Callable[[dict], None] | None, seed: int = 0):
        self.events: list[dict] = []
        self._emit = emit
        self.t0 = time.time()
        self.llm = LLM()
        H.register_mock(self.llm); CL.register_mock(self.llm); review.register_mock(self.llm); _register_extractor(self.llm)
        self.ledger = Ledger()
        self.run_id = self.ledger.start_run("run", {"seed": seed})
        self.seed = seed
        self.online_literature = os.environ.get("SKEPTIC_OFFLINE") != "1"   # live paper search by default
        self.llm.meter.listeners.append(lambda snap: self.emit("tokens", **snap))

    def emit(self, type_: str, **kw):
        ev = {"type": type_, "ts": round(time.time() - self.t0, 3), **kw}
        self.events.append(ev)
        if self._emit:
            self._emit(ev)

    def court_emit(self, ev: dict):
        ev = dict(ev); t = ev.pop("type"); ev.pop("ts", None)
        self.emit(t, **ev)


def _meta(ctx: Ctx, name: str, **extra):
    title, blurb = TITLES[name]
    ctx.emit("meta", scenario=name, title=title, blurb=blurb, honesty=HONESTY, **extra)


def _naive(ctx: Ctx, search: Search, descr: Callable[[int], str]):
    t = search.obs()
    from scipy import stats as st
    n = search.T
    p = 2 * st.t.sf(np.abs(t), max(n - 1, 1))
    idx = [int(i) for i in range(search.K) if p[i] < 0.05 and t[i] > 0]          # in the order the analyst found them
    ctx.emit("naive", count=len(idx), total=search.K,
             claims=[{"id": search.hyps[i]["id"], "text": descr(i), "t": float(t[i]), "p": float(p[i])} for i in idx])


def _search_ledger_event(ctx: Ctx):
    ctx.emit("ledger", **ctx.ledger.effective_search_size(ctx.run_id), by_agent=ctx.ledger.by_agent(ctx.run_id))


def _hunt(ctx: Ctx, families: list[str], rounds: int, per: int, fields: set | None) -> list[dict]:
    seen: set[str] = set()
    hyps: list[dict] = []
    for rnd in range(rounds):
        ctx.emit("agent", agent="Hunters", text=f"Round {rnd+1}/{rounds}: Hunters are proposing hypotheses", round=rnd)
        for fam in families:
            hu = H.Hunter(fam, ctx.llm, seed=ctx.seed, fields=fields)
            hs, bad = hu.propose(per, seen, rnd)
            for h in hs:
                h["round"] = rnd
                hyps.append(h)
                ctx.ledger.record(ctx.run_id, h["agent"], fam, h["id"], spec=h["spec"], params=h.get("params"), seed=ctx.seed,
                                  tokens=0, status="proposed")
                ctx.emit("proposal", agent=h["agent"], family=fam, id=h["id"], description=h["description"], formula=h["spec"]["formula"],
                         prior=h["prior"], source_paper=h.get("source_paper"), universe=h["spec"]["universe"], round=rnd)
            if bad:
                ctx.emit("agent", agent=hu.name, text=f"{bad} proposal(s) rejected by the safe-DSL validator", round=rnd)
            _search_ledger_event(ctx)
    return hyps


# ----------------------------------------------------------------------------------------------------- scenarios
def run_market(ctx: Ctx, params: dict, real: bool = False) -> dict:
    name = "equities" if real else "market"
    holdout = 504 if real else 252
    panel = EQ.load_panel() if real else simulate_market_panel(ctx.seed, decay_after=None)
    _meta(ctx, name, panel=panel.meta, shape=list(panel.shape), standin=bool(panel.meta.get("is_standin")))
    ctx.emit("agent", agent="Cleaner", text="Profiling data and choosing cleaning options (every choice is a forking path)")
    clean_panel, rep = CL.clean(panel, ctx.llm, ctx.ledger, ctx.run_id, search_end=panel.shape[0] - holdout, seed=ctx.seed)
    ctx.emit("data_quality", **{k: v for k, v in rep.items() if k != "profile"}, profile=rep["profile"])
    _search_ledger_event(ctx)
    fams = H.FAMILIES if not real else ["momentum", "mean_reversion", "calendar", "volatility", "volume_sentiment"]
    fields = None if not real else {"close", "ret", "volume"}
    rounds, per = int(params.get("rounds", 4)), int(params.get("per_hunter", 12))
    hyps = _hunt(ctx, fams, rounds, per, fields)
    ctx.emit("agent", agent="Backtester", text=f"Backtesting {len(hyps)} hypotheses (zero tokens)")
    search = build_market_search(clean_panel, hyps, holdout=holdout,
                                 progress=lambda i, n: ctx.emit("backtest_progress", done=i, total=n) if i % 20 == 0 or i == n else None)
    ctx.llm.meter.stat("backtests", len(hyps))
    search.meta.update(panel.meta)
    _naive(ctx, search, lambda i: search.hyps[i]["description"])
    return _court(ctx, search, params, truth=not real)


def run_synthetic(ctx: Ctx, params: dict) -> dict:
    d = generate(SynthConfig(seed=ctx.seed, strength_mult=float(params.get("strength", 1.0))))
    search = from_synthetic(d, ctx.seed)
    _meta(ctx, "synthetic", standin=True, setup={"signals": search.K, "planted": 5, "decoys": 10, "traps": 5}, panel={"note": "Simulated; ground truth known."})
    ctx.ledger.record_many(ctx.run_id, [{"agent": h["agent"], "family": h["family"], "hypothesis_id": h["id"], "spec": h["spec"], "seed": ctx.seed,
                                         "status": "proposed"} for h in search.hyps])
    ctx.emit("agent", agent="Hunters", text=f"{search.K} candidate signals are on the table")
    _search_ledger_event(ctx)
    _naive(ctx, search, lambda i: search.hyps[i]["description"])
    return _court(ctx, search, params, truth=True, log_tests=False)


def _librarian(ctx: Ctx, search: Search):
    """Librarian: find papers per hypothesis family, check they APPLY to this scenario, hand the evidence to the Prosecutors."""
    dates = search.dates
    sc = LIT.Scenario.from_meta(search.meta, dates, n_assets=int(search.meta.get("n_assets", 117 if search.meta.get("survivorship_bias") else 150)))
    lib = LIT.Librarian(sc, online=bool(ctx.online_literature))
    ctx.emit("agent", agent="Librarian", text=f"Searching the literature ({'LIVE: Crossref + arXiv' if lib.online else 'offline, curated index only'}) and checking it against this scenario ({sc.start_year}-{sc.end_year}, {sc.universe})")
    evid, per_family = {}, {}
    for h in search.hyps:
        evid[h["id"]] = lib.evidence(h)
        per_family.setdefault(h.get("family", "?"), evid[h["id"]])
    for fam, ev in per_family.items():
        top = [p for p in ev["papers"] if p.get("relevance", 0) > 0][:2]
        if top:
            ctx.emit("literature", family=fam, scenario={"start_year": sc.start_year, "end_year": sc.end_year, "universe": sc.universe, "survivor_universe": sc.survivor_universe},
                     papers=top, has_direct_support=ev["has_direct_support"], expected_haircut=ev["expected_haircut"], caveats=ev["caveats"],
                     live=lib.online, search_status=list(lib.status))
    search.literature = evid
    search.extra_attack_fns = (search.extra_attack_fns or []) + [LIT.literature_attack_factory(evid)]


def _court(ctx: Ctx, search: Search, params: dict, truth: bool, log_tests: bool = False) -> dict:
    cfg = CourtConfig(seed=ctx.seed, n_worlds=int(params.get("n_worlds", 300)), n_rounds=int(params.get("rounds", 4)),
                      use_alpha_investing=bool(params.get("alpha_investing", True)))
    for h in search.hyps:
        h.setdefault("round", -1)
    review_fn = review.make_llm_review(ctx.llm) if search.domain in ("market",) else None
    if search.domain in ("market", "synthetic") and search.pnl is not None and search.literature is None:
        _librarian(ctx, search)
    court = Court(ctx.ledger, ctx.llm.meter, review_fn, ctx.court_emit, cfg)
    res = court.run(search, ctx.run_id, log_tests=log_tests)
    if truth and search.truth is not None:
        sc = score_against_truth(search, res)
        ctx.emit("truth_check", score=sc, matches=[{"id": h["id"], "truth": str(search.truth[i]), "description": h.get("description")}
                                                   for i, h in enumerate(search.hyps) if h["id"] in set(res["discoveries"])])
    ctx.emit("tokens", **ctx.llm.meter.snapshot())
    ctx.emit("done_summary", discoveries=len(res["discoveries"]), tested=search.K, ledger=res["ledger"])
    return {"search": search, "result": res}


def run_betting(ctx: Ctx, params: dict) -> dict:
    df, info = BT.load(ledger=ctx.ledger, run_id=ctx.run_id)
    search = BT.build_search(df, info)
    _meta(ctx, "betting", standin=True, panel=info, claim=search.meta["claim"])
    ctx.emit("agent", agent="Cleaner", text=f"{info['dropped_missing_odds']} bets dropped for missing odds; draws count as lost bets",
             round=0)
    ctx.ledger.record_many(ctx.run_id, [{"agent": h["agent"], "family": h["family"], "hypothesis_id": h["id"], "spec": h["spec"], "seed": ctx.seed,
                                         "status": "proposed"} for h in search.hyps])
    ctx.emit("agent", agent="Hunters", text=f"{search.K} betting rules (sport, side, odds bucket, weekday, month...) are on the table")
    _search_ledger_event(ctx)
    _naive(ctx, search, lambda i: f"{search.hyps[i]['description']}  (ROI {100*search.pnl[i].sum()/max(1,search.hyps[i]['spec']['n_bets']):+.0f}%)")
    return _court(ctx, search, {**params, "rounds": params.get("rounds", 3)}, truth=False)


def _tabular(ctx: Ctx, name: str, spec: TB.TabularSpec, params: dict, standin=True) -> dict:
    search = TB.build_tabular_search(spec, ctx.seed)
    _meta(ctx, name, standin=standin, claim=spec.claim, panel={"n": search.meta["n"], "n_search": search.meta["n_search"], "n_holdout": search.meta["n_holdout"],
                                                              "treat": spec.treat, "outcomes": spec.outcomes, "strata": spec.strata_cols or [],
                                                              "note": "Simulated data with known truth." if standin else "User-supplied data."})
    ctx.ledger.record_many(ctx.run_id, [{"agent": h["agent"], "family": h["family"], "hypothesis_id": h["id"], "spec": h["spec"], "seed": ctx.seed,
                                         "status": "proposed"} for h in search.hyps])
    ctx.emit("agent", agent="Hunters", text=f"{search.K} segment/metric effects were examined")
    _search_ledger_event(ctx)
    t = search.obs()
    from scipy import stats as st
    raw = np.array([h["spec"]["headline_t"] for h in search.hyps])
    p = 2 * st.norm.sf(np.abs(raw))
    idx = [int(i) for i in np.argsort(p) if p[i] < 0.05]
    ctx.emit("naive", count=len(idx), total=search.K,
             claims=[{"id": search.hyps[i]["id"], "text": search.hyps[i]["description"], "t": float(raw[i]), "p": float(p[i])} for i in idx])
    return _court(ctx, search, {**params, "rounds": params.get("rounds", 2)}, truth=False)


def run_health(ctx, params): return _tabular(ctx, "health", CD.health_supplement(ctx.seed), params)
def run_ab(ctx, params): return _tabular(ctx, "ab_test", CD.ab_test(ctx.seed), params)


def run_upload(ctx: Ctx, params: dict) -> dict:
    df = pd.read_csv(io.StringIO(params["csv"])) if "csv" in params else pd.read_csv(params["path"])
    treat, outcome = params["treat"], params["outcome"]
    sub = params.get("subgroups") or [c for c in df.columns if c not in (treat, outcome) and (df[c].dtype == object or df[c].nunique() <= 8)][:6]
    strata = params.get("strata") or None
    d = df.copy()
    if d[treat].nunique() > 2:
        d[treat] = (d[treat] > d[treat].median()).astype(int)
    elif d[treat].dtype == object:
        d[treat] = pd.factorize(d[treat])[0]
    d = d.dropna(subset=[treat, outcome])
    spec = TB.TabularSpec(df=d, treat=treat, outcomes=[outcome], subgroup_cols=sub, strata_cols=strata, time_col=params.get("time"),
                          claim=params.get("claim", f"{treat} affects {outcome}"), pairs=bool(params.get("pairs", True)))
    return _tabular(ctx, "upload", spec, params, standin=False)


# ----------------------------------------------------------------------------------------------------- replication
EXTRACT_SCHEMA = {"type": "object", "properties": {
    "name": {"type": "string"}, "feature_formula": {"type": "string"}, "direction": {"enum": ["long_high", "long_low"]},
    "claimed_sharpe": {"type": ["number", "null"]}, "source": {"type": ["string", "null"]}, "publication_year": {"type": ["integer", "null"]},
    "rationale": {"type": "string"}}, "required": ["name", "feature_formula", "rationale"]}

_CATALOG = [
    (("12-1", "momentum", "past winners", "winners"), "ts_return(lag(close, 21), 231)", "12-1 month momentum", "Jegadeesh & Titman (1993)", 1993),
    (("reversal", "past week", "short-term", "short term", "losers"), "-ts_return(close, 5)", "short-term reversal", "Jegadeesh (1990)", 1990),
    (("low volatility", "low-volatility", "low risk", "low beta", "idiosyncratic"), "-rolling_std(ret, 60)", "low-volatility anomaly", "Ang, Hodrick, Xing & Zhang (2006)", 2006),
    (("value", "book-to-market", "book to market"), "btm", "value (book-to-market)", "Fama & French (1992)", 1992),
    (("volume",), "-(volume / rolling_mean(volume, 20))", "volume shock fade", "Gervais, Kaniel & Mingelgrin (2001)", 2001),
]


def _register_extractor(llm: LLM):
    def mock(hint: dict) -> dict:
        txt = hint["text"].lower()
        for keys, formula, name, src, yr in _CATALOG:
            if any(k in txt for k in keys):
                return {"name": name, "feature_formula": formula, "direction": "long_high", "claimed_sharpe": None, "source": src,
                        "publication_year": yr, "rationale": f"Matched the described effect to the canonical '{name}' construction."}
        return {"name": "unrecognised claim", "feature_formula": "ts_return(close, 20)", "direction": "long_high", "claimed_sharpe": None,
                "source": None, "publication_year": None, "rationale": "Could not map the text to a known construction; defaulted to 20-day momentum."}
    llm.register_mock("extractor", mock)


def run_replication(ctx: Ctx, params: dict) -> dict:
    text = params.get("text", "Short-term reversal: stocks that fell over the past week tend to rebound the next week.")
    dataset = params.get("dataset", "market")
    real = dataset == "equities" and EQ.available()
    _meta(ctx, "replication", standin=not real, claim=text)
    ctx.emit("agent", agent="Extractor", text="Reading the paper's claim and turning it into a structured hypothesis")
    obj, _ = ctx.llm.complete_json("extractor", "You extract testable hypotheses from finance papers. JSON only.",
                                   f"Extract the effect, a feature formula in the DSL, and metadata.\nDSL: {H.DSL_DOC}\nClaim: {text}", EXTRACT_SCHEMA,
                                   mock_hint={"text": text})
    from .features_dsl import DSLError, parse, scale_windows
    try:
        parse(obj["feature_formula"])
    except DSLError as e:
        ctx.emit("error", message=f"Extractor produced an invalid formula: {e}")
        return {}
    sign = -1 if obj.get("direction") == "long_low" else 1
    base = obj["feature_formula"] if sign == 1 else f"-({obj['feature_formula']})"
    ctx.emit("replication", stage="extracted", extracted=obj, formula=base)
    panel = EQ.load_panel() if real else simulate_market_panel(ctx.seed, decay_after=900)
    holdout = 504 if real else 252
    lib = LIT.Librarian(LIT.Scenario.from_meta(panel.meta, panel.dates, n_assets=panel.shape[1]), online=bool(ctx.online_literature))
    ctx.emit("agent", agent="Librarian", text=("Searching Crossref and arXiv LIVE for the source paper" if lib.online else "Offline: curated index only") + " and checking it against this scenario")
    found = lib.find(obj["name"] + " anomaly stock returns", None, top=4)
    found = [p for p in found if p.get("relevance", 0) > 0][:6]
    ctx.emit("literature", family="replication", papers=found, has_direct_support=any(p["relevance"] >= 4 for p in found),
             live=lib.online, search_status=list(lib.status),
             scenario={"start_year": int(panel.dates[0].year), "end_year": int(panel.dates[-1].year), "universe": panel.meta.get("source"),
                       "survivor_universe": bool(panel.meta.get("survivorship_bias"))},
             expected_haircut=max([p.get("expected_haircut", 0) for p in found], default=0.0), caveats=sorted({c for p in found[:3] for c in p.get("caveats", [])})[:5])
    pub = pd.Timestamp(params["pub_date"]) if params.get("pub_date") else (panel.dates[900] if not real else None)
    if not real:
        ctx.emit("agent", agent="Replicator", text=f"Stand-in world: the effect is arbitrage-limited and decays after {pub.date()}")
    clean_panel, rep = CL.clean(panel, ctx.llm, ctx.ledger, ctx.run_id, search_end=panel.shape[0] - holdout, seed=ctx.seed)
    ctx.emit("data_quality", **{k: v for k, v in rep.items() if k != "profile"}, profile=rep["profile"])
    # researcher degrees of freedom: the claim plus the variants anyone replicating would try; ALL are counted
    variants = [("as published", base)]
    for fac in (0.5, 0.75, 1.5, 2.0):
        try:
            variants.append((f"windows x{fac}", scale_windows(base, fac)))
        except Exception:
            pass
    variants.append(("opposite sign", f"-({base})"))
    variants.append(("held 5 days", base))
    hyps = []
    for i, (lab, f) in enumerate(variants):
        h = {"id": f"rep_{i}", "family": "replication", "agent": "replication-hunter", "description": f"{obj['name']} ({lab})",
             "prior": 0.8 if i == 0 else 0.3, "round": 0, "source_paper": obj.get("source"),
             "spec": {"formula": f, "universe": "current_constituents" if real else "point_in_time", "hold": 5 if lab == "held 5 days" else 1,
                      "uses_future_bar": False, "windows": []}}
        if real:
            h["spec"]["universe"] = "point_in_time"
        hyps.append(h)
        ctx.ledger.record(ctx.run_id, h["agent"], "replication", h["id"], spec=h["spec"], seed=ctx.seed, status="proposed")
        ctx.emit("proposal", agent=h["agent"], family="replication", id=h["id"], description=h["description"], formula=f, prior=h["prior"], round=0,
                 source_paper=obj.get("source"), universe=h["spec"]["universe"])
    _search_ledger_event(ctx)
    search = build_market_search(clean_panel, hyps, holdout=holdout)
    search.meta.update(panel.meta)
    # hand the Librarian's papers to the Prosecutors (literature attack): same evidence for every replication variant
    direct = [p for p in found if p["relevance"] >= 4]
    rep_ev = {"papers": found, "has_direct_support": bool(direct), "caveats": sorted({c for p in found[:3] for c in p.get("caveats", [])})[:5],
              "expected_haircut": max([p.get("expected_haircut", 0) for p in direct], default=0.0), "prior_suggestion": 0.8 if direct else 0.3}
    search.literature = {h["id"]: rep_ev for h in hyps}
    search.extra_attack_fns = (search.extra_attack_fns or []) + [LIT.literature_attack_factory(search.literature)]
    out = _court(ctx, search, {**params, "rounds": 1, "n_worlds": params.get("n_worlds", 300)}, truth=False)
    # decay analysis on the as-published construction
    pnl = np.concatenate([search.pnl[0], search.pnl_hold[0]])
    dates = panel.dates[1:1 + len(pnl)]
    dec = {"stage": "decay", "formula": base}
    if pub is not None and dates[0] < pub < dates[-1]:
        pre, post = pnl[dates < pub], pnl[dates >= pub]
        sh = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(252)) if len(x) > 20 and x.std() > 0 else 0.0
        diff_t = float((pre.mean() - post.mean()) / np.sqrt(pre.var(ddof=1) / len(pre) + post.var(ddof=1) / len(post)))
        dec.update(pub_date=str(pub.date()), sharpe_pre=sh(pre), sharpe_post=sh(post), decay_t=diff_t, n_pre=len(pre), n_post=len(post),
                   verdict=("Decayed after publication" if sh(post) < 0.5 * sh(pre) and diff_t > 1.64 else "No clear decay" if sh(pre) > 0 else "Weak before publication"))
        roll = pd.Series(pnl).rolling(252).apply(lambda x: x.mean() / (x.std() + 1e-12) * np.sqrt(252), raw=True).dropna()
        dec["rolling_sharpe"] = [float(v) for v in roll.iloc[::10]]
        dec["pub_index"] = int(np.searchsorted(dates, pub) / 10)
    else:
        dec.update(note="The whole sample (2010-2025) is after the paper was published, so this tests whether the effect still works, not whether it decayed.",
                   sharpe_full=float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)))
    primary = out["result"]["verdicts"]["rep_0"]
    dec["primary_verdict"] = primary["verdict"]; dec["confession"] = primary["confession"]; dec["cause_of_death"] = primary["cause_of_death"]
    ctx.emit("replication", **dec)
    return out


SCENARIOS = {"market": lambda c, p: run_market(c, p, False), "equities": lambda c, p: run_market(c, p, True), "synthetic": run_synthetic,
             "betting": run_betting, "ab_test": run_ab, "health": run_health, "upload": run_upload, "replication": run_replication}


def run_scenario(name: str, params: dict | None = None, emit: Callable[[dict], None] | None = None, seed: int = 0) -> tuple[list[dict], dict]:
    params = params or {}
    ctx = Ctx(emit, seed)
    try:
        out = SCENARIOS[name](ctx, params)
    except Exception as e:                                   # surface to the UI; never hang a live run
        import traceback
        ctx.emit("error", message=f"{type(e).__name__}: {e}", trace=traceback.format_exc()[-1500:])
        raise
    ctx.emit("done", scenario=name, seconds=round(time.time() - ctx.t0, 1))
    return ctx.events, out
