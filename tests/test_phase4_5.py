import numpy as np
import pytest
from skeptic.backend import features_dsl as D, cleaner
from skeptic.backend.llm import LLM, TokenMeter
from skeptic.backend.ledger import Ledger
from skeptic.backend.hunters import base as H
from skeptic.backend.prosecutors import attacks as A, review
from skeptic.backend.adapters.market_panel import simulate_market_panel, backtest
from skeptic.backend.search import build_market_search, from_synthetic
from skeptic.backend.adapters.synthetic import SynthConfig, generate
from skeptic.backend.court.pipeline import Court, CourtConfig, score_against_truth


@pytest.mark.parametrize("bad", ["__import__('os').system('x')", "close.__class__", "lambda: 1", "open('f')",
                                  "ts_return(close, n)", "ts_return(close, 0)", "ts_return(close, 9999)", "'a'+close",
                                  "close ** 2", "eval('1')", "unknown_field", "rank(close, 3)", "[close]", "ts_return(close, 5.5)"])
def test_dsl_rejects_unsafe(bad):
    with pytest.raises(D.DSLError):
        D.parse(bad)


def test_dsl_evaluates_known_values_and_flags_lead():
    P = simulate_market_panel(1, n_days=200, n_assets=30)
    dsl = P.as_dsl()
    r = D.evaluate("ts_return(close, 5)", dsl)
    c = P.fields["close"]
    assert np.allclose(r[10:20], c[10:20] / c[5:15] - 1, equal_nan=True)
    assert D.parse("lead(ret, 1)").uses_lead and not D.parse("-ts_return(close, 5)").uses_lead
    # causal: changing future data must not change today's feature
    c2 = dict(dsl); c2["close"] = c.copy(); c2["close"][100:] = 1.0
    assert np.allclose(D.evaluate("rolling_mean(close, 10)", dsl)[:99], D.evaluate("rolling_mean(close, 10)", c2)[:99], equal_nan=True)
    assert D.scale_windows("ts_return(close, 10) - rolling_std(ret, 20)", 2.0) == "ts_return(close, 20) - rolling_std(ret, 40)"


def test_mock_hunters_valid_deterministic_and_deduped():
    llm = LLM(); H.register_mock(llm)
    a, b = set(), set()
    h1, _ = H.Hunter("momentum", llm, seed=3).propose(15, a, 0)
    h2, _ = H.Hunter("momentum", llm, seed=3).propose(15, b, 0)
    assert [x["id"] for x in h1] == [x["id"] for x in h2] and len({x["id"] for x in h1}) == len(h1)
    assert any(x["source_paper"] for x in h1) and all(x["spec"]["formula"] for x in h1)


@pytest.fixture(scope="module")
def market():
    llm = LLM(); H.register_mock(llm); cleaner.register_mock(llm); review.register_mock(llm)
    L = Ledger(); run = L.start_run("t")
    P = simulate_market_panel(0)
    Pc, rep = cleaner.clean(P, llm, L, run, search_end=P.shape[0] - 252)
    return llm, L, run, P, Pc, rep


def test_cleaner_logs_every_choice(market):
    llm, L, run, P, Pc, rep = market
    e = L.effective_search_size(run)
    assert e["cleaning_choices"] == 3 and e["cleaning_paths"] == 8
    assert np.isfinite(Pc.fields["ret"][P.alive]).all()      # gaps filled


def test_prosecutors_catch_lookahead_survivorship_beta_costs(market):
    llm, L, run, P, Pc, rep = market
    mk = lambda f, uni="point_in_time", i=0: {"id": f"h{i}", "family": "x", "prior": .5, "description": "", "spec": {
        "formula": f, "universe": uni, "hold": 1, "uses_future_bar": D.parse(f).uses_lead, "windows": D.parse(f).windows}}
    hyps = [mk("lead(ret, 1) + 0.0 * close", i=0), mk("-ts_return(close, 10)", "current_constituents", 1),
            mk("-ts_return(close, 10)", i=2), mk("btm", i=3), mk("-ts_return(close, 1)", i=4)]
    S = build_market_search(Pc, hyps)
    res = {i: {a["attack"]: a for a in A.prosecute(S, i)} for i in range(5)}
    assert not res[0]["leakage"]["survived"]
    assert not res[1]["survivorship"]["survived"] and res[2]["survivorship"]["survived"]
    assert not res[3]["confounder"]["survived"]            # value is a known factor
    assert not res[4]["costs"]["survived"]                 # 1-day reversal: turnover eats the edge


def test_court_end_to_end_synthetic_and_market(market):
    d = generate(SynthConfig(seed=4, n_days=900))
    S = from_synthetic(d, 4)
    L = Ledger(); run = L.start_run("s")
    ev = []
    res = Court(L, emit=ev.append, cfg=CourtConfig(n_worlds=150, seed=4)).run(S, run)
    sc = score_against_truth(S, res)
    assert sc["fp_trap"] == 0 and sc["fp_beta"] == 0 and sc["fp_null"] == 0
    types = {e["type"] for e in ev}
    assert {"null_audit", "verdict", "credibility", "ledger", "done"} <= types
    assert "I tried" in next(iter(res["verdicts"].values()))["confession"]
    for hid in res["discoveries"]:                          # pre-registered, hashed, scored out of sample
        assert len(res["prereg"][hid]["hash"]) == 64 and "oos" in res["prereg"][hid]


def test_no_eval_or_exec_in_codebase():
    import pathlib, re
    root = pathlib.Path(__file__).resolve().parents[1] / "backend"
    for f in root.rglob("*.py"):
        src = f.read_text(encoding="utf-8")
        assert not re.search(r"(?<![\w.])(eval|exec)\(", src.replace("evaluate(", "")), f


def test_librarian_finds_applicable_papers_and_feeds_prosecutor(monkeypatch):
    from skeptic.backend import literature as LIT
    paper = {"title": "Short-term reversal in stock returns", "authors": "A. Author", "year": 2005, "venue": "J Fin",
             "abstract": "Ten day return reversal mean reversion in stock returns.", "url": "https://doi.org/10.1/r", "source": "OpenAlex (live)", "cited_by": 10}
    monkeypatch.setattr(LIT, "live_search", lambda q, n=6: ([paper], ["OpenAlex: 1 results"]))
    sc = LIT.Scenario(start_year=2010, end_year=2025, n_assets=117, survivor_universe=True)
    lib = LIT.Librarian(sc, online=True)
    ev = lib.evidence({"description": "10-day reversal", "family": "mean_reversion", "spec": {"formula": "-ts_return(close, 10)"}})
    assert ev["has_direct_support"] and ev["expected_haircut"] >= 0.3
    assert any("after this paper appeared" in c for c in ev["caveats"])
    monkeypatch.setattr(LIT, "live_search", lambda q, n=6: ([], ["offline"]))
    lib = LIT.Librarian(sc, online=True)
    none = lib.evidence({"description": "weekday tilt", "family": "calendar", "spec": {"formula": "dow() * rolling_mean(ret, 5)"}})
    assert not none["has_direct_support"]
    # attack: an implausibly good, unsupported idea is flagged (advisory, medium severity)
    class S:  # minimal stand-in search
        hyps = [{"id": "x"}]
        pnl = np.random.default_rng(0).normal(0.004, 0.01, (1, 800))
    r = LIT.literature_attack_factory({"x": none})(S, 0, None)
    assert r["attack"] == "literature" and r["severity"] == "medium" and not r["survived"]


def test_methods_scout_validates_and_never_fabricates(monkeypatch):
    from skeptic.backend import literature as L
    good = {"title": "Data snooping and the reality check", "abstract": "x " * 60 + "backtest multiple testing", "type": "article", "venue": "Econometrica", "cited_by": 900, "authors": "White", "year": 2000, "url": "u", "doi": "d", "source": "OpenAlex (live)"}
    bad = [{**good, "type": "preprint", "title": "a"}, {**good, "cited_by": 2, "title": "b"}, {**good, "abstract": "", "title": "c"}, {**good, "title": "d", "abstract": "y " * 60}]
    monkeypatch.setattr(L, "openalex", lambda q, rows=5, timeout=10: [good] + bad)
    r = L.methods_scout(True)
    assert all(len(v) == 1 and v[0]["title"] == good["title"] for v in r["papers"].values() if v)
    assert L.methods_scout(False)["papers"] == {}
