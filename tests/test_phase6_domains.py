import numpy as np
from skeptic.backend.adapters import claims_data as CD, tabular as TB
from skeptic.backend.court.pipeline import Court, CourtConfig
from skeptic.backend.ledger import Ledger
from skeptic.backend.runner import run_scenario


def _court(search, rounds=2, worlds=150):
    L = Ledger(); run = L.start_run("t")
    return Court(L, cfg=CourtConfig(n_worlds=worlds, n_rounds=rounds)).run(search, run)


def test_health_headline_dies_to_simpsons_paradox_but_true_effect_found():
    res = _court(TB.build_tabular_search(CD.health_supplement()))
    head = res["verdicts"]["health_score|overall"]
    assert head["verdict"] == "LUCK" and "Simpson" in head["cause_of_death"]
    sim = next(a for a in head["attacks"] if a["attack"] == "simpson")["evidence"]
    assert sim["pooled_effect"] < 0 < sim["stratum_adjusted_effect"]          # reversal
    assert any(d.startswith("health_score|age_band=") for d in res["discoveries"])  # real within-stratum benefit found


def test_ab_test_real_segment_found_and_peekers_die():
    res = _court(TB.build_tabular_search(CD.ab_test()))
    assert "conversion|device=mobile" in res["discoveries"]
    assert all(d.startswith("conversion|") for d in res["discoveries"])       # only the planted metric survives
    causes = [v["cause_of_death"] for v in res["verdicts"].values() if v["cause_of_death"]]
    assert any("peeking" in c for c in causes)


def test_betting_edge_is_not_skill():
    ev, out = run_scenario("betting", {"n_worlds": 150})
    v = out["result"]["verdicts"]["bet|all predicted bets"]
    assert v["verdict"] != "DISCOVERY" and "placebo" in (v["cause_of_death"] or "").lower()
    assert out["result"]["discoveries"] == []


def test_runner_streams_all_event_types_and_is_deterministic():
    ev1, o1 = run_scenario("synthetic", {"n_worlds": 120}, seed=3)
    ev2, o2 = run_scenario("synthetic", {"n_worlds": 120}, seed=3)
    assert o1["result"]["discoveries"] == o2["result"]["discoveries"]
    types = {e["type"] for e in ev1}
    assert {"meta", "naive", "null_audit", "verdict", "credibility", "ledger", "literature", "truth_check", "done"} <= types
    sc = next(e for e in ev1 if e["type"] == "truth_check")["score"]
    assert sc["fp_trap"] == 0 and sc["fp_beta"] == 0 and sc["fp_null"] == 0


def test_conclusion_has_plain_english_answer():
    from skeptic.backend import conclude as C
    ev, _ = run_scenario("health", {"n_worlds": 150})
    r = C.conclude("Do supplement users have better health?", ev)
    assert r["answer"].startswith("Short answer:") and "not proof of cause" in r["answer"]
    none = C.plain_answer("q?", {"findings": [], "n_tested": 5, "n_luck": 5, "n_unclear": 0, "n_valid_total": 0})
    assert "we cannot say" in none
