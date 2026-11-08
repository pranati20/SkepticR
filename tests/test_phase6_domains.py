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
    assert r["answer"].startswith("Yes.") and "Why:" in r["answer"] and "not proof of cause" in r["answer"]
    none = C.plain_answer("q?", {"findings": [], "n_tested": 5, "n_luck": 5, "n_unclear": 0, "n_valid_total": 0})
    assert none.startswith("Can't say")


def test_question_decides_the_analysis_not_a_fixed_answer():
    from skeptic.backend import conclude as C
    assert C.classify("What are the outliers in the data?") == "outliers"
    assert C.classify("Does X increase Y?") == "effect"
    assert C.classify("tell me a joke") == "other"
    ev, _ = run_scenario("health", {"n_worlds": 120, "question": "What are the outliers in the health supplement data?"})
    c = next(e for e in ev if e["type"] == "conclusion")
    assert c["kind"] == "outliers" and c["findings"] == [] and "outlier" in c["answer"]
    assert not c["answer"].startswith("Yes.")
    ev, _ = run_scenario("health", {"n_worlds": 120, "question": "tell me a joke"})
    assert next(e for e in ev if e["type"] == "conclusion")["answer"].startswith("Can't say")


def test_every_paper_says_what_was_used_from_it():
    from skeptic.backend import literature as L
    p = {"title": "t", "abstract": "Background sentence here is long enough to count. We show that survivorship bias inflates average returns materially."}
    u = L.used_paragraph(p, L.METHOD_USE["survivorship"], ["survivorship"])
    assert "survivorship bias inflates" in u and "Survivorship Skeptic" in u and "abstract only" in u
    assert "No abstract" in L.used_paragraph({"title": "t"}, "x")


def test_papers_tab_entries_say_what_was_used(tmp_path, monkeypatch):
    monkeypatch.setenv("SKEPTIC_LIBRARY_PATH", str(tmp_path / "p.json"))
    from skeptic.backend import library as LB
    LB.record([{"title": "A", "url": "u1", "abstract": "We study survivorship bias in returns and show it matters a great deal.", "used": "What the paper says (its abstract): \"x\" How we use it: X."}], "ctx1")
    LB.record([{"title": "B", "url": "u2", "abstract": "Plain abstract about sequential testing and stopping rules in trials."}], "peeking question")
    es = {e["title"]: e for e in LB.entries()}
    assert es["A"]["used"][0].endswith("How we use it: X.")
    assert "peeking question" in es["B"]["used"][0] and "abstract only" in es["B"]["used"][0]
