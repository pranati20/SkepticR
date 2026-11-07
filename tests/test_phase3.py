import numpy as np
from skeptic.backend.ledger import Ledger
from skeptic.backend.court.alpha_investing import AlphaInvesting
from skeptic.backend.court import judge as J, prereg, power, null_worlds as NW, stats as S


def test_alpha_investing_never_negative_and_pays():
    ai = AlphaInvesting()
    lv = ai.propose_level(0.5)
    assert 0 < lv < ai.wealth0
    w0 = ai.wealth
    ai.update(lv, False)
    assert ai.wealth < w0
    ai.update(lv, True)
    assert ai.wealth > w0 - 0.1
    for _ in range(500):
        ai.update(ai.propose_level(0.5), False)
    assert ai.wealth >= 0 and ai.propose_level(0.5) < 1e-3


def test_prior_scales_level_but_not_beyond_wealth():
    ai = AlphaInvesting()
    assert ai.propose_level(0.9) > ai.propose_level(0.1)
    assert ai.propose_level(1.0) / (1 - ai.propose_level(1.0)) <= ai.wealth


def test_qvalues_known_answer():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    q = J.qvalues(p, 1.0)
    assert np.isclose(q[0], 0.01) and q.max() <= 1 and (np.diff(q[np.argsort(p)]) >= -1e-12).all()


def test_confession_text():
    s = J.confession(500, 3, 0.012)
    assert "500" in s and "3 cleaning" in s and "1.2%" in s


def test_prereg_hash_and_oos():
    L = Ledger(); run = L.start_run()
    rec = prereg.register(L, run, "h1", {"f": "mom", "n": 12}, +1)
    assert prereg.verify(rec, {"f": "mom", "n": 12}) and not prereg.verify(rec, {"f": "mom", "n": 13})
    rng = np.random.default_rng(0)
    res = prereg.score_out_of_sample(L, rec, rng.normal(0.002, 0.01, 400))
    assert res["confirmed"]
    res2 = prereg.score_out_of_sample(L, rec, rng.normal(0, 0.01, 400))
    assert not res2["confirmed"]


def test_power_monotone():
    c = 3.5
    assert power.power_at(2.0, 1260, c) > power.power_at(1.0, 1260, c)
    assert power.days_needed(1.0, crit=c) > power.days_needed(2.0, crit=c)


def test_judge_verdicts():
    rng = np.random.default_rng(1)
    pnl = rng.standard_t(4, (80, 400)) / np.sqrt(2); pnl[5] += 0.3
    fac = np.zeros((400, 1))
    r = NW.run_null_worlds(pnl, fac, S.raw_t, "sign_flip", n_worlds=200, seed=1)
    search = {"effective": 80, "cleaning_choices": 2}
    conf = J.confidence_scores(r.p_unadjusted)
    v = J.judge_one("h5", r, 5, 0.05, [], search, conf[5], 400)
    assert v.verdict == J.DISCOVERY
    killed = J.judge_one("h5", r, 5, 0.05, [{"attack": "costs", "severity": "high", "survived": False}], search, conf[5], 400)
    assert killed.verdict == J.LUCK and killed.cause_of_death == "Died after costs"
    worst = int(np.argmin(np.abs(r.observed)))
    v2 = J.judge_one("hw", r, worst, 0.05, [], search, conf[worst], 400)
    assert v2.verdict == J.LUCK and "multiple comparisons" in v2.cause_of_death
