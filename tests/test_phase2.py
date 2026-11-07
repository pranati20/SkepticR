import numpy as np
from skeptic.backend.ledger import Ledger
from skeptic.backend.court import null_worlds as NW, stats as S
from skeptic.backend.adapters.synthetic import SynthConfig, generate


def _noise(K, T, seed):
    r = np.random.default_rng(seed)
    return r.standard_t(4, (K, T)) / np.sqrt(2), np.zeros((T, 1))


def test_fwer_controlled_on_pure_noise():
    """Known answer: with no signal anywhere, P(any claim) ~ alpha (allow Monte-Carlo slack)."""
    hits = 0
    n = 60
    for s in range(n):
        pnl, fac = _noise(40, 300, s)
        r = NW.run_null_worlds(pnl, fac, S.raw_t, "stationary_bootstrap", n_worlds=150, seed=s)
        hits += (r.p_stepdown < 0.05).any()
    assert hits / n <= 0.12


def test_strong_signal_found_in_noise():
    pnl, fac = _noise(100, 500, 3)
    pnl[7] += 0.25                       # annualised SR ~ 4
    r = NW.run_null_worlds(pnl, fac, S.raw_t, "sign_flip", n_worlds=200, seed=0)
    assert r.p_stepdown[7] < 0.05 and (r.p_stepdown < 0.05).sum() == 1


def test_adjusted_p_monotone_and_ordered():
    pnl, fac = _noise(50, 300, 5)
    r = NW.run_null_worlds(pnl, fac, S.raw_t, "sign_flip", n_worlds=100, seed=1)
    assert (r.p_single >= r.p_unadjusted - 1e-12).all()
    assert (r.p_stepdown <= r.p_single + 1e-12).all()


def test_stationary_indices_valid():
    idx = NW.stationary_indices(np.random.default_rng(0), 500, 10)
    assert idx.min() >= 0 and idx.max() < 500 and len(idx) == 500


def test_rc_spa_detect_and_not_on_noise():
    pnl, _ = _noise(50, 400, 9)
    assert NW.reality_check_and_spa(pnl, B=200)["spa_p"] > 0.05
    pnl[3] += 0.2
    assert NW.reality_check_and_spa(pnl, B=200)["spa_p"] < 0.05


def test_synthetic_planted_recovered_decoys_rejected_by_alpha_stat():
    d = generate(SynthConfig(seed=11))
    r = NW.run_null_worlds(d.pnl, d.factors, S.alpha_t, "stationary_bootstrap", n_worlds=200, seed=2)
    claim = (r.p_stepdown < 0.05) & (r.observed > 0)
    assert (claim & d.is_true).sum() >= 3          # recovers most planted
    assert (claim & (d.kinds == "decoy")).sum() == 0
    assert (claim & (d.kinds == "null")).sum() == 0


def test_ledger_counts_everything():
    L = Ledger()
    run = L.start_run("t")
    for a in ("momentum-hunter", "meanrev-hunter"):
        for i in range(5):
            L.record(run, a, "momentum", f"h{a}{i}", spec={"i": i}, statistic=1.0)
    L.record_cleaning(run, "cleaner", "winsorize", 0.01, n_options=2, rows_affected=12)
    L.record_cleaning(run, "cleaner", "gaps", "ffill", n_options=2)
    e = L.effective_search_size(run)
    assert e["tests"] == 10 and e["cleaning_choices"] == 2 and e["agents"] == 2
    assert e["cleaning_paths"] == 4 and e["potential_paths"] == 40
    L.set_status_by_hypothesis(run, "hmomentum-hunter0", "luck")
    assert L.status_counts(run)["luck"] == 1
