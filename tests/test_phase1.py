import numpy as np
from skeptic.backend.adapters.synthetic import SynthConfig, generate
from skeptic.backend.court import stats as S


def test_bh_known_answer():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    # BH q=0.05, m=10: thresholds .005,.01,.015,.02,... -> first two pass
    assert S.benjamini_hochberg(p, 0.05).sum() == 2


def test_bonferroni_known_answer():
    p = np.array([0.001, 0.02, 0.2, 0.5])
    assert list(S.bonferroni(p, 0.05)) == [True, False, False, False]


def test_generator_shapes_and_truth():
    d = generate(SynthConfig(n_days=400, n_assets=60, n_signals=80, seed=1))
    assert d.pnl.shape == (80, 400)
    assert d.is_true.sum() == 5
    assert (d.kinds == "decoy").sum() == 10
    assert sum(m["spec"]["uses_future_bar"] for m in d.meta) == 3
    assert np.isfinite(d.pnl).all()


def test_generator_is_seeded():
    a = generate(SynthConfig(n_days=200, n_assets=40, n_signals=60, seed=7))
    b = generate(SynthConfig(n_days=200, n_assets=40, n_signals=60, seed=7))
    assert np.array_equal(a.pnl, b.pnl)


def test_planted_signals_beat_nulls():
    d = generate(SynthConfig(seed=2))
    sr = S.sharpe(d.pnl)
    assert sr[d.is_true].mean() > 1.2
    assert abs(sr[d.kinds == "null"].mean()) < 0.2
