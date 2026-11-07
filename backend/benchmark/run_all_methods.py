"""Phase 1 check: run baselines over many seeds and print a methods table.

    python -m skeptic.backend.benchmark.run_all_methods --seeds 5
"""
from __future__ import annotations

import argparse
import time
import numpy as np

from ..adapters.synthetic import SynthConfig, generate
from ..court import stats as S
from ..court import null_worlds as NW

METHODS = {
    "Naive (p<0.05)": lambda d, p: S.naive(p),
    "Single train/test split": lambda d, p: S.train_test_split(d.pnl),
    "Bonferroni": lambda d, p: S.bonferroni(p),
    "Benjamini-Hochberg": lambda d, p: S.benjamini_hochberg(p),
    "Deflated Sharpe": lambda d, p: S.deflated_sharpe(d.pnl),
}


def null_world_methods(n_worlds: int = 300, method: str = "stationary_bootstrap") -> dict:
    """Phase-2 'Skeptic' rows: whole-search max-statistic test (Romano-Wolf step-down), alpha = 0.05.
    Prosecutors (Phase 5) are layered on top for the full court; these rows are the null-world engine alone."""
    def make(stat):
        def fn(d, p):
            r = NW.run_null_worlds(d.pnl, d.factors, stat, method, n_worlds=n_worlds, seed=int(d.cfg.seed) + 1000)
            return (r.p_stepdown < 0.05) & (r.observed > 0)          # positive-direction discoveries only
        return fn
    return {"Null-world search audit (raw t)": make(S.raw_t),
            "Null-world audit (factor-adj alpha)": make(S.alpha_t)}


def score(claims: np.ndarray, data) -> dict:
    truth = data.is_true
    kinds = data.kinds
    n_claim = int(claims.sum())
    tp = int((claims & truth).sum())
    fp = n_claim - tp
    return {
        "claimed": n_claim, "tp": tp, "fp": fp,
        "precision": tp / n_claim if n_claim else np.nan,
        "recall": tp / truth.sum(),
        "fdr": fp / n_claim if n_claim else 0.0,
        "fp_decoy": int((claims & (kinds == "decoy")).sum()),
        "fp_trap": int((claims & np.char.startswith(kinds.astype(str), "trap")).sum()),
        "fp_null": int((claims & (kinds == "null")).sum()),
    }


def ci95(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float(np.mean(x)) if len(x) else float("nan"), 0.0
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(len(x)))


def run(n_seeds: int = 5, extra_methods: dict | None = None, cfg_kwargs: dict | None = None, verbose=True):
    methods = dict(METHODS)
    methods.update(extra_methods or {})
    rows = {m: [] for m in methods}
    t0 = time.time()
    for seed in range(n_seeds):
        data = generate(SynthConfig(seed=seed, **(cfg_kwargs or {})))
        p = S.pvalues_two_sided(data.pnl)
        for m, fn in methods.items():
            rows[m].append(score(fn(data, p), data))
        if verbose:
            print(f"  seed {seed} done ({time.time() - t0:.0f}s)", flush=True)
    return rows


def print_table(rows):
    hdr = f"{'Method':26s} {'claimed':>8s} {'TP':>5s} {'FP':>6s} {'precision':>14s} {'recall':>14s} {'FDR':>14s}  FP: decoy/trap/null"
    print(hdr)
    print("-" * len(hdr))
    for m, rs in rows.items():
        g = lambda k: [r[k] for r in rs]
        pm, pc = ci95(g("precision")); rm, rc = ci95(g("recall")); fm, fc = ci95(g("fdr"))
        print(f"{m:26s} {np.mean(g('claimed')):8.1f} {np.mean(g('tp')):5.2f} {np.mean(g('fp')):6.1f} "
              f"{pm:8.2f}+/-{pc:4.2f} {rm:8.2f}+/-{rc:4.2f} {fm:8.2f}+/-{fc:4.2f}  "
              f"{np.mean(g('fp_decoy')):.1f}/{np.mean(g('fp_trap')):.1f}/{np.mean(g('fp_null')):.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    a = ap.parse_args()
    print(f"Synthetic benchmark: 500 signals, 5 planted, 10 confounded decoys, 5 traps; {a.seeds} seeds")
    print_table(run(a.seeds, extra_methods=null_world_methods()))
