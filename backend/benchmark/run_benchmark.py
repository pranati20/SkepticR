"""The proof. Runs every method over many seeds of the synthetic planted-signal benchmark and writes demo/benchmark.json.

    python -m skeptic.backend.benchmark.run_benchmark --seeds 20 --power-seeds 6

Reports precision / recall / FDR with 95% CIs, calibration (stated confidence vs actual hit rate), a power curve
(detection rate vs true signal strength) and measured failure modes. Everything is seeded.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..adapters.synthetic import SynthConfig, generate
from ..court import null_worlds as NW, stats as S
from ..court.pipeline import Court, CourtConfig, score_against_truth
from ..ledger import Ledger
from ..search import from_synthetic

OUT = Path(__file__).resolve().parents[2] / "demo" / "benchmark.json"
N_WORLDS = 200


def court_discoveries(search, seed, **cfgkw):
    L = Ledger()
    run = L.start_run("bench")
    res = Court(L, cfg=CourtConfig(seed=seed, n_worlds=N_WORLDS, **cfgkw)).run(search, run)
    disc = np.array([h["id"] in set(res["discoveries"]) for h in search.hyps])
    conf = np.array([res["verdicts"][h["id"]]["confidence"] for h in search.hyps])
    verd = np.array([res["verdicts"][h["id"]]["verdict"] for h in search.hyps])
    return disc, conf, verd, res


def _score(claims, truth_kind):
    truth = truth_kind == "alpha"
    n, tp = int(claims.sum()), int((claims & truth).sum())
    return {"claimed": n, "tp": tp, "fp": n - tp, "precision": tp / n if n else None, "recall": tp / max(1, int(truth.sum())),
            "fdr": (n - tp) / n if n else 0.0, "fp_decoy": int((claims & (truth_kind == "beta")).sum()),
            "fp_trap": int((claims & (truth_kind == "trap")).sum()), "fp_null": int((claims & (truth_kind == "null")).sum())}


def one_seed(args):
    seed, strength = args
    d = generate(SynthConfig(seed=seed, strength_mult=strength))
    se = from_synthetic(d, seed)
    kind = se.truth
    p = S.pvalues_two_sided(d.pnl)
    out = {"seed": seed, "strength": strength, "methods": {}, "detect": {}}
    claims = {
        "Naive analyst (p<0.05)": S.naive(p),
        "Single train/test split": S.train_test_split(d.pnl),
        "Bonferroni": S.bonferroni(p),
        "Benjamini-Hochberg": S.benjamini_hochberg(p),
        "Deflated Sharpe": S.deflated_sharpe(d.pnl),
    }
    nw = NW.run_null_worlds(d.pnl, d.factors, S.raw_t, "stationary_bootstrap", N_WORLDS, seed=seed + 1000)
    claims["Null-world audit only"] = (nw.p_stepdown < 0.05) & (nw.observed > 0)
    disc, conf, verd, res = court_discoveries(se, seed)
    claims["SKEPTIC (full court)"] = disc
    for m, c in claims.items():
        out["methods"][m] = _score(c, kind)
        out["detect"][m] = [(float(h["true_sr"]), bool(c[i])) for i, h in enumerate(d.meta) if h["kind"] == "planted"]
    out["calib"] = [(float(c), bool(k == "alpha")) for c, k in zip(conf, kind)]
    out["verdicts"] = {v: int((verd == v).sum()) for v in set(verd)}
    out["cant_tell_true"] = int(((verd == "CAN'T TELL YET") & (kind == "alpha")).sum())
    out["luck_true"] = int(((verd == "LUCK") & (kind == "alpha")).sum())
    return out


def failure_modes(seed: int = 0) -> list[dict]:
    """Measured, honest failure modes."""
    fm = []
    # 1. hidden factor: a decoy loads on a factor the Prosecutor does not know about
    d = generate(SynthConfig(seed=seed))
    se = from_synthetic(d, seed)
    se.factors, se.factors_hold, se.factor_names = d.factors[:, :2], d.factors_holdout[:, :2], d.factor_names[:2]
    disc, *_ = court_discoveries(se, seed)
    fm.append({"title": "Hidden confounders pass", "detail": "If a decoy loads on a factor the Prosecutor is not given (here VAL and MOM withheld), "
               "the confounder attack cannot see it. Skeptic only controls for factors it knows about.",
               "metric": f"{int((disc & (se.truth == 'beta')).sum())} of 10 decoys declared DISCOVERY when VAL and MOM are hidden "
                         f"(vs {int((court_discoveries(from_synthetic(d, seed), seed)[0] & (se.truth == 'beta')).sum())} with full factors)"})
    # 2. weak real signals are not detectable
    fm.append({"title": "Weak real effects look like luck", "detail": "A real signal needs enough data to stand out from the best result luck can produce in "
               "a search of this size. Below that, Skeptic says LUCK or CAN'T TELL YET even when the effect is real. See the power curve.",
               "metric": "power curve: detection rate vs true Sharpe"})
    # 3. traps rely on spec-level checks in the synthetic benchmark
    fm.append({"title": "Synthetic traps are caught from the spec", "detail": "In the synthetic benchmark leakage and survivorship are flagged in the hypothesis spec. "
               "On the market panel the Prosecutor re-runs the strategy to find them from data. A leak hidden from both would pass.",
               "metric": "see equities demo for the data-driven version"})
    # 4. null generator assumptions
    rng = np.random.default_rng(seed)
    k = 0
    for s in range(40):
        pnl = rng.standard_t(4, (80, 500)) / np.sqrt(2)
        pnl *= (1 + 2 * (np.sin(np.arange(500) / 40) > 0.8))              # regime-switching volatility the bootstrap partly mixes
        r = NW.run_null_worlds(pnl, np.zeros((500, 1)), S.raw_t, "stationary_bootstrap", 100, seed=s)
        k += (r.p_stepdown < 0.05).any()
    fm.append({"title": "Null-world validity depends on the null generator", "detail": "Block bootstrap keeps short-range dependence only. Long memory or "
               "regime structure the generator does not preserve can make p-values too optimistic.",
               "metric": f"family-wise false alarm rate on pure noise with volatility regimes: {k/40:.0%} (nominal 5%)"})
    # 5. alpha-investing starvation
    fm.append({"title": "Credibility budget can run dry", "detail": "Alpha-investing deliberately shrinks the level each round spends when the swarm keeps failing. "
               "A swarm with no real signals becomes more selective and eventually cannot declare anything, by design.", "metric": "see Credibility Meter"})
    return fm


def run(n_seeds=20, power_seeds=6, workers=4):
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        main = list(ex.map(one_seed, [(s, 1.0) for s in range(n_seeds)]))
        strengths = [0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2]
        power = list(ex.map(one_seed, [(100 + s, st) for st in strengths if st != 1.0 for s in range(power_seeds)]))
    ci = lambda x: (lambda a: (float(np.mean(a)), float(1.96 * np.std(a, ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0))(np.array([v for v in x if v is not None], float))
    methods = list(main[0]["methods"])
    table = {}
    for m in methods:
        rows = [r["methods"][m] for r in main]
        table[m] = {k: {"mean": ci([x[k] for x in rows])[0], "ci95": ci([x[k] for x in rows])[1]}
                    for k in ("claimed", "tp", "fp", "precision", "recall", "fdr", "fp_decoy", "fp_trap", "fp_null")}
        # pooled FDR (total fp / total claimed) avoids seeds with zero claims
        tf, tc = sum(x["fp"] for x in rows), sum(x["claimed"] for x in rows)
        table[m]["pooled_fdr"] = tf / tc if tc else 0.0
    # calibration
    pts = np.array([c for r in main for c in r["calib"]], float)
    edges = [0, 0.05, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
    calib = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (pts[:, 0] >= lo) & (pts[:, 0] < hi)
        if m.sum():
            calib.append({"lo": lo, "hi": min(hi, 1.0), "mean_stated": float(pts[m, 0].mean()), "actual": float(pts[m, 1].mean()), "n": int(m.sum())})
    # power curve pooled by true SR
    allruns = main + power
    sr_bins = [0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 8.0]
    curve = {}
    for m in ("SKEPTIC (full court)", "Benjamini-Hochberg", "Bonferroni", "Naive analyst (p<0.05)"):
        pts_m = np.array([x for r in allruns for x in r["detect"][m]], float)
        curve[m] = []
        for lo, hi in zip(sr_bins[:-1], sr_bins[1:]):
            mm = (pts_m[:, 0] >= lo) & (pts_m[:, 0] < hi)
            if mm.sum() >= 5:
                curve[m].append({"sr": float(pts_m[mm, 0].mean()), "detect": float(pts_m[mm, 1].mean()), "n": int(mm.sum())})
    verd = {}
    for r in main:
        for k, v in r["verdicts"].items():
            verd[k] = verd.get(k, 0) + v
    result = {"n_seeds": n_seeds, "n_worlds": N_WORLDS, "table": table, "calibration": calib, "power_curve": curve,
              "verdict_totals_per_run": {k: v / n_seeds for k, v in verd.items()},
              "real_signals_called_luck_per_run": float(np.mean([r["luck_true"] for r in main])),
              "real_signals_cant_tell_per_run": float(np.mean([r["cant_tell_true"] for r in main])),
              "failure_modes": failure_modes(), "setup": {"signals": 500, "planted": 5, "decoys": 10, "traps": 5, "assets": 200,
                                                          "search_days": 1260, "holdout_days": 252, "planted_sr": [1.8, 2.2, 2.6, 3.0, 3.5]},
              "runtime_s": time.time() - t0}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    return result


def print_summary(r):
    hdr = f"{'Method':26s}{'claimed':>8s}{'TP':>6s}{'FP':>7s}{'precision':>12s}{'recall':>10s}{'FDR(pooled)':>13s}   FP decoy/trap/null"
    print(hdr); print("-" * len(hdr))
    for m, v in r["table"].items():
        pr = v["precision"]["mean"]
        print(f"{m:26s}{v['claimed']['mean']:8.1f}{v['tp']['mean']:6.2f}{v['fp']['mean']:7.1f}{pr:8.2f}±{v['precision']['ci95']:.2f}"
              f"{v['recall']['mean']:7.2f}±{v['recall']['ci95']:.2f}{v['pooled_fdr']:10.2f}      "
              f"{v['fp_decoy']['mean']:.1f}/{v['fp_trap']['mean']:.1f}/{v['fp_null']['mean']:.1f}")
    print("\nCalibration (stated confidence -> actual hit rate):")
    for c in r["calibration"]:
        print(f"  [{c['lo']:.2f},{c['hi']:.2f})  stated {c['mean_stated']:.2f}  actual {c['actual']:.2f}  n={c['n']}")
    print("\nPower (detection rate by true Sharpe):")
    for m, pts in r["power_curve"].items():
        print(f"  {m:26s}", " ".join(f"{p['sr']:.1f}:{p['detect']:.2f}" for p in pts))
    print(f"\nreal signals called LUCK per run: {r['real_signals_called_luck_per_run']:.2f}; CAN'T TELL: {r['real_signals_cant_tell_per_run']:.2f}")
    print("Failure modes:"); [print("  -", f["title"], "|", f["metric"]) for f in r["failure_modes"]]
    print(f"runtime {r['runtime_s']:.0f}s -> {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--power-seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    print_summary(run(a.seeds, a.power_seeds, a.workers))
