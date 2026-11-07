"""Null-world engine: audits the WHOLE search, not just the winner. Zero LLM tokens.

Idea (White's Reality Check / Westfall-Young max-statistic permutation logic, applied to the agent loop):
  1. Build many "null worlds" that destroy any real signal but keep nasty structure
     (fat tails, volatility clustering, autocorrelation, cross-sectional correlation).
  2. In each world re-run the EXACT SAME search (every hypothesis the swarm tried, same params).
  3. Record the best statistic luck produced. The adjusted p-value of a finding is the fraction of worlds
     whose best statistic is >= the finding's.

World generators below operate on a (K,T) matrix of per-hypothesis pnl plus (T,f) factor returns, with
the *same* time transformation applied to every row so cross-sectional correlation survives.
Generators for raw-feature domains (label permutation, circular shift of outcomes) are at the bottom.

Assumptions (also surfaced in the UI/README):
  sign_flip            symmetric innovations given the volatility path; kills serial dependence in the *mean*,
                       keeps variance clustering & cross-correlation.
  stationary_bootstrap Politis-Romano; keeps short-range serial dependence (block ~ mean length b) and
                       cross-correlation; weaker for long memory.
  (phase_surrogate)    IAAFT-style surrogate for raw FEATURE series (see bottom). NOT used on pnl: phase
                       randomisation pins a series' sample mean at exactly zero, so it cannot null a mean-based
                       statistic. Use it to scramble a feature against its outcome instead.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

StatFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


# ----------------------------------------------------------------------------- world generators
def _demean(pnl: np.ndarray, fac: np.ndarray):
    return pnl - pnl.mean(axis=1, keepdims=True), fac - fac.mean(axis=0, keepdims=True)


def stationary_indices(rng: np.random.Generator, T: int, mean_block: float = 10.0) -> np.ndarray:
    """Politis-Romano stationary bootstrap time indices (vectorised)."""
    restart = rng.random(T) < 1.0 / mean_block
    restart[0] = True
    start_val = rng.integers(0, T, T)
    ar = np.arange(T)
    last = np.maximum.accumulate(np.where(restart, ar, 0))
    return (start_val[last] + (ar - last)) % T


def world_sign_flip(pnl, fac, rng, **_):
    p, f = _demean(pnl, fac)
    s = rng.choice([-1.0, 1.0], size=p.shape[1])
    return p * s, f * s[:, None]


def world_stationary_bootstrap(pnl, fac, rng, mean_block: float = 10.0, **_):
    p, f = _demean(pnl, fac)
    idx = stationary_indices(rng, p.shape[1], mean_block)
    return p[:, idx], f[idx]


GENERATORS = {
    "sign_flip": world_sign_flip,
    "stationary_bootstrap": world_stationary_bootstrap,
}


# ----------------------------------------------------------------------------- engine
@dataclass
class NullResult:
    method: str
    n_worlds: int
    observed: np.ndarray                 # signed stats (K,)
    null_stats: np.ndarray               # (B,K) |stat| in each null world
    p_single: np.ndarray = None          # single-step max-stat adjusted p (K,)  (White RC style, per hypothesis)
    p_stepdown: np.ndarray = None        # Romano-Wolf step-down adjusted p (K,)
    p_unadjusted: np.ndarray = None      # (1+#{null >= obs}) / (B+1) per hypothesis alone
    null_max: np.ndarray = None          # (B,) best-by-luck statistic per world
    search_size: int = 0

    def luck_rate(self, k: int) -> float:
        """Fraction of null worlds where luck alone matched/beat hypothesis k's statistic anywhere in the search."""
        return float(self.p_single[k])


def adjusted_pvalues(obs_abs: np.ndarray, null_abs: np.ndarray):
    """single-step max-T and Romano-Wolf step-down p-values from a (B,K) null matrix."""
    B, K = null_abs.shape
    null_max = null_abs.max(axis=1)
    p_single = (1.0 + (null_max[:, None] >= obs_abs[None, :]).sum(axis=0)) / (B + 1.0)
    p_unadj = (1.0 + (null_abs >= obs_abs[None, :]).sum(axis=0)) / (B + 1.0)
    order = np.argsort(-obs_abs)                              # strongest first
    nul_sorted = null_abs[:, order]
    suffix_max = np.maximum.accumulate(nul_sorted[:, ::-1], axis=1)[:, ::-1]     # max over k' at/after rank j
    p_sorted = (1.0 + (suffix_max >= obs_abs[order][None, :]).sum(axis=0)) / (B + 1.0)
    p_sorted = np.maximum.accumulate(p_sorted)                # enforce monotonicity
    p_step = np.empty(K)
    p_step[order] = np.minimum(p_sorted, 1.0)
    return p_single, p_step, p_unadj, null_max


def run_null_worlds(pnl: np.ndarray, factors: np.ndarray, statistic: StatFn, method: str = "stationary_bootstrap",
                    n_worlds: int = 500, seed: int = 0, **gen_kwargs) -> NullResult:
    """Re-run the exact same search (`statistic` on every row) in `n_worlds` null worlds."""
    gen = GENERATORS[method]
    rng = np.random.default_rng(seed)
    obs = statistic(pnl, factors)
    null_abs = np.empty((n_worlds, pnl.shape[0]))
    for b in range(n_worlds):
        pw, fw = gen(pnl, factors, rng, **gen_kwargs)
        null_abs[b] = np.abs(statistic(pw, fw))
    p_single, p_step, p_unadj, null_max = adjusted_pvalues(np.abs(obs), null_abs)
    return NullResult(method=method, n_worlds=n_worlds, observed=obs, null_stats=null_abs, p_single=p_single,
                      p_stepdown=p_step, p_unadjusted=p_unadj, null_max=null_max, search_size=pnl.shape[0])


# ----------------------------------------------------------------------------- RC / SPA family-level tests
def _bootstrap_means(d: np.ndarray, B: int, rng, mean_block: float):
    K, T = d.shape
    out = np.empty((K, B))
    for b in range(B):
        out[:, b] = d[:, stationary_indices(rng, T, mean_block)].mean(axis=1)
    return out


def reality_check_and_spa(pnl: np.ndarray, B: int = 500, mean_block: float = 10.0, seed: int = 0) -> dict:
    """White (2000) Reality Check and Hansen (2005) SPA p-values against a zero benchmark.

    Family-level tests: 'is the best of ALL these rules better than nothing?' (positive direction).
    """
    rng = np.random.default_rng(seed)
    K, T = pnl.shape
    dbar = pnl.mean(axis=1)
    boot = _bootstrap_means(pnl, B, rng, mean_block)                       # (K,B)
    omega = np.sqrt(T) * boot.std(axis=1, ddof=1)
    omega = np.where(omega <= 0, np.nan, omega)
    t_obs = np.sqrt(T) * dbar / omega
    stat_rc = max(np.nanmax(np.sqrt(T) * dbar), 0.0)
    stat_spa = max(np.nanmax(t_obs), 0.0)
    # Reality Check: recentre at dbar (least favourable)
    rc_null = np.maximum((np.sqrt(T) * (boot - dbar[:, None])).max(axis=0), 0.0)
    # SPA: studentised, consistent recentring that drops clearly-poor rules
    A = np.sqrt(2 * np.log(np.log(T)))
    g_c = np.where(t_obs >= -A, dbar, 0.0)
    spa_null = np.maximum(np.nanmax(np.sqrt(T) * (boot - g_c[:, None]) / omega[:, None], axis=0), 0.0)
    return {"rc_stat": float(stat_rc), "rc_p": float((1 + (rc_null >= stat_rc).sum()) / (B + 1)),
            "spa_stat": float(stat_spa), "spa_p": float((1 + (spa_null >= stat_spa).sum()) / (B + 1))}


# ----------------------------------------------------------------------------- raw-feature domain generators
def label_permutation(labels: np.ndarray, rng: np.random.Generator, strata: np.ndarray | None = None) -> np.ndarray:
    """Permute outcome labels (A/B tests, sport results, subgroup effects); optionally within strata."""
    out = labels.copy()
    if strata is None:
        return rng.permutation(out)
    for s in np.unique(strata):
        m = np.where(strata == s)[0]
        out[m] = rng.permutation(out[m])
    return out


def circular_shift_outcome(outcome: np.ndarray, rng: np.random.Generator, min_shift: int = 20) -> np.ndarray:
    """Circularly shift an outcome series relative to its features (keeps the outcome's own autocorrelation)."""
    T = len(outcome)
    return np.roll(outcome, int(rng.integers(min_shift, max(min_shift + 1, T - min_shift))), axis=0)


def placebo_dates(n_events: int, T: int, rng: np.random.Generator, exclude: np.ndarray | None = None) -> np.ndarray:
    """Fake event dates for placebo tests."""
    pool = np.setdiff1d(np.arange(T), exclude if exclude is not None else [])
    return np.sort(rng.choice(pool, size=n_events, replace=False))


def phase_surrogate(x: np.ndarray, rng: np.random.Generator, n_iter: int = 10) -> np.ndarray:
    """IAAFT surrogate of a 1-D feature series: same amplitude distribution and ~same power spectrum, random phases.
    Use on the *feature* (not on pnl) so its relationship to the outcome is destroyed but its own dynamics survive."""
    x = np.asarray(x, float)
    sorted_x = np.sort(x)
    amp = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(n_iter):
        Y = np.fft.rfft(y)
        y = np.fft.irfft(amp * np.exp(1j * np.angle(Y)), n=len(x))
        y = sorted_x[np.argsort(np.argsort(y))]
    return y
