"""Plain-numpy/scipy statistics: per-signal tests and the multiple-testing baselines.

References: Bonferroni; Benjamini & Hochberg (1995); Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.
Zero LLM tokens anywhere in this module.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

ANN = 252


def sharpe(pnl: np.ndarray) -> np.ndarray:
    """Annualised Sharpe of each row of pnl (K,T)."""
    sd = pnl.std(axis=1, ddof=1)
    sd = np.where(sd == 0, np.nan, sd)
    return pnl.mean(axis=1) / sd * np.sqrt(ANN)


def tstat(pnl: np.ndarray) -> np.ndarray:
    T = pnl.shape[1]
    return sharpe(pnl) / np.sqrt(ANN) * np.sqrt(T)


def pvalues_two_sided(pnl: np.ndarray) -> np.ndarray:
    T = pnl.shape[1]
    t = tstat(pnl)
    return 2 * stats.t.sf(np.abs(t), df=T - 1)


# ---------------- baselines: each returns a boolean "claimed discovery" mask ----------------

def naive(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Naive analyst: anything with p < alpha (positive direction only is *not* enforced)."""
    return p < alpha


def bonferroni(p: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return p * len(p) < alpha


def benjamini_hochberg(p: np.ndarray, q: float = 0.05) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, m + 1) / m
    passed = p[order] <= thresh
    out = np.zeros(m, bool)
    if passed.any():
        k = np.max(np.where(passed)[0])
        out[order[: k + 1]] = True
    return out


def train_test_split(pnl: np.ndarray, frac: float = 0.6, alpha: float = 0.05) -> np.ndarray:
    """Single split: significant (same sign) in the first `frac` AND in the held-out remainder."""
    c = int(pnl.shape[1] * frac)
    a, b = pnl[:, :c], pnl[:, c:]
    return (pvalues_two_sided(a) < alpha) & (pvalues_two_sided(b) < alpha) & (np.sign(a.mean(1)) == np.sign(b.mean(1)))


def deflated_sharpe_prob(pnl: np.ndarray, n_trials: int | None = None) -> np.ndarray:
    """Probability that the true Sharpe > 0 after deflating for `n_trials` (Bailey & Lopez de Prado)."""
    K, T = pnl.shape
    n = n_trials or K
    sr = pnl.mean(axis=1) / pnl.std(axis=1, ddof=1)           # per-period Sharpe
    skew = stats.skew(pnl, axis=1)
    kurt = stats.kurtosis(pnl, axis=1, fisher=False)
    gamma = 0.5772156649
    sr0 = np.sqrt(np.var(sr, ddof=1)) * ((1 - gamma) * stats.norm.ppf(1 - 1 / n)
                                          + gamma * stats.norm.ppf(1 - 1 / (n * np.e)))
    denom = np.sqrt(np.maximum(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom)


def deflated_sharpe(pnl: np.ndarray, level: float = 0.95) -> np.ndarray:
    return deflated_sharpe_prob(pnl) > level
