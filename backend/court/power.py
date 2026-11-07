"""Statistical power for the whole-search audit (no LLM)."""
from __future__ import annotations

import numpy as np
from scipy import stats

ANN = 252


def critical_value(null_max: np.ndarray, alpha: float = 0.05) -> float:
    """Best-by-luck statistic that luck beats only `alpha` of the time (from the null worlds)."""
    return float(np.quantile(null_max, 1 - alpha))


def power_at(sr_true: float, n_days: int, crit: float) -> float:
    """P(audit flags a hypothesis whose true annualised Sharpe is sr_true), t ~ N(sr*sqrt(T/252), 1)."""
    return float(stats.norm.sf(crit - sr_true * np.sqrt(n_days / ANN)))


def days_needed(sr: float, crit_per_sqrt_t: float | None = None, crit: float = 3.5, power: float = 0.8) -> int | None:
    """Days of data needed for an effect of annualised Sharpe `sr` to clear `crit` with probability `power`.
    Note `crit` is taken at the current search size; more data does not change it."""
    if sr <= 0:
        return None
    z = crit + stats.norm.ppf(power)
    return int(np.ceil((z / sr) ** 2 * ANN))
