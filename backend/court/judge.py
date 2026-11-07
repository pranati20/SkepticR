"""The Judge: combines null-world evidence, alpha-investing level, Prosecutor results and statistical power.

Verdicts (never overclaims; reduces false discoveries and quantifies uncertainty, it does not prove truth):
  DISCOVERY      p_adj <= level AND survived every high-severity attack
  LUCK           p_adj >= p_luck (luck routinely does this well somewhere in the search) OR killed by an attack
  CAN'T TELL YET in between: not enough evidence to call it real, not enough to rule it out; says how much
                 more data would be needed
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import power as P

DISCOVERY, LUCK, CANT_TELL = "DISCOVERY", "LUCK", "CAN'T TELL YET"

CAUSE_OF_DEATH = {
    "multiple_comparisons": "Died of multiple comparisons",
    "leakage": "Died of leakage",
    "survivorship": "Died of survivorship bias",
    "costs": "Died after costs",
    "regime": "Died in a bad regime",
    "concentration": "Died of concentration",
    "confounder": "Died of beta (it was never alpha)",
    "fragility": "Died of parameter fragility",
    "simpson": "Died to Simpson's paradox",
    "peeking": "Died of peeking",
}


@dataclass
class Verdict:
    hypothesis_id: str
    verdict: str
    confidence: float                 # calibrated P(real) estimate in [0,1]
    p_adj: float
    p_luck_rate: float                # fraction of identical searches where luck alone matches this result
    level: float                      # alpha-investing level it had to beat
    cause_of_death: str | None
    confession: str
    search_size: int
    cleaning_choices: int
    n_null_worlds: int
    power_at_observed: float | None = None
    more_data_days: int | None = None
    attacks: list = field(default_factory=list)
    statistic: float | None = None

    def to_dict(self):
        return self.__dict__.copy()


def confession(search_size: int, cleaning_choices: int, p_luck_rate: float) -> str:
    pct = 100 * p_luck_rate
    pct_s = f"{pct:.1f}%" if pct < 10 else f"{pct:.0f}%"
    return (f"I tried {search_size:,} ideas (including {cleaning_choices} cleaning choices) to find this. "
            f"Luck alone would produce something this good {pct_s} of the time.")


def storey_pi0(p: np.ndarray, lam: float = 0.5) -> float:
    return float(min(1.0, (p > lam).mean() / (1 - lam)))


def qvalues(p: np.ndarray, pi0: float = 1.0) -> np.ndarray:
    """Storey q-values: estimated FDR if we reject everything at least as extreme as k."""
    m = len(p)
    order = np.argsort(p)
    q = pi0 * p[order] * m / np.arange(1, m + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(q, 1.0)
    return out


def confidence_scores(p_unadj_emp: np.ndarray) -> np.ndarray:
    """Calibrated-by-construction P(real): 1 - q-value computed from the empirical-null marginal p-values."""
    pi0 = storey_pi0(p_unadj_emp)
    return 1.0 - qvalues(p_unadj_emp, pi0)


def judge_one(hyp_id: str, null, k: int, level: float, attacks: list[dict] | None, search: dict,
              confidence: float, n_days: int, p_luck: float = 0.5, alpha_crit: float = 0.05,
              p_adj_override: float | None = None) -> Verdict:
    """Decide one hypothesis. `null` is a NullResult; k its index in the searched family."""
    attacks = attacks or []
    p_adj = float(null.p_stepdown[k]) if p_adj_override is None else float(p_adj_override)
    p_rate = float(null.p_single[k])
    sr = float(null.observed[k]) / np.sqrt(n_days) * np.sqrt(252) if n_days else 0.0
    crit = P.critical_value(null.null_max, alpha_crit)
    killers = [a for a in attacks if a.get("severity") == "high" and not a.get("survived", True)]
    common = dict(hypothesis_id=hyp_id, p_adj=p_adj, p_luck_rate=p_rate, level=level,
                  search_size=search["effective"], cleaning_choices=search["cleaning_choices"],
                  n_null_worlds=null.n_worlds, attacks=attacks, statistic=float(null.observed[k]),
                  confession=confession(search["effective"], search["cleaning_choices"], p_rate))
    if killers:
        cause = CAUSE_OF_DEATH.get(killers[0]["attack"], f"Died of {killers[0]['attack']}")
        return Verdict(verdict=LUCK, confidence=min(confidence, 0.05), cause_of_death=cause, **common)
    if p_adj <= level:
        return Verdict(verdict=DISCOVERY, confidence=confidence, cause_of_death=None,
                       power_at_observed=P.power_at(sr, n_days, crit), **common)
    if p_adj >= p_luck:
        return Verdict(verdict=LUCK, confidence=min(confidence, 0.05),
                       cause_of_death=CAUSE_OF_DEATH["multiple_comparisons"], **common)
    return Verdict(verdict=CANT_TELL, confidence=confidence, cause_of_death=None,
                   power_at_observed=P.power_at(sr, n_days, crit),
                   more_data_days=P.days_needed(sr, crit=crit), **common)
