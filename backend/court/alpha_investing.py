"""Foster & Stine (2008) generalised alpha-investing: online control of the marginal FDR.

The swarm holds an "alpha wealth" balance (the Credibility Meter). Each round of tests spends wealth;
each validated discovery pays wealth back. When wealth runs low the swarm must be more selective.

Design note: the *prior* of a hypothesis (literature-motivated > random) scales how much wealth a round may
spend, so well-motivated ideas get a bigger level. The prior NEVER touches the null-world test itself.
The null audit supplies the p-value; alpha-investing supplies the level it must beat.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AlphaInvesting:
    wealth0: float = 0.09
    payout: float = 0.05              # omega: earned per rejection (must be <= alpha for mFDR control)
    base_share: float = 0.4           # fraction of wealth a unit-prior round may stake
    min_wealth: float = 1e-4          # below this the swarm is broke
    wealth: float = field(init=False)
    history: list = field(default_factory=list)

    def __post_init__(self):
        self.wealth = self.wealth0
        self.history.append({"event": "init", "wealth": self.wealth})

    def propose_level(self, prior: float = 0.5, min_level: float = 0.0) -> float:
        """Level alpha_j for the next test. prior in [0,1]; prior=0.5 -> factor 1.

        `min_level` is the smallest p-value the null audit can resolve (1/(n_worlds+1)). Without a floor,
        repeated failures shrink the level below that resolution and no test can ever reject again (a
        "sterile lock"). With the floor the swarm can still win wealth back, at the price that when it is
        nearly broke the mFDR guarantee is only approximate; such tests are marked `floored` in the history."""
        s = self.wealth * self.base_share * (0.5 + min(max(prior, 0.0), 1.0))
        s = min(s, self.wealth)                       # can never stake more than we hold
        level = s / (1.0 + s)                         # cost of failure = alpha/(1-alpha) = s
        if level < min_level:
            self.history.append({"event": "floored", "level": level, "min_level": min_level, "wealth": self.wealth})
            level = min_level
        return level

    def update(self, level: float, rejected: bool, label: str = "") -> float:
        if rejected:
            self.wealth += min(self.payout, level)          # Foster-Stine: reward omega <= the level alpha_j just spent
        else:
            self.wealth -= level / (1.0 - level)
        self.wealth = max(self.wealth, 0.0)
        self.history.append({"event": "reject" if rejected else "spend", "level": level,
                             "wealth": self.wealth, "label": label})
        return self.wealth

    @property
    def broke(self) -> bool:
        return self.wealth < self.min_wealth

    def max_ideas_allowed(self, floor_prior: float) -> float:
        """Selectivity signal for Hunters: as wealth shrinks, the minimum prior they should submit rises."""
        frac = self.wealth / self.wealth0
        return min(1.0, max(0.0, floor_prior + (1 - frac) * (1 - floor_prior)))
