"""Synthetic planted-signal benchmark generator.

Simulates a daily cross-section of assets with realistic noise:
  * Student-t (df=4) innovations (fat tails)
  * shared stochastic volatility path (volatility clustering)
  * AR(1) idio autocorrelation, sector shocks + 4 known factors (cross-sectional correlation)

Then builds `n_signals` candidate long/short signals, summarised as daily PnL series:
  * null     - persistent random scores (no information)
  * planted  - scores equal to a persistent latent component that really predicts returns
  * decoy    - scores are loadings on a *known factor with a premium* (confounded: beta, not alpha)
  * trap_lookahead   - score uses the next-bar return (spec flag uses_future_bar=True)
  * trap_survivorship - score = "is currently listed" on a universe where delisted names drifted down

Ground truth lives in `meta`. Only planted signals count as true discoveries.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

FACTOR_NAMES = ["MKT", "SIZE", "VAL", "MOM"]
FAMILIES = ["momentum", "mean_reversion", "seasonality", "volatility", "value", "sentiment"]


@dataclass
class SynthConfig:
    n_assets: int = 200
    n_days: int = 1260                      # ~5 years
    n_signals: int = 500
    planted_sr: tuple = (1.8, 2.2, 2.6, 3.0, 3.5)   # target annualised Sharpe of planted signals
    strength_mult: float = 1.0              # scales planted SRs (power curve knob)
    n_decoys_val: int = 5
    n_decoys_mom: int = 5
    n_lookahead: int = 3
    n_survivor: int = 2
    cost_bps_default: float = 3.0
    n_holdout: int = 252                    # later slice never touched during search (pre-registration scoring)
    seed: int = 0


@dataclass
class SynthData:
    cfg: SynthConfig
    pnl: np.ndarray            # (K, T) daily gross pnl per signal
    turnover: np.ndarray       # (K,) mean daily one-way gross turnover
    factors: np.ndarray        # (T, 4) factor returns aligned with pnl dates
    factor_names: list
    returns: np.ndarray        # (T, N) next-day asset returns aligned with pnl dates
    pnl_holdout: np.ndarray = None       # (K, H) later slice, never used by search
    factors_holdout: np.ndarray = None   # (H, 4)
    meta: list = field(default_factory=list)   # list of dict, length K

    @property
    def kinds(self):
        return np.array([m["kind"] for m in self.meta])

    @property
    def is_true(self):
        return self.kinds == "planted"


def _tnoise(rng, shape):
    return rng.standard_t(4, size=shape) / np.sqrt(2.0)


def _ar1_matrix(rng, T, shape_tail, phi):
    """Unit-variance AR(1) paths, phi scalar or broadcastable to shape_tail."""
    out = np.empty((T,) + tuple(shape_tail))
    x = rng.normal(size=shape_tail)
    s = np.sqrt(1 - np.asarray(phi) ** 2)
    for t in range(T):
        x = phi * x + s * rng.normal(size=shape_tail)
        out[t] = x
    return out


def _weights(S):
    """Cross-sectionally demean scores (axis 0) and scale to unit gross exposure."""
    Sc = S - S.mean(axis=0, keepdims=True)
    g = np.abs(Sc).sum(axis=0, keepdims=True)
    g[g == 0] = 1.0
    return Sc / g


def generate(cfg: SynthConfig | None = None) -> SynthData:
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    T, N, K = cfg.n_days + cfg.n_holdout, cfg.n_assets, cfg.n_signals
    T1 = T + 1

    # --- shared stochastic volatility path (clustering) ---
    g = np.zeros(T1)
    shocks = rng.normal(size=T1)
    for t in range(1, T1):
        g[t] = 0.98 * g[t - 1] + 0.06 * shocks[t]
    h = np.exp(g)[:, None]

    # --- known factors with premia ---
    mu_f = np.array([0.0003, 0.00005, 0.00017, 0.00017])
    sd_f = np.array([0.010, 0.001, 0.001, 0.001])
    F = mu_f + sd_f * h * _tnoise(rng, (T1, 4))
    B = np.column_stack([rng.normal(1.0, 0.3, N)] + [rng.normal(0, 1, N) for _ in range(3)])

    # --- sector shocks + idio with AR(1) ---
    sector = rng.integers(0, 10, N)
    sec = 0.004 * h * _tnoise(rng, (T1, 10))
    sig_i = 0.015 * np.exp(0.3 * rng.normal(size=N))
    eps = sig_i * h * _tnoise(rng, (T1, N))
    for t in range(1, T1):
        eps[t] += 0.05 * eps[t - 1]
    R = F @ B.T + sec[:, sector] + eps

    # --- survivorship structure: 15% of names delist; they drifted down throughout ---
    delisted = rng.random(N) < 0.15
    R[:, delisted] -= 0.0005

    # --- planted latent predictors, calibrated to target SR ---
    n_pl = len(cfg.planted_sr)
    phis = rng.uniform(0.98, 0.995, n_pl)
    M = np.stack([_ar1_matrix(rng, T1, (N,), phis[k]) for k in range(n_pl)], axis=2)  # (T1,N,n_pl)
    for k in range(n_pl):
        W = _weights(M[:T, :, k].T).T                      # (T,N)
        base = (W * R[1:T1]).sum(axis=1)                    # pnl without the planted component
        target_mean = cfg.planted_sr[k] * cfg.strength_mult / np.sqrt(252) * base.std()
        q = (W * M[:T, :, k]).sum(axis=1).mean()
        R[1:T1] += (target_mean / q) * M[:T, :, k]          # r_{t+1} += c * m_t

    # --- slot assignment ---
    kinds = (["planted"] * n_pl + ["decoy_val"] * cfg.n_decoys_val + ["decoy_mom"] * cfg.n_decoys_mom
             + ["trap_lookahead"] * cfg.n_lookahead + ["trap_survivorship"] * cfg.n_survivor)
    kinds += ["null"] * (K - len(kinds))
    pnl = np.zeros((K, T))
    turn = np.zeros(K)
    Rn = R[1:T1]                                            # (T,N) returns realised after score at t

    def run(idx, score_fn):
        G = len(idx)
        wprev = None
        tot = np.zeros(G)
        for t in range(T):
            w = _weights(score_fn(t))
            pnl[idx, t] = (w * Rn[t][:, None]).sum(axis=0)
            if wprev is not None:
                tot += np.abs(w - wprev).sum(axis=0)
            wprev = w
        turn[idx] = tot / (T - 1)

    kinds_arr = np.array(kinds)
    # planted
    idx = np.where(kinds_arr == "planted")[0]
    run(idx, lambda t: M[t])
    # nulls
    idx = np.where(kinds_arr == "null")[0]
    phi_n = rng.uniform(0.9, 0.99, len(idx))
    state = rng.normal(size=(N, len(idx)))
    sn = np.sqrt(1 - phi_n ** 2)

    def null_fn(t):
        nonlocal state
        state = phi_n * state + sn * rng.normal(size=state.shape)
        return state
    run(idx, null_fn)
    # decoys (confounded on a known factor)
    for kind, col in (("decoy_val", 2), ("decoy_mom", 3)):
        idx = np.where(kinds_arr == kind)[0]
        if len(idx) == 0:
            continue
        load = B[:, col][:, None]
        nz = 0.5 * _ar1_matrix(rng, T, (N, len(idx)), 0.95)
        run(idx, lambda t, load=load, nz=nz: load + nz[t])
    # lookahead traps: score peeks at the next-bar return
    idx = np.where(kinds_arr == "trap_lookahead")[0]
    if len(idx):
        a = 3.0 / np.sqrt(N * 252)
        zn = rng.normal(size=(T, N, len(idx)))
        run(idx, lambda t: a * (Rn[t] / (Rn[t].std() + 1e-12))[:, None] + zn[t])
    # survivorship traps: static "currently listed" score
    idx = np.where(kinds_arr == "trap_survivorship")[0]
    if len(idx):
        base = np.where(delisted, -1.0, 1.0)[:, None]
        nz = 0.2 * _ar1_matrix(rng, T, (N, len(idx)), 0.99)
        run(idx, lambda t: base + nz[t])

    # --- shuffle so position leaks nothing; build meta ---
    perm = rng.permutation(K)
    pnl, turn, kinds_arr = pnl[perm], turn[perm], kinds_arr[perm]
    planted_sr_by_slot = {}
    for j, i in enumerate(np.where(np.array(kinds) == "planted")[0]):
        planted_sr_by_slot[i] = cfg.planted_sr[j] * cfg.strength_mult
    meta = []
    for new_i, old_i in enumerate(perm):
        kind = kinds[old_i]
        meta.append({
            "id": f"sig_{new_i:04d}",
            "kind": "decoy" if kind.startswith("decoy") else kind,
            "subkind": kind,
            "true_sr": planted_sr_by_slot.get(old_i, 0.0),
            "family": FAMILIES[int(rng.integers(0, len(FAMILIES)))],
            "spec": {
                "uses_future_bar": kind == "trap_lookahead",
                "universe": "current_constituents" if kind == "trap_survivorship" else "point_in_time",
                "factor_loading": {"decoy_val": "VAL", "decoy_mom": "MOM"}.get(kind),
            },
        })
    S_ = cfg.n_days
    return SynthData(cfg=cfg, pnl=pnl[:, :S_], turnover=turn, factors=F[1:T1][:S_], factor_names=FACTOR_NAMES,
                     returns=Rn[:S_].astype(np.float32), meta=meta,
                     pnl_holdout=pnl[:, S_:], factors_holdout=F[1:T1][S_:])
