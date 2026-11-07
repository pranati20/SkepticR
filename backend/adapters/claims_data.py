"""Simulated tabular datasets for the two non-market demos (clearly flagged as simulated)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .tabular import TabularSpec


def health_supplement(seed: int = 0, n: int = 8000) -> TabularSpec:
    """Headline: 'supplement users are far less healthy'. Truth: supplements help (+0.3 SD) in EVERY age band.
    Older people take supplements much more AND are less healthy, so the pooled comparison reverses (Simpson's paradox)."""
    rng = np.random.default_rng(seed)
    age = rng.choice(3, n, p=[0.4, 0.35, 0.25])
    sex = rng.integers(0, 2, n); region = rng.integers(0, 4, n); bmi = rng.choice(3, n, p=[.3, .45, .25]); smoker = (rng.random(n) < .2).astype(int)
    p_take = np.array([0.12, 0.35, 0.70])[age]
    take = (rng.random(n) < p_take).astype(int)
    score = np.array([0.9, 0.0, -1.1])[age] + 0.3 * take - 0.25 * smoker + rng.normal(0, 1, n)
    df = pd.DataFrame({"age_band": np.array(["18-39", "40-59", "60+"])[age], "sex": np.array(["F", "M"])[sex],
                       "region": np.array(["N", "S", "E", "W"])[region], "bmi": np.array(["low", "mid", "high"])[bmi],
                       "smoker": np.array(["no", "yes"])[smoker], "supplement": take, "health_score": score,
                       "enrolled": pd.date_range("2022-01-01", periods=n, freq="h")})
    return TabularSpec(df=df, treat="supplement", outcomes=["health_score"],
                       subgroup_cols=["sex", "region", "bmi", "smoker", "age_band"], strata_cols=["age_band"],
                       time_col="enrolled", claim="Supplement users are less healthy than non-users (so supplements are harmful).",
                       pairs=True)


def ab_test(seed: int = 0, n: int = 9000) -> TabularSpec:
    """Randomised experiment, 8 metrics x many segments, analyst peeks 10 times. One REAL effect: conversion on mobile."""
    rng = np.random.default_rng(seed)
    dev = rng.choice(3, n, p=[.55, .35, .10]); ctry = rng.choice(4, n, p=[.4, .3, .2, .1]); new = (rng.random(n) < .45).astype(int)
    plan = rng.choice(3, n, p=[.6, .3, .1])
    treat = rng.integers(0, 2, n)
    base = rng.normal(size=(n, 8))
    mobile = (dev == 0)
    # 8 metrics; only 'conversion' on mobile has a real effect (+0.22 SD)
    base[:, 0] += 0.22 * treat * mobile
    names = ["conversion", "revenue", "session_len", "pages_per_visit", "retention_7d", "nps", "tickets", "ad_clicks"]
    df = pd.DataFrame(base, columns=names)
    df["variant"] = treat
    df["device"] = np.array(["mobile", "desktop", "tablet"])[dev]; df["country"] = np.array(["US", "UK", "DE", "BR"])[ctry]
    df["user_type"] = np.array(["returning", "new"])[new]; df["plan"] = np.array(["free", "pro", "team"])[plan]
    df["arrival"] = np.arange(n)
    return TabularSpec(df=df, treat="variant", outcomes=names, subgroup_cols=["device", "country", "user_type", "plan"],
                       strata_cols=None, time_col="arrival", claim="The new checkout variant improves metrics (analyst peeked 10 times).",
                       peek_looks=10, pairs=False, holdout_frac=0.25)
