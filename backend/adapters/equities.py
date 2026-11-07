"""Real daily US equities from the pre-downloaded yfinance file (data/equities/prices.csv.gz).

Honest limits (shown in the UI): the universe is TODAY's large caps, so companies that failed or were acquired are
missing -> survivorship bias is built into the data. There are no fundamentals or sentiment in this file.
Everything here is post-publication for the classic anomalies (2010-2025), so replication tests answer 'does it still
work', not 'did it decay since publication'.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .market_panel import Panel

PATH = Path(__file__).resolve().parents[2] / "data" / "equities" / "prices.csv.gz"


def available() -> bool:
    return PATH.exists()


def load_panel(path: Path | str = PATH, max_assets: int | None = None) -> Panel:
    raw = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    close, vol = raw["close"], raw["volume"]
    if max_assets:
        close, vol = close.iloc[:, :max_assets], vol.iloc[:, :max_assets]
    close = close.ffill(limit=3)
    ret = close.pct_change(fill_method=None)
    alive = close.notna().to_numpy()
    f = {"close": close.to_numpy(), "ret": ret.to_numpy(), "volume": vol.reindex(close.index).to_numpy(),
         "btm": np.full(close.shape, np.nan), "sentiment": np.full(close.shape, np.nan)}
    return Panel(fields=f, dates=close.index, tickers=list(close.columns), alive=alive, listed_today=np.ones(close.shape[1], bool),
                 meta={"source": "yfinance (pre-downloaded)", "is_standin": False, "survivorship_bias": True,
                       "note": "Real prices for 117 current large-cap US stocks, 2010-2025. Universe = today's survivors: delisted names are "
                               "absent, so any strategy that buys past losers or holds long is flattered. No fundamentals or sentiment.",
                       "fields": ["close", "ret", "volume"], "planted_alpha": [], "planted_beta": []})
