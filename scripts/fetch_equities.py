"""Pre-download daily adjusted prices/volumes for ~120 large-cap US tickers (yfinance) into data/equities/.
Run once with network; afterwards everything works offline.   python skeptic/scripts/fetch_equities.py"""
import sys
from pathlib import Path
import pandas as pd
import yfinance as yf

TICKERS = """AAPL MSFT AMZN GOOGL META NVDA TSLA JPM JNJ V PG UNH HD MA XOM CVX KO PEP ABBV MRK COST WMT BAC PFE
CSCO ORCL ABT TMO CRM ACN MCD NKE DHR LIN TXN NEE UPS PM BMY AMGN HON QCOM LOW IBM CAT GS MS INTC
SBUX AMD BLK ISRG MDT GE AXP DE SPGI NOW LMT MMM CVS T VZ CMCSA DIS ADP MO MDLZ GILD TJX SYK BKNG PLD
CB USB PNC SCHW C WFC COP EOG SLB OXY DUK SO D AEP EXC MMC CL KMB GIS HSY SHW ECL APD ITW EMR ETN
ROK FDX CSX NSC UNP BA RTX GD NOC HAL BK TRV ALL AIG MET PRU AFL HUM CI ELV ZTS BDX BSX EW""".split()


def main(out="skeptic/data/equities/prices.csv.gz", start="2010-01-01", end="2025-06-30"):
    d = yf.download(sorted(set(TICKERS)), start=start, end=end, auto_adjust=True, progress=False, threads=False)
    close, vol = d["Close"].dropna(how="all"), d["Volume"].reindex(d["Close"].dropna(how="all").index)
    ok = close.columns[close.notna().mean() > 0.9]
    long = pd.concat({"close": close[ok], "volume": vol[ok]}, axis=1)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    long.to_csv(out)
    print(f"saved {out}: {close[ok].shape[1]} tickers x {close.shape[0]} days; dropped {len(set(TICKERS)) - len(ok)}")


if __name__ == "__main__":
    main(*sys.argv[1:])
