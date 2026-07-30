"""
Bootstrap real GICS-like sector/industry labels for the workbook tickers.

The cross-sectional pipeline currently fakes sectors with
`sector_assign = np.arange(n) % 11` (alphabetical order modulo 11), which
makes 11 of 22 node features pure noise and turns the "sector-based proxy"
ablation into a random block graph. This fetches real sector/industry from
yfinance as an interim fix until Bloomberg GICS_SECTOR_NAME arrives.

Output: sectors_bootstrap.csv with columns
    ticker (Bloomberg format, e.g. "A UN"), yf_ticker, sector, industry
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd
import yfinance as yf


def bloomberg_to_yf(bbg_ticker: str) -> str:
    """'A UN' -> 'A'; '002475 CH' -> '' (non-US, skip)."""
    parts = str(bbg_ticker).split()
    if not parts:
        return ""
    base, exch = parts[0], (parts[1] if len(parts) > 1 else "")
    # Only US listings map cleanly to plain yfinance symbols.
    if exch not in ("UN", "UW", "UA", "US", "UR", "UQ"):
        return ""
    return base.replace("/", "-")


def main():
    xlsx = sys.argv[1]
    out_path = Path(sys.argv[2])

    sheets = pd.read_excel(xlsx, sheet_name=None, usecols=["ticker"])
    tickers = sorted({str(t) for df in sheets.values() for t in df["ticker"].dropna().unique()})
    print(f"{len(tickers)} unique tickers from workbook", flush=True)

    rows = []
    for i, bbg in enumerate(tickers):
        yft = bloomberg_to_yf(bbg)
        sector = industry = ""
        if yft:
            try:
                info = yf.Ticker(yft).info
                sector = info.get("sector") or ""
                industry = info.get("industry") or ""
            except Exception as exc:  # network/ratelimit/delisted
                print(f"  [{i}] {bbg} -> {yft}: {type(exc).__name__}", flush=True)
        rows.append({"ticker": bbg, "yf_ticker": yft, "sector": sector, "industry": industry})
        if (i + 1) % 25 == 0:
            got = sum(1 for r in rows if r["sector"])
            print(f"  {i+1}/{len(tickers)} done, {got} with sector", flush=True)
            pd.DataFrame(rows).to_csv(out_path, index=False)
        time.sleep(0.15)

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    n_ok = int((df["sector"] != "").sum())
    print(f"\nwrote {out_path}")
    print(f"coverage: {n_ok}/{len(df)} ({n_ok/len(df):.1%})")
    print("\nsector distribution:")
    print(df[df["sector"] != ""]["sector"].value_counts().to_string())


if __name__ == "__main__":
    main()
