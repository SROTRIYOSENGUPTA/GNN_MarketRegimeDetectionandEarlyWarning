"""
Build the canonical sector/industry file used by run_xsec_rank.py --sector-file.

Sources, in priority order:
  1. Bloomberg GICS (`gics_500.csv`): authoritative 4-level GICS hierarchy plus
     CUSIP. Pulled against the CURRENT S&P 500, so firms that exited the index
     during the sample are absent — itself a concrete illustration of why
     historical index membership (INDX_MWEIGHT_HIST) matters.
  2. yfinance bootstrap (`data_sectors_bootstrap.csv`): Yahoo's own taxonomy.
     Used ONLY at the sector level, via an explicit crosswalk to GICS names, and
     only for firms Bloomberg didn't cover.

Deliberate choice: the fallback fills the SECTOR level only. Industry group /
industry / sub-industry stay blank for fallback firms rather than splicing two
incompatible taxonomies together — Yahoo's "industry" is not a GICS industry, and
mixing them would manufacture phantom categories that corrupt the granularity
comparison. Blank means "unknown bucket" downstream, which is honest.

Output: data_sectors_gics.csv with columns
    ticker, sector, gics_industry_group, gics_industry, gics_sub_industry,
    cusip, source

Note on truncation: Bloomberg BDP returned category names clipped to 30
characters ("Semiconductors & Semiconductor" for "... Equipment"). This is
cosmetic for grouping — the distinct-category counts match published GICS
(11 sectors / 25 industry groups) — but full names must be restored before any
paper table quotes them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# Yahoo Finance sector names -> GICS sector names.
YF_TO_GICS = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
    "Communication Services": "Communication Services",
    "Energy": "Energy",
    "Industrials": "Industrials",
    "Real Estate": "Real Estate",
    "Utilities": "Utilities",
}


def base_ticker(t: str) -> str:
    """'FITB UW' -> 'FITB'. Lets us match across exchange-code differences."""
    return str(t).split()[0].upper()


def main():
    gics_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/Users/ssrotriyo/Desktop/gics_500.csv")
    boot_path = REPO / "data_sectors_bootstrap.csv"
    out_path = REPO / "data_sectors_gics.csv"

    gics = pd.read_csv(gics_path, keep_default_na=False)
    gics = gics[gics["ticker"].astype(str).str.strip() != ""]
    boot = pd.read_csv(boot_path, keep_default_na=False)

    lookup = {}
    for _, r in gics.iterrows():
        rec = r.to_dict()
        lookup[r["ticker"]] = rec
        lookup.setdefault(base_ticker(r["ticker"]), rec)

    rows = []
    for _, w in boot.iterrows():
        ticker = w["ticker"]
        src = lookup.get(ticker) or lookup.get(base_ticker(ticker))
        if src is not None:
            rows.append({
                "ticker": ticker,
                "sector": src["gics_sector"],
                "gics_industry_group": src["gics_industry_group"],
                "gics_industry": src["gics_industry"],
                "gics_sub_industry": src["gics_sub_industry"],
                "cusip": "" if "N/A" in str(src["cusip"]) else src["cusip"],
                "source": "bloomberg_gics",
            })
            continue

        yf_sector = str(w["sector"]).strip()
        mapped = YF_TO_GICS.get(yf_sector, "")
        if yf_sector and not mapped:
            print(f"  WARNING: unmapped yfinance sector {yf_sector!r} for {ticker}", flush=True)
        rows.append({
            "ticker": ticker,
            "sector": mapped,
            # Finer levels intentionally blank: Yahoo's taxonomy is not GICS.
            "gics_industry_group": "",
            "gics_industry": "",
            "gics_sub_industry": "",
            "cusip": "",
            "source": "yfinance_sector_only" if mapped else "unmapped",
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)

    print(f"wrote {out_path}\n")
    print(out["source"].value_counts().to_string())
    print()
    for col in ["sector", "gics_industry_group", "gics_industry", "gics_sub_industry"]:
        nonblank = out[out[col].astype(str).str.strip() != ""]
        print(f"  {col:<22} {nonblank[col].nunique():>4} distinct  |  {len(nonblank):>3}/{len(out)} firms")
    print()
    print(f"  cusip present          {int((out['cusip'].astype(str).str.strip() != '').sum()):>4}/{len(out)}")


if __name__ == "__main__":
    main()
