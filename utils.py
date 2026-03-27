from __future__ import annotations

import random
from typing import Iterable

import numpy as np
import pandas as pd

# Best-effort default universe (Yahoo Finance tickers, NSE suffix).
# Note: NIFTY constituents can change over time; the app will skip tickers
# that fail to download so you can edit the list in the UI if needed.
NIFTY50_TICKERS: list[str] = [
    "ADANIPORTS.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BPCL.NS",
    "BHARTIARTL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DIVISLAB.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "GRASIM.NS",
    "HCLTECH.NS",
    "HDFCBANK.NS",
    "HDFCLIFE.NS",
    "HEROMOTOCO.NS",
    "HINDALCO.NS",
    "HINDUNILVR.NS",
    "ICICIBANK.NS",
    "INDUSINDBK.NS",
    "INFY.NS",
    "ITC.NS",
    "JSWSTEEL.NS",
    "KOTAKBANK.NS",
    "LT.NS",
    "M&M.NS",
    "MARUTI.NS",
    "NESTLEIND.NS",
    "NTPC.NS",
    "ONGC.NS",
    "POWERGRID.NS",
    "RELIANCE.NS",
    "SBILIFE.NS",
    "SBIN.NS",
    "SHREECEM.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TATAPOWER.NS",
    "TECHM.NS",
    "TCS.NS",
    "TITAN.NS",
    "ULTRACEMCO.NS",
    "UPL.NS",
    "VEDL.NS",
    "WIPRO.NS",
]


def parse_tickers_text(tickers_text: str) -> list[str]:
    """
    Parse tickers from a comma/newline/space-separated text input.
    """
    if not tickers_text or not tickers_text.strip():
        return []
    raw = (
        tickers_text.replace("\n", ",")
        .replace("\t", ",")
        .replace(" ", ",")
        .split(",")
    )
    tickers = [t.strip().upper() for t in raw if t and t.strip()]
    # Deduplicate while preserving order.
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducibility (best-effort)."""
    random.seed(seed)
    np.random.seed(seed)


def validate_columns(df: pd.DataFrame, required: Iterable[str], *, df_name: str = "DataFrame") -> None:
    """Raise a helpful error if expected columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing columns: {missing}")


def safe_pct(x: float) -> float:
    """Format-safe percent conversion."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return x * 100.0

