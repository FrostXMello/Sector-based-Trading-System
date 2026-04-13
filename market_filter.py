from __future__ import annotations

import pandas as pd

from feature_engineering import build_features


def compute_market_filter_frame(
    nifty_df: pd.DataFrame,
    *,
    use_rsi_filter: bool = False,
    rsi_min: float = 50.0,
) -> pd.DataFrame:
    out = build_features(nifty_df)
    if use_rsi_filter:
        out["Market_ON"] = (out["Close"] > out["MA50"]) & (out["RSI14"] > float(rsi_min))
    else:
        out["Market_ON"] = out["Close"] > out["MA50"]
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out


def is_market_favorable(market_frame: pd.DataFrame, date: pd.Timestamp) -> bool:
    d = pd.Timestamp(date).normalize()
    row = market_frame.loc[market_frame["Date"] <= d]
    if row.empty:
        return False
    return bool(row.iloc[-1]["Market_ON"])
