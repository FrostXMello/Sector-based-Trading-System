from __future__ import annotations

import numpy as np
import pandas as pd

from utils import validate_columns


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute RSI using a simple rolling average of gains/losses.

    Note: This uses information up to and including the current day `t` (no look-ahead).
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def build_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a leak-free dataset for next-day prediction.

    Output DataFrame contains:
    - Trade-date fields: Date (the feature date), Close
    - Next-day fields for backtesting: Next_Date, Next_Close
    - Features: MA20, MA50, RSI14, Daily_Return, Rolling_Volatility_10
    - Target: Target (1 if next-day return > 0 else 0)
    """
    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    validate_columns(df, required, df_name="input prices")

    data = df.copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    # Features computed using only information up to day t.
    data["MA20"] = data["Close"].rolling(window=20, min_periods=20).mean()
    data["MA50"] = data["Close"].rolling(window=50, min_periods=50).mean()
    data["RSI14"] = _compute_rsi(data["Close"], period=14)
    data["Daily_Return"] = data["Close"].pct_change()
    data["Rolling_Volatility_10"] = data["Daily_Return"].rolling(window=10, min_periods=10).std()

    # Next-day return (label) aligned to features at day t.
    data["Next_Daily_Return"] = data["Daily_Return"].shift(-1)
    data["Target"] = (data["Next_Daily_Return"] > 0).astype(int)

    # Next-day values for backtesting the signal generated at day t.
    data["Next_Close"] = data["Close"].shift(-1)
    data["Next_Date"] = data["Date"].shift(-1)

    feature_cols = ["MA20", "MA50", "RSI14", "Daily_Return", "Rolling_Volatility_10"]
    drop_cols = feature_cols + ["Target", "Next_Close", "Next_Date"]

    # Drop any rows where rolling indicators or next-day label are undefined.
    data = data.dropna(subset=drop_cols).reset_index(drop=True)

    # Guard against infinities.
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=drop_cols).reset_index(drop=True)

    return data

