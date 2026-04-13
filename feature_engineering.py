from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "MA20",
    "MA50",
    "RSI14",
    "Daily_Return",
    "Rolling_Volatility_10",
    "TrendStrength",
]


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_features(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the required technical feature set.
    """
    req = {"Date", "Close"}
    if not req.issubset(price_df.columns):
        raise ValueError(f"Missing required columns: {sorted(req - set(price_df.columns))}")

    df = price_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)

    df["MA20"] = df["Close"].rolling(20, min_periods=20).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=50).mean()
    df["MA100"] = df["Close"].rolling(100, min_periods=100).mean()
    df["RSI14"] = compute_rsi(df["Close"], 14)
    df["Daily_Return"] = df["Close"].pct_change()
    df["Rolling_Volatility_10"] = df["Daily_Return"].rolling(10, min_periods=10).std()
    df["Rolling_Volatility_5"] = df["Daily_Return"].rolling(5, min_periods=5).std()
    df["TrendStrength"] = (df["MA20"] > df["MA50"]).astype(int)
    if "High" in df.columns and "Low" in df.columns:
        df["RangeExpansion"] = (df["High"] - df["Low"]) / df["Close"].replace(0, np.nan)
        df["RangeExpansionMean5"] = df["RangeExpansion"].rolling(5, min_periods=5).mean()
    else:
        df["RangeExpansion"] = np.nan
        df["RangeExpansionMean5"] = np.nan
    df["Prev5High"] = df["High"].shift(1).rolling(5, min_periods=5).max() if "High" in df.columns else df["Close"].shift(1).rolling(5, min_periods=5).max()
    df["Prev3High"] = df["High"].shift(1).rolling(3, min_periods=3).max() if "High" in df.columns else df["Close"].shift(1).rolling(3, min_periods=3).max()
    df["High20"] = df["High"].shift(1).rolling(20, min_periods=20).max() if "High" in df.columns else df["Close"].shift(1).rolling(20, min_periods=20).max()
    df["PrevClose"] = df["Close"].shift(1)
    df["PrevClose2"] = df["Close"].shift(2)
    df["PrevClose3"] = df["Close"].shift(3)
    df["Return_3d"] = df["Close"] / df["Close"].shift(3) - 1.0
    df["PrevDayHigh"] = df["High"].shift(1) if "High" in df.columns else np.nan
    df["MA20_prev"] = df["MA20"].shift(1)
    df["MA20_prev2"] = df["MA20"].shift(2)
    df["RSI14_prev"] = df["RSI14"].shift(1)
    df["RSI_Zone"] = np.select(
        [df["RSI14"] < 40, df["RSI14"] <= 60],
        ["Weak", "Neutral"],
        default="Strong",
    )
    return df


def add_relative_strength(feature_df: pd.DataFrame, nifty_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stock-vs-NIFTY relative strength features on 5/10 day horizons.

    rs_5 = stock_5_day_return - nifty_5_day_return
    rs_10 = stock_10_day_return - nifty_10_day_return
    """
    df = feature_df.copy()
    ndf = nifty_df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    ndf["Date"] = pd.to_datetime(ndf["Date"]).dt.normalize()

    ndf = ndf.sort_values("Date").drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    ndf["NIFTY_R5"] = ndf["Close"] / ndf["Close"].shift(5) - 1.0
    ndf["NIFTY_R10"] = ndf["Close"] / ndf["Close"].shift(10) - 1.0

    merged = df.merge(ndf[["Date", "NIFTY_R5", "NIFTY_R10"]], on="Date", how="left")
    merged["RS_5"] = (merged["Close"] / merged["Close"].shift(5) - 1.0) - merged["NIFTY_R5"]
    merged["RS_10"] = (merged["Close"] / merged["Close"].shift(10) - 1.0) - merged["NIFTY_R10"]
    return merged


def add_target(
    feature_df: pd.DataFrame,
    *,
    horizon_days: int = 5,
    threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Target = 1 if forward return over horizon_days > threshold else 0.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be >= 1")

    df = feature_df.copy()
    df["Forward_Return"] = df["Close"].shift(-horizon_days) / df["Close"] - 1.0
    # Require trend persistence in the look-ahead path: future closes should stay above MA20.
    future_above_ma20 = pd.Series(True, index=df.index)
    for k in range(1, horizon_days + 1):
        future_above_ma20 &= (df["Close"].shift(-k) > df["MA20"].shift(-k))
    df["Target"] = ((df["Forward_Return"] > float(threshold)) & future_above_ma20).astype(int)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + ["Forward_Return", "Target"]).reset_index(drop=True)
    return df


def build_labeled_dataset(
    price_df: pd.DataFrame,
    *,
    ticker: str | None = None,
    horizon_days: int = 5,
    threshold: float = 0.01,
) -> pd.DataFrame:
    df = add_target(build_features(price_df), horizon_days=horizon_days, threshold=threshold)
    if ticker is not None:
        df["Ticker"] = str(ticker)
    return df

