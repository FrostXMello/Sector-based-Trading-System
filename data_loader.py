from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

from utils import validate_columns


@dataclass(frozen=True)
class DownloadConfig:
    period: str = "5y"
    interval: str = "1d"
    auto_adjust: bool = False


def download_stock_data(ticker: str, cfg: DownloadConfig | None = None) -> pd.DataFrame:
    """
    Download daily OHLCV data using yfinance.

    Returns a DataFrame with columns:
    Date, Open, High, Low, Close, Volume
    """
    if cfg is None:
        cfg = DownloadConfig()

    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("ticker must be a non-empty string")

    ticker = ticker.strip().upper()

    df = yf.download(
        tickers=ticker,
        period=cfg.period,
        interval=cfg.interval,
        auto_adjust=cfg.auto_adjust,
        progress=False,
        threads=True,
    )

    if df is None or df.empty:
        raise ValueError(f"No data returned for ticker: {ticker}")

    # yfinance returns DateTimeIndex; columns are often Open/High/Low/Close/Adj Close/Volume.
    df = df.reset_index()

    # yfinance frequently returns MultiIndex columns like (Price, Ticker) for single-ticker downloads.
    # We only need the first level (e.g., "Open", "High", "Close", ..., "Date").
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [str(col[0]) for col in df.columns]

    # Some yfinance versions name the date column "Date"; others can be "index".
    date_col = "Date" if "Date" in df.columns else df.columns[0]
    df = df.rename(columns={date_col: "Date"})

    needed = ["Open", "High", "Low", "Close", "Volume"]
    validate_columns(df, needed, df_name="raw download")
    df = df[["Date"] + needed].copy()

    # Clean numeric issues, sort, remove duplicate dates.
    df = df.replace([np.inf, -np.inf], np.nan)
    df["Date"] = pd.to_datetime(df["Date"], utc=False)
    df = df.sort_values("Date").drop_duplicates(subset=["Date"], keep="last")

    # Drop rows with missing OHLCV values.
    df = df.dropna(subset=needed).reset_index(drop=True)

    return df

