from __future__ import annotations

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd

from data_loader import DownloadConfig, download_stock_data


NIFTY50_TICKERS: list[str] = [
    "ADANIENT.NS",
    "ADANIPORTS.NS",
    "APOLLOHOSP.NS",
    "ASIANPAINT.NS",
    "AXISBANK.NS",
    "BAJAJ-AUTO.NS",
    "BAJFINANCE.NS",
    "BAJAJFINSV.NS",
    "BEL.NS",
    "BHARTIARTL.NS",
    "BPCL.NS",
    "BRITANNIA.NS",
    "CIPLA.NS",
    "COALINDIA.NS",
    "DRREDDY.NS",
    "EICHERMOT.NS",
    "ETERNAL.NS",
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
    "SHRIRAMFIN.NS",
    "SUNPHARMA.NS",
    "TATACONSUM.NS",
    "TATAMOTORS.NS",
    "TATASTEEL.NS",
    "TCS.NS",
    "TECHM.NS",
    "TITAN.NS",
    "TRENT.NS",
    "WIPRO.NS",
]


@dataclass(frozen=True)
class DataEngineConfig:
    years: int = 5
    intraday_intervals: tuple[str, ...] = ("1m", "5m")
    include_daily: bool = True


# In-memory cache for faster repeated backtests in the same app session.
_PRICE_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}
_CACHE_LOCK = Lock()


def _cached_download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    key = (str(ticker), str(period), str(interval))
    with _CACHE_LOCK:
        in_cache = key in _PRICE_CACHE
    if not in_cache:
        df = download_stock_data(ticker, DownloadConfig(period=period, interval=interval))
        with _CACHE_LOCK:
            _PRICE_CACHE[key] = df
    # Return a copy so downstream code can safely modify without mutating cache.
    with _CACHE_LOCK:
        return _PRICE_CACHE[key].copy()


def load_daily_universe(tickers: list[str] | None = None, years: int = 5) -> dict[str, pd.DataFrame]:
    period = f"{max(1, int(years))}y"
    out: dict[str, pd.DataFrame] = {}
    names = list(tickers or NIFTY50_TICKERS)
    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(_cached_download, t, period, "1d"): t for t in names}
        for f in as_completed(fut):
            t = fut[f]
            try:
                out[t] = f.result()
            except Exception:
                continue
    return out


def load_multi_timeframe_universe(
    tickers: list[str] | None = None,
    cfg: DataEngineConfig | None = None,
) -> dict[str, dict[str, pd.DataFrame]]:
    config = cfg or DataEngineConfig()
    out: dict[str, dict[str, pd.DataFrame]] = {}
    names = list(tickers or NIFTY50_TICKERS)

    def _load_one(t: str) -> tuple[str, dict[str, pd.DataFrame]]:
        frames: dict[str, pd.DataFrame] = {}
        if config.include_daily:
            period = f"{max(1, int(config.years))}y"
            try:
                frames["1d"] = _cached_download(t, period, "1d")
            except Exception:
                pass
        if config.intraday_intervals:
            for itv in tuple(config.intraday_intervals):
                period = "1y"
                itv_norm = str(itv).lower()
                if itv_norm in {"1m", "2m", "5m", "15m", "30m"}:
                    period = "60d"
                elif itv_norm in {"60m", "90m", "1h"}:
                    period = "730d"
                try:
                    frames[itv_norm] = _cached_download(t, period, itv_norm)
                except Exception:
                    pass
        return t, frames

    with ThreadPoolExecutor(max_workers=6) as ex:
        fut = {ex.submit(_load_one, t): t for t in names}
        for f in as_completed(fut):
            try:
                t, frames = f.result()
                if frames:
                    out[t] = frames
            except Exception:
                continue
    return out
