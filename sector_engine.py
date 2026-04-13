from __future__ import annotations

import numpy as np
import pandas as pd

from data_loader import DownloadConfig, download_stock_data
from sector_universe import SECTOR_INDEX_YAHOO


def _safe_score(close: pd.Series) -> tuple[float, float, float, float]:
    ret = close.pct_change().dropna()
    if len(ret) < 12:
        return float("nan"), float("nan"), float("nan"), float("nan")

    ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else float("nan")
    ret_10d = float(close.iloc[-1] / close.iloc[-11] - 1.0) if len(close) >= 11 else float("nan")
    vol = float(ret.tail(20).std())
    if not np.isfinite(vol) or vol <= 0:
        score = float("nan")
    else:
        score = (0.6 * ret_5d + 0.4 * ret_10d) / vol
    return ret_5d, ret_10d, vol, float(score)


def compute_sector_scores(
    sector_prices: dict[str, pd.DataFrame] | None = None,
    *,
    top_n: int = 3,
    period: str = "2y",
) -> tuple[pd.DataFrame, list[str]]:
    """
    Compute sector momentum-volatility scores and return ranked table + top sectors.
    """
    top_n = int(max(1, min(7, int(top_n))))

    if sector_prices is None:
        sector_prices = {}
        for sector, ticker in SECTOR_INDEX_YAHOO.items():
            try:
                sector_prices[sector] = download_stock_data(ticker, DownloadConfig(period=period, interval="1d"))
            except Exception:
                continue

    rows: list[dict] = []
    for sector, df in sector_prices.items():
        if df is None or df.empty or "Close" not in df.columns:
            continue
        close = df["Close"].astype(float)
        ret_5d, ret_10d, vol, score = _safe_score(close)
        rows.append(
            {
                "Sector": sector,
                "Return_5d": ret_5d,
                "Return_10d": ret_10d,
                "Volatility": vol,
                "SectorScore": score,
            }
        )

    rank_df = pd.DataFrame(rows)
    if rank_df.empty:
        return rank_df, []

    rank_df = rank_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["SectorScore"])
    if rank_df.empty:
        return rank_df, []
    rank_df = rank_df.sort_values("SectorScore", ascending=False).reset_index(drop=True)
    top_sectors = rank_df["Sector"].head(top_n).tolist()
    return rank_df, top_sectors
