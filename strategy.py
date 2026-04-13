from __future__ import annotations

import numpy as np
import pandas as pd


def probability_to_signal(
    probability: float,
    price: float,
    ma20: float,
    ma50: float,
    rsi14: float,
    prev_close: float,
    ma20_prev: float,
    pullback_band: float,
    prob_buy_min: float,
    sell_prob_max: float,
    rsi_min: float,
) -> str:
    p = float(np.clip(probability, 0.0, 1.0))
    band = float(max(0.0, pullback_band))
    # (A) Pullback bounce: within ±band of MA20, up day
    pullback_bounce = (
        float(price) >= float(ma20) * (1.0 - band)
        and float(price) <= float(ma20) * (1.0 + band)
        and float(price) > float(prev_close)
    )
    # (B) MA20 reclaim: yesterday below MA20, today above (point-in-time MA20)
    reclaim_ok = (
        np.isfinite(ma20_prev)
        and float(prev_close) < float(ma20_prev)
        and float(price) > float(ma20)
    )
    entry_shape = pullback_bounce or reclaim_ok

    base_buy = (
        p > prob_buy_min
        and float(price) > float(ma50)
        and entry_shape
        and float(rsi14) > float(rsi_min)
    )
    if base_buy:
        return "BUY"
    if p < sell_prob_max:
        return "SELL"
    return "HOLD"


def rank_sector_candidates(
    candidates: pd.DataFrame,
    *,
    pullback_band: float = 0.02,
    prob_buy_min: float = 0.60,
    sell_prob_max: float = 0.40,
    rsi_min: float = 45.0,
) -> pd.DataFrame:
    req = {
        "Ticker",
        "Sector",
        "Price",
        "MA20",
        "MA50",
        "RSI14",
        "PrevClose",
        "MA20_prev",
        "Probability",
    }
    if not req.issubset(candidates.columns):
        raise ValueError(f"Missing required columns: {sorted(req - set(candidates.columns))}")
    out = candidates.copy()
    out = out[out["Price"] > out["MA50"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["MA20", "MA50", "RSI14", "PrevClose", "Probability"])
    out = out.sort_values(["Sector", "Probability"], ascending=[True, False]).reset_index(drop=True)
    out["Signal"] = out.apply(
        lambda r: probability_to_signal(
            float(r["Probability"]),
            float(r["Price"]),
            float(r["MA20"]),
            float(r["MA50"]),
            float(r["RSI14"]),
            float(r["PrevClose"]),
            float(r["MA20_prev"]) if pd.notna(r["MA20_prev"]) else float("nan"),
            float(pullback_band),
            float(prob_buy_min),
            float(sell_prob_max),
            float(rsi_min),
        ),
        axis=1,
    )
    out = out[out["Signal"] == "BUY"].copy()
    return out


def select_top_per_sector(ranked_candidates: pd.DataFrame) -> pd.DataFrame:
    if ranked_candidates.empty:
        return ranked_candidates.copy()
    return (
        ranked_candidates.sort_values(["Sector", "Probability"], ascending=[True, False])
        .groupby("Sector", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
