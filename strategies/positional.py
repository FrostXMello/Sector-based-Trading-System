from __future__ import annotations

import pandas as pd


def positional_entry(
    row: pd.Series,
    prob: float,
    sector_top2_consecutive: bool,
    *,
    prob_min: float = 0.65,
    rsi_min: float = 50.0,
    rsi_max: float = 60.0,
) -> bool:
    trend_ok = float(row["Close"]) > float(row["MA50"]) > float(row["MA100"])
    rsi_ok = float(rsi_min) <= float(row["RSI14"]) <= float(rsi_max)
    breakout = float(row["Close"]) > float(row["High20"])
    return bool(sector_top2_consecutive and trend_ok and rsi_ok and breakout and float(prob) > float(prob_min))


def positional_exit(
    price: float,
    ma50: float,
    ret: float,
    hold_days: int,
    *,
    stop_loss: float = -0.04,
    time_exit_days: int = 60,
) -> bool:
    return ret <= float(stop_loss) or price < ma50 or hold_days >= int(time_exit_days)
