from __future__ import annotations

import pandas as pd


def short_swing_entry(
    row: pd.Series,
    prob: float,
    sector_ok: bool,
    *,
    prob_min: float = 0.60,
    rsi_min: float = 50.0,
    rsi_max: float = 65.0,
) -> bool:
    breakout = float(row["Close"]) > float(row["Prev3High"])
    trend_ok = float(row["Close"]) > float(row["MA20"]) > float(row["MA50"])
    rsi_ok = float(rsi_min) <= float(row["RSI14"]) <= float(rsi_max)
    return bool(sector_ok and trend_ok and rsi_ok and breakout and float(prob) > float(prob_min))


def short_swing_exit(ret: float, hold_days: int, *, stop_loss: float = -0.015, target: float = 0.04, time_exit_days: int = 5) -> bool:
    return ret <= float(stop_loss) or ret >= float(target) or hold_days >= int(time_exit_days)
