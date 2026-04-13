from __future__ import annotations

import pandas as pd


def mid_swing_entry(
    row: pd.Series,
    prob: float,
    sector_ok: bool,
    *,
    pullback_band: float = 0.03,
    prob_min: float = 0.65,
    rsi_min: float = 45.0,
) -> bool:
    price = float(row["Close"])
    ma20 = float(row["MA20"])
    ma20_prev = float(row["MA20_prev"])
    ma20_prev2 = float(row["MA20_prev2"])
    ma50 = float(row["MA50"])
    ret3 = float(row["Return_3d"])
    rsi = float(row["RSI14"])
    prev_close = float(row["PrevClose"])
    prev_close2 = float(row["PrevClose2"])
    prev5_high = float(row["Prev5High"])
    rs_5 = float(row["RS_5"])
    rs_10 = float(row["RS_10"])

    near_today = abs(price / ma20 - 1.0) <= float(pullback_band)
    near_prev1 = abs(prev_close / ma20_prev - 1.0) <= float(pullback_band)
    near_prev2 = abs(prev_close2 / ma20_prev2 - 1.0) <= float(pullback_band)
    near_ma20_recent = near_today or near_prev1 or near_prev2
    breakout_continuation = price > prev5_high
    relative_strength_ok = (rs_5 > 0.0) and (rs_10 > 0.0)

    return bool(
        sector_ok
        and (price > ma20 > ma50)
        and (ret3 > 0.0)
        and near_ma20_recent
        and breakout_continuation
        and (rsi > float(rsi_min))
        and (float(prob) > float(prob_min))
        and relative_strength_ok
    )


def mid_swing_exit(
    price: float,
    ma20: float,
    ret: float,
    hold_days: int,
    *,
    stop_loss: float = -0.02,
    time_exit_days: int = 15,
    ma20_exit_only_if_return_below: float = 0.03,
) -> bool:
    ma20_exit = (price < ma20) and (ret < float(ma20_exit_only_if_return_below))
    return bool(ret <= float(stop_loss) or ma20_exit or hold_days >= int(time_exit_days))
