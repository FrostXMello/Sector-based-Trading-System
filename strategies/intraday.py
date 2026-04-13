from __future__ import annotations

import pandas as pd


def intraday_entry_signal(
    intraday_df: pd.DataFrame,
    trade_date: pd.Timestamp,
    *,
    volume_spike_mult: float = 1.5,
) -> bool:
    if intraday_df is None or intraday_df.empty:
        return False
    df = intraday_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    if len(df) < 30:
        return False
    d0 = pd.Timestamp(trade_date).normalize()
    day = df[df["Date"].dt.normalize() == d0].copy()
    if day.empty:
        return False

    first_15 = day[(day["Date"].dt.hour == 9) & (day["Date"].dt.minute <= 29)]
    trade_window = day[(day["Date"].dt.hour > 9) | ((day["Date"].dt.hour == 9) & (day["Date"].dt.minute >= 30))]
    trade_window = trade_window[(trade_window["Date"].dt.hour < 11) | ((trade_window["Date"].dt.hour == 11) & (trade_window["Date"].dt.minute <= 30))]
    if first_15.empty or trade_window.empty:
        return False

    orb_high = float(first_15["High"].max())
    vol_mean = float(day["Volume"].rolling(20, min_periods=5).mean().iloc[-1]) if "Volume" in day.columns else 0.0
    last = trade_window.iloc[-1]
    vwap = (day["Close"] * day["Volume"]).cumsum() / day["Volume"].replace(0, pd.NA).cumsum() if "Volume" in day.columns else day["Close"]
    vwap_last = float(vwap.fillna(method="ffill").iloc[-1])
    vol_ok = ("Volume" in day.columns) and (float(last["Volume"]) > volume_spike_mult * max(vol_mean, 1.0))
    return float(last["Close"]) > orb_high and float(last["Close"]) > vwap_last and vol_ok
