from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 100000.0


def backtest_long_cash(signals_df: pd.DataFrame, cfg: BacktestConfig | None = None) -> dict:
    """
    Simple backtest:
    - Start with cash only.
    - If signal==BUY -> invest all cash into shares at `Close` of the trade day.
    - If signal==SELL -> sell all shares at `Close` of the trade day (go back to cash).
    - HOLD -> keep current position.

    Equity is marked-to-market daily using `Next_Close` (end of the predicted next day).
    """
    if cfg is None:
        cfg = BacktestConfig()

    required = ["Date", "Close", "Next_Date", "Next_Close", "Signal"]
    missing = [c for c in required if c not in signals_df.columns]
    if missing:
        raise ValueError(f"signals_df missing columns: {missing}")

    data = signals_df.sort_values("Date").reset_index(drop=True).copy()

    cash = float(cfg.initial_capital)
    shares = 0.0

    equity_rows = []
    for _, row in data.iterrows():
        trade_price = float(row["Close"])
        next_price = float(row["Next_Close"])
        signal = str(row["Signal"]).upper().strip()
        next_date = row["Next_Date"]

        # Execute at trade-date close; equity shown at next day's close.
        if signal == "BUY" and shares == 0.0:
            shares = cash / trade_price
            cash = 0.0
        elif signal == "SELL" and shares != 0.0:
            cash = shares * trade_price
            shares = 0.0

        portfolio_value_next = cash + shares * next_price
        equity_rows.append({"Date": next_date, "PortfolioValue": portfolio_value_next})

    equity_curve = pd.DataFrame(equity_rows)
    final_value = float(equity_curve["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / cfg.initial_capital - 1.0) * 100.0

    return {
        "equity_curve": equity_curve,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
    }


def plot_equity_curve(equity_curve: pd.DataFrame, *, title: str = "Equity Curve") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_datetime(equity_curve["Date"])
    y = equity_curve["PortfolioValue"].astype(float)
    ax.plot(x, y, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


@dataclass(frozen=True)
class PortfolioBacktestConfig:
    """
    Portfolio backtest configuration for a basket of independent single-stock strategies.

    For this demo we:
    - allocate capital equally per ticker
    - backtest each ticker with the same long/cash logic
    - align equity curves by date and sum them
    """

    initial_capital: float = 100000.0
    allocation: str = "equal"  # only "equal" supported in this simple demo


def backtest_portfolio_long_cash(
    signals_by_ticker: Dict[str, pd.DataFrame],
    cfg: PortfolioBacktestConfig | None = None,
) -> dict:
    """
    Backtest a basket of tickers by running `backtest_long_cash` per ticker and summing.

    Returns:
      - equity_curve: DataFrame with Date + total PortfolioValue
      - final_portfolio_value
      - total_return_pct
      - per_ticker_final: dict[ticker] = final_value
    """
    if cfg is None:
        cfg = PortfolioBacktestConfig()

    tickers = [t for t in signals_by_ticker.keys() if signals_by_ticker[t] is not None]
    if not tickers:
        raise ValueError("signals_by_ticker is empty")

    if cfg.allocation.lower() != "equal":
        raise ValueError("Only allocation='equal' is supported")

    n = len(tickers)
    initial_cap_per = float(cfg.initial_capital) / n

    equity_by_ticker: dict[str, pd.Series] = {}
    per_ticker_final: dict[str, float] = {}
    start_dates = []

    for ticker in tickers:
        bt = backtest_long_cash(signals_by_ticker[ticker], cfg=BacktestConfig(initial_capital=initial_cap_per))
        eq = bt["equity_curve"].copy()
        series = eq.set_index("Date")["PortfolioValue"].astype(float).sort_index()
        equity_by_ticker[ticker] = series
        per_ticker_final[ticker] = float(series.iloc[-1])
        start_dates.append(series.index.min())

    global_dates = pd.date_range(start=min(start_dates), end=max(s.index.max() for s in equity_by_ticker.values()), freq="D")

    # Sum all tickers' equity curves after aligning on a shared calendar.
    # Any missing days are treated as "forward-filled positions" and prior to a
    # ticker's first point we assume it remains in cash (initial_cap_per).
    total_values = np.zeros(len(global_dates), dtype=float)
    for ticker, series in equity_by_ticker.items():
        aligned = series.reindex(global_dates).ffill()
        aligned = aligned.fillna(initial_cap_per)
        total_values += aligned.values

    equity_curve = pd.DataFrame({"Date": global_dates, "PortfolioValue": total_values})
    final_value = float(equity_curve["PortfolioValue"].iloc[-1])
    total_return_pct = (final_value / float(cfg.initial_capital) - 1.0) * 100.0

    return {
        "equity_curve": equity_curve,
        "final_portfolio_value": final_value,
        "total_return_pct": total_return_pct,
        "per_ticker_final": per_ticker_final,
    }

