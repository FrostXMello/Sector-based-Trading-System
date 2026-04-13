from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from data_loader import DownloadConfig, download_stock_data
from feature_engineering import FEATURE_COLUMNS, build_features, build_labeled_dataset
from market_filter import compute_market_filter_frame, is_market_favorable
from model import ModelConfig, train_model
from risk_management import (
    allocation_from_probability,
    apply_transaction_cost,
    should_partial_book,
    stop_loss_triggered,
)
from sector_engine import compute_sector_scores
from sector_universe import STOCKS_BY_SECTOR
from strategy import rank_sector_candidates, select_top_per_sector


def _portfolio_stats(equity_curve: pd.DataFrame) -> dict:
    eq = equity_curve["PortfolioValue"].astype(float)
    ret = eq.pct_change().dropna()
    total_return = float(eq.iloc[-1] / eq.iloc[0] - 1.0) if len(eq) else 0.0
    sharpe = float(np.sqrt(252) * ret.mean() / ret.std()) if len(ret) > 1 and ret.std() > 0 else float("nan")
    dd = eq / eq.cummax() - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    win_rate = float((ret > 0).mean()) if len(ret) else float("nan")
    return {
        "Total Return": total_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Win Rate": win_rate,
    }


def _download_universe(period: str = "5y") -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    sector_prices: dict[str, pd.DataFrame] = {}
    stock_prices: dict[str, pd.DataFrame] = {}

    from sector_universe import SECTOR_INDEX_YAHOO

    for sector, idx_ticker in SECTOR_INDEX_YAHOO.items():
        try:
            sector_prices[sector] = download_stock_data(idx_ticker, DownloadConfig(period=period, interval="1d"))
        except Exception:
            # Fallback to shorter history when Yahoo times out on long lookbacks.
            try:
                sector_prices[sector] = download_stock_data(idx_ticker, DownloadConfig(period="3y", interval="1d"))
            except Exception:
                continue

    for tickers in STOCKS_BY_SECTOR.values():
        for ticker in tickers:
            if ticker in stock_prices:
                continue
            try:
                stock_prices[ticker] = download_stock_data(ticker, DownloadConfig(period=period, interval="1d"))
            except Exception:
                continue

    return sector_prices, stock_prices


def run_weekly_sector_backtest(
    *,
    period: str = "1y",
    top_sectors: int = 3,
    horizon_days: int = 5,
    model_type: str = "Random Forest",
    initial_capital: float = 100000.0,
    transaction_cost_bps: float = 10.0,
    max_positions: int = 3,
    max_new_trades_per_week: int = 2,
    prob_buy_min: float = 0.60,
    sell_prob_max: float = 0.40,
    pullback_band: float = 0.02,
    rsi_entry_min: float = 45.0,
    market_rsi_min: float = 50.0,
    use_market_rsi_filter: bool = False,
    entry_alloc_high: float = 0.40,
    entry_alloc_low: float = 0.30,
    entry_prob_high_alloc_cutoff: float = 0.75,
    exit_prob_min: float = 0.50,
    stop_loss_pct: float = -0.015,
    partial_take_profit_pct: float = 0.025,
    partial_book_fraction: float = 0.50,
    time_exit_days: int = 8,
    progress_callback: Callable[[float, str], None] | None = None,
) -> dict:
    def _progress(p: float, msg: str) -> None:
        if progress_callback is not None:
            progress_callback(float(max(0.0, min(1.0, p))), str(msg))

    _progress(0.02, "Downloading sector and stock universe...")
    sector_prices, stock_prices = _download_universe(period=period)
    if len(sector_prices) < 3:
        loaded = sorted(sector_prices.keys())
        raise RuntimeError(f"Insufficient sector index data. Loaded sectors: {loaded}")

    _progress(0.12, "Downloading NIFTY 50 and computing market regime filter...")
    nifty_df = download_stock_data("^NSEI", DownloadConfig(period=period, interval="1d"))
    nifty_df["Date"] = pd.to_datetime(nifty_df["Date"]).dt.normalize()
    market_frame = compute_market_filter_frame(
        nifty_df,
        use_rsi_filter=use_market_rsi_filter,
        rsi_min=market_rsi_min,
    )

    _progress(0.20, "Building stock feature cache...")
    stock_features: dict[str, pd.DataFrame] = {}
    for t, px in stock_prices.items():
        ff = build_features(px)
        ff["Date"] = pd.to_datetime(ff["Date"]).dt.normalize()
        stock_features[t] = ff

    all_dates = sorted(pd.to_datetime(nifty_df["Date"]).dt.normalize().unique())
    if len(all_dates) < 80:
        raise RuntimeError("Not enough dates for robust backtest.")

    _progress(0.28, "Initializing daily simulation engine...")
    cash = float(initial_capital)
    positions: dict[str, dict] = {}
    equity_rows: list[dict] = []
    rebalance_log: list[dict] = []
    latest_snapshot: dict = {}
    current_model = None
    this_week_trades = 0
    week_key = None
    market_on_rebalances = 0
    candidate_rows_seen = 0
    buy_signals_seen = 0
    entries_taken = 0

    bench_shares = float(initial_capital) / float(nifty_df.iloc[0]["Close"])
    benchmark_curve = pd.DataFrame({"Date": nifty_df["Date"], "PortfolioValue": bench_shares * nifty_df["Close"].astype(float)})

    def _asof_row(df: pd.DataFrame, d: pd.Timestamp):
        part = df[df["Date"] <= d]
        return None if part.empty else part.iloc[-1]

    total_days = len(all_dates)
    last_progress_bucket = -1
    for i_day, d in enumerate(all_dates):
        d = pd.Timestamp(d).normalize()
        sim_frac = (i_day + 1) / max(1, total_days)
        bucket = int(sim_frac * 20)
        if bucket != last_progress_bucket:
            last_progress_bucket = bucket
            _progress(0.28 + 0.64 * sim_frac, f"Running backtest simulation ({i_day + 1}/{total_days} days)...")
        wk = (int(d.year), int(d.isocalendar().week))
        if week_key != wk:
            week_key = wk
            this_week_trades = 0

        # Exit management (daily).
        to_close: list[str] = []
        for ticker, pos in positions.items():
            row = _asof_row(stock_features[ticker], d)
            if row is None:
                continue
            price = float(row["Close"])
            ma20 = float(row["MA20"])
            hold_days = int(pos["hold_days"])

            prob = float(pos["entry_prob"])
            if current_model is not None and all(c in row.index for c in FEATURE_COLUMNS):
                x = pd.DataFrame([{c: float(row[c]) for c in FEATURE_COLUMNS}])
                prob = float(current_model.predict_proba(x)[:, 1][0])

            if stop_loss_triggered(float(pos["entry_price"]), price, stop_loss=stop_loss_pct):
                to_close.append(ticker)
            elif (price < ma20) and ((price / float(pos["entry_price"]) - 1.0) < 0.01):
                to_close.append(ticker)
            elif prob < float(exit_prob_min):
                to_close.append(ticker)
            elif hold_days >= int(time_exit_days):
                to_close.append(ticker)
            elif should_partial_book(
                float(pos["entry_price"]),
                price,
                bool(pos["partial_booked"]),
                take_profit_pct=partial_take_profit_pct,
            ):
                sell_shares = float(pos["shares"]) * float(partial_book_fraction)
                proceeds = apply_transaction_cost(sell_shares * price, transaction_cost_bps)
                cash += proceeds
                pos["shares"] = float(pos["shares"]) - sell_shares
                pos["partial_booked"] = True
                positions[ticker] = pos

            pos["hold_days"] = hold_days + 1
            positions[ticker] = pos

        for ticker in to_close:
            if ticker not in positions:
                continue
            row = _asof_row(stock_features[ticker], d)
            if row is None:
                continue
            price = float(row["Close"])
            proceeds = apply_transaction_cost(float(positions[ticker]["shares"]) * price, transaction_cost_bps)
            cash += proceeds
            del positions[ticker]

        # Entry decisions evaluated daily when market regime is ON.
        rank_df = pd.DataFrame()
        selected = pd.DataFrame()
        top: list[str] = []
        market_on = is_market_favorable(market_frame, d)
        if market_on:
            market_on_rebalances += 1
            sector_hist = {}
            for s, sdf in sector_prices.items():
                part = sdf[pd.to_datetime(sdf["Date"]).dt.normalize() <= d]
                if len(part) >= 30:
                    sector_hist[s] = part
            rank_df, top = compute_sector_scores(sector_hist, top_n=top_sectors)

            if top:
                train_rows: list[pd.DataFrame] = []
                candidate_rows: list[dict] = []
                for sector in top:
                    for ticker in STOCKS_BY_SECTOR.get(sector, []):
                        px = stock_prices.get(ticker)
                        feat = stock_features.get(ticker)
                        if px is None or feat is None:
                            continue
                        hist = px[pd.to_datetime(px["Date"]).dt.normalize() <= d]
                        if len(hist) < 120:
                            continue
                        labeled = build_labeled_dataset(hist, ticker=ticker, horizon_days=horizon_days, threshold=0.01)
                        if len(labeled) < 100:
                            continue
                        train_rows.append(labeled)
                        row = _asof_row(feat, d)
                        if row is None:
                            continue
                        candidate_rows.append(
                            {
                                "Sector": sector,
                                "Ticker": ticker,
                                "Price": float(row["Close"]),
                                "MA20": float(row["MA20"]),
                                "MA50": float(row["MA50"]),
                                "RSI14": float(row["RSI14"]),
                                "PrevClose": float(row["PrevClose"]),
                                "MA20_prev": float(row["MA20_prev"]) if pd.notna(row["MA20_prev"]) else float("nan"),
                                **{c: float(row[c]) for c in FEATURE_COLUMNS},
                            }
                        )
                if train_rows and candidate_rows:
                    candidate_rows_seen += len(candidate_rows)
                    train_df = pd.concat(train_rows, ignore_index=True).sort_values("Date").reset_index(drop=True)
                    current_model, _test, _acc = train_model(
                        train_df,
                        FEATURE_COLUMNS,
                        ModelConfig(model_type=model_type, min_rows=100),
                    )
                    cand_df = pd.DataFrame(candidate_rows)
                    cand_df["Probability"] = current_model.predict_proba(cand_df[FEATURE_COLUMNS])[:, 1]
                    ranked = rank_sector_candidates(
                        cand_df,
                        pullback_band=pullback_band,
                        prob_buy_min=prob_buy_min,
                        sell_prob_max=sell_prob_max,
                        rsi_min=rsi_entry_min,
                    )
                    buy_signals_seen += int(len(ranked))
                    selected = select_top_per_sector(ranked).head(int(top_sectors)).copy()

                    for _, r in selected.iterrows():
                        if len(positions) >= max_positions or this_week_trades >= max_new_trades_per_week:
                            break
                        ticker = str(r["Ticker"])
                        if ticker in positions:
                            continue
                        price = float(r["Price"])
                        alloc = allocation_from_probability(
                            float(r["Probability"]),
                            high_prob_cutoff=entry_prob_high_alloc_cutoff,
                            alloc_high=entry_alloc_high,
                            alloc_low=entry_alloc_low,
                        )
                        budget = min(cash, float(initial_capital) * alloc)
                        if budget <= 0 or price <= 0:
                            continue
                        shares = budget / price
                        cost = shares * price
                        total_cost = cost * (1.0 + transaction_cost_bps / 10000.0)
                        if total_cost > cash:
                            continue
                        cash -= total_cost
                        positions[ticker] = {
                            "shares": shares,
                            "entry_price": price,
                            "entry_prob": float(r["Probability"]),
                            "entry_date": d,
                            "hold_days": 0,
                            "partial_booked": False,
                        }
                        this_week_trades += 1
                        entries_taken += 1

        # Log every calendar day in the NIFTY series so OFF regimes and gaps are visible.
        rebalance_log.append(
            {
                "Date": d,
                "Market_ON": market_on,
                "TopSectors": ", ".join(top),
                "SelectedStocks": ", ".join(selected["Ticker"].tolist()) if len(selected) else "",
                "OpenPositions": ", ".join(sorted(positions.keys())),
                "NewTradesThisWeek": this_week_trades,
            }
        )

        last_nifty = _asof_row(nifty_df, d)
        nifty_close_str = float(last_nifty["Close"]) if last_nifty is not None else float("nan")
        latest_snapshot = {
            "market_on": market_on,
            "sector_ranking": rank_df,
            "top_sectors": top,
            "selected_stocks": selected[["Sector", "Ticker", "Probability", "Signal"]] if len(selected) else pd.DataFrame(),
            "allocation": pd.DataFrame(
                [{"Ticker": k, "Weight": (positions[k]["shares"] * float(_asof_row(stock_features[k], d)["Close"])) / max(1e-9, cash + sum(positions[t]["shares"] * float(_asof_row(stock_features[t], d)["Close"]) for t in positions))} for k in positions]
            ),
            "diagnostics": pd.DataFrame(
                [
                    {"Metric": "Price data source", "Value": "yfinance live download (no local OHLCV cache)"},
                    {"Metric": "Simulation last NIFTY close", "Value": nifty_close_str},
                    {"Metric": "Market-ON days (evaluated)", "Value": market_on_rebalances},
                    {"Metric": "Candidates Evaluated", "Value": candidate_rows_seen},
                    {"Metric": "BUY Signals After Filters", "Value": buy_signals_seen},
                    {"Metric": "Entries Taken", "Value": entries_taken},
                ]
            ),
            "settings": pd.DataFrame(
                [
                    {"Parameter": "transaction_cost_bps", "Value": transaction_cost_bps},
                    {"Parameter": "max_positions", "Value": max_positions},
                    {"Parameter": "max_new_trades_per_week", "Value": max_new_trades_per_week},
                    {"Parameter": "prob_buy_min", "Value": prob_buy_min},
                    {"Parameter": "pullback_band", "Value": pullback_band},
                    {"Parameter": "sell_prob_max", "Value": sell_prob_max},
                    {"Parameter": "rsi_entry_min", "Value": rsi_entry_min},
                    {"Parameter": "use_market_rsi_filter", "Value": use_market_rsi_filter},
                    {"Parameter": "market_rsi_min", "Value": market_rsi_min},
                    {"Parameter": "exit_prob_min", "Value": exit_prob_min},
                    {"Parameter": "stop_loss_pct", "Value": stop_loss_pct},
                    {"Parameter": "partial_take_profit_pct", "Value": partial_take_profit_pct},
                    {"Parameter": "partial_book_fraction", "Value": partial_book_fraction},
                    {"Parameter": "time_exit_days", "Value": time_exit_days},
                ]
            ),
        }

        portfolio_value = cash
        for ticker, pos in positions.items():
            row = _asof_row(stock_features[ticker], d)
            if row is None:
                continue
            portfolio_value += float(pos["shares"]) * float(row["Close"])
        equity_rows.append({"Date": d, "PortfolioValue": portfolio_value})

    equity_curve = pd.DataFrame(equity_rows)
    if equity_curve.empty:
        raise RuntimeError("Backtest produced no portfolio points.")
    _progress(0.95, "Computing metrics and packaging results...")
    bench = benchmark_curve.set_index("Date").reindex(equity_curve["Date"]).ffill().reset_index()
    metrics = _portfolio_stats(equity_curve)
    _progress(1.0, "Backtest complete.")
    return {
        "equity_curve": equity_curve,
        "benchmark_curve": bench,
        "metrics": metrics,
        "latest_snapshot": latest_snapshot,
        "rebalance_log": pd.DataFrame(rebalance_log),
    }

