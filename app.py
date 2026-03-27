from __future__ import annotations

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

from backtest import (
    BacktestConfig,
    PortfolioBacktestConfig,
    backtest_long_cash,
    backtest_portfolio_long_cash,
    plot_equity_curve,
)
from data_loader import DownloadConfig, download_stock_data
from feature_engineering import build_model_dataset
from model import ModelConfig, train_model
from strategy import attach_signals
from utils import NIFTY50_TICKERS, parse_tickers_text


st.set_page_config(page_title="Multi-Factor AI Financial Decision System", layout="wide")
st.title("Multi-Factor AI Financial Decision System")
st.caption("Predict next-day direction using technical features and backtest BUY/SELL/HOLD decisions.")


@st.cache_data(ttl=60 * 60)
def load_data_cached(ticker: str):
    # Keep download config explicit so cached results are stable.
    return download_stock_data(ticker, cfg=DownloadConfig(period="5y", interval="1d", auto_adjust=False))


col_left, col_right = st.columns([1, 1])
with col_left:
    universe_mode = st.selectbox("Universe Mode", options=["Single Ticker", "NIFTY 50 Portfolio"], index=0)

    if universe_mode == "Single Ticker":
        ticker = st.text_input("Stock ticker", value="RELIANCE.NS")
    else:
        st.write("Default NIFTY 50 tickers (Yahoo Finance). Edit if needed (comma/newline separated).")
        tickers_text = st.text_area("NIFTY 50 tickers", value=",".join(NIFTY50_TICKERS), height=160)
        max_tickers = st.slider("Max tickers to run (for speed)", 1, 50, 50, step=1)

    model_choice = st.selectbox("Model", options=["Logistic Regression", "Random Forest"], index=0)

    st.subheader("Thresholding (convert probability -> BUY/SELL/HOLD)")
    threshold_mode = st.selectbox("Threshold Mode", options=["Quantile", "Fixed"], index=0)

    if threshold_mode == "Fixed":
        buy_threshold = st.slider("BUY if Proba >", 0.5, 0.95, 0.6, step=0.01)
        sell_threshold = st.slider("SELL if Proba <", 0.05, 0.5, 0.4, step=0.01)
        buy_quantile = 0.7
        sell_quantile = 0.3
    else:
        buy_threshold = 0.6
        sell_threshold = 0.4
        buy_quantile = st.slider("BUY top quantile (>=)", 0.5, 0.95, 0.7, step=0.01)
        sell_quantile = st.slider("SELL bottom quantile (<=)", 0.05, 0.5, 0.3, step=0.01)

    initial_capital_total = st.number_input("Initial capital (portfolio or single)", value=100000.0, min_value=1000.0, step=1000.0)

    run = st.button("Run Model", type="primary")

with col_right:
    st.markdown("### Output")
    latest_signal_placeholder = st.empty()
    latest_proba_placeholder = st.empty()
    accuracy_placeholder = st.empty()
    backtest_placeholder = st.empty()
    signal_counts_placeholder = st.empty()
    proba_hist_placeholder = st.empty()
    portfolio_table_placeholder = st.empty()


def _run_single_pipeline(single_ticker: str):
    if not single_ticker or not single_ticker.strip():
        raise ValueError("Please enter a valid ticker.")

    model_type = "logreg" if model_choice == "Logistic Regression" else "rf"

    prices = load_data_cached(single_ticker.strip().upper())
    model_df = build_model_dataset(prices)

    result = train_model(model_df, cfg=ModelConfig(model_type=model_type))
    signals_df = attach_signals(
        result.test_df,
        result.test_proba,
        threshold_mode="fixed" if threshold_mode == "Fixed" else "quantile",
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        buy_quantile=buy_quantile,
        sell_quantile=sell_quantile,
    )

    backtest = backtest_long_cash(signals_df, cfg=BacktestConfig(initial_capital=float(initial_capital_total)))

    latest_row = signals_df.iloc[-1]
    latest = {
        "signal": str(latest_row["Signal"]),
        "proba": float(latest_row["Proba"]),
        "trade_date": latest_row["Date"],
    }

    counts = signals_df["Signal"].value_counts()
    return latest, float(result.test_accuracy), backtest, counts.to_dict(), result.test_proba


def _run_nifty50_pipeline():
    tickers = parse_tickers_text(tickers_text)
    tickers = tickers[:max_tickers]
    if not tickers:
        raise ValueError("No tickers provided.")

    model_type = "logreg" if model_choice == "Logistic Regression" else "rf"

    signals_by_ticker: dict[str, pd.DataFrame] = {}
    latest_rows: list[dict] = []

    # Note: this is intentionally sequential for clearer error reporting.
    for t in tickers:
        try:
            prices = load_data_cached(t)
            model_df = build_model_dataset(prices)
            result = train_model(model_df, cfg=ModelConfig(model_type=model_type))
            signals_df = attach_signals(
                result.test_df,
                result.test_proba,
                threshold_mode="fixed" if threshold_mode == "Fixed" else "quantile",
                buy_threshold=buy_threshold,
                sell_threshold=sell_threshold,
                buy_quantile=buy_quantile,
                sell_quantile=sell_quantile,
            )
            signals_by_ticker[t] = signals_df

            lr = signals_df.iloc[-1]
            latest_rows.append(
                {
                    "Ticker": t,
                    "Latest_Signal": str(lr["Signal"]),
                    "Latest_Proba": float(lr["Proba"]),
                    "Test_Accuracy": float(result.test_accuracy),
                }
            )
        except Exception as e:
            # Skip tickers that fail to download or cannot produce enough indicator data.
            st.warning(f"Skipping {t}: {e}")

    if not signals_by_ticker:
        raise RuntimeError("All tickers failed. Try editing the universe list or thresholds.")

    portfolio_backtest = backtest_portfolio_long_cash(
        signals_by_ticker,
        cfg=PortfolioBacktestConfig(initial_capital=float(initial_capital_total), allocation="equal"),
    )

    table_df = pd.DataFrame(latest_rows).sort_values("Ticker").reset_index(drop=True)
    return table_df, portfolio_backtest


if run:
    try:
        with st.spinner("Fetching data, training model, predicting signals, and backtesting..."):
            if universe_mode == "Single Ticker":
                latest, test_acc, backtest, counts, test_proba = _run_single_pipeline(ticker)
            else:
                table_df, portfolio_backtest = _run_nifty50_pipeline()

        if universe_mode == "Single Ticker":
            latest_signal_placeholder.metric("Latest Signal (for next day)", latest["signal"])
            latest_proba_placeholder.metric("Model Probability (P(next return > 0))", f"{latest['proba']:.3f}")
            accuracy_placeholder.info(f"Test Accuracy: {test_acc:.3f}")
            backtest_placeholder.success(
                f"Final Portfolio Value: {backtest['final_portfolio_value']:.2f} | Total Return: {backtest['total_return_pct']:.2f}%"
            )
            signal_counts_placeholder.write(f"Signal counts on test: {counts}")

            if counts.get("BUY", 0) == 0 or counts.get("SELL", 0) == 0:
                st.warning("You got almost no trading signals. Switch Threshold Mode to `Quantile` to force buys/sells.")

            # Probability histogram helps you understand why the equity can look flat.
            fig_hist = plt.figure(figsize=(8, 3.5))
            plt.hist(test_proba, bins=30, alpha=0.8, color="#1f77b4")
            plt.title("Predicted probability distribution (test period)")
            plt.xlabel("Proba(P(next return > 0))")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.25)
            proba_hist_placeholder.pyplot(fig_hist, use_container_width=True)

            st.markdown("### Portfolio Growth (Equity Curve)")
            fig = plot_equity_curve(backtest["equity_curve"], title="Equity Curve (Test Period)")
            st.pyplot(fig, use_container_width=True)
        else:
            latest_signal_placeholder.metric("NIFTY50 runs", f"{len(table_df)} tickers")
            latest_proba_placeholder.metric("Portfolio total return", f"{portfolio_backtest['total_return_pct']:.2f}%")

            accuracy_placeholder.info("Each ticker has its own test accuracy (see table).")
            backtest_placeholder.success(
                f"Final Portfolio Value: {portfolio_backtest['final_portfolio_value']:.2f} | Total Return: {portfolio_backtest['total_return_pct']:.2f}%"
            )

            portfolio_table_placeholder.dataframe(table_df, use_container_width=True, hide_index=True)

            st.markdown("### Portfolio Growth (Equity Curve) - Aggregated")
            fig = plot_equity_curve(portfolio_backtest["equity_curve"], title="NIFTY50 Aggregated Equity Curve (Test Period)")
            st.pyplot(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to run pipeline: {e}")

