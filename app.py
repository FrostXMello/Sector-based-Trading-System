import streamlit as st
import time
import inspect

from backtest.backtest_engine import STRATEGY_SPECS, run_multi_strategy_backtest
from run_summary_multi import build_multi_strategy_summary_text


st.set_page_config(page_title="Multi-Strategy NIFTY50 Trading Platform", layout="wide")
st.title("Multi-Strategy NIFTY 50 Trading Platform")
st.caption("Intraday + short swing + mid swing + positional, with shared model and combined portfolio backtest.")

nav_options = ["Intraday", "Short Swing", "Mid Swing", "Positional", "Run All Strategies"]
nav = st.radio("Navigation", nav_options, horizontal=True)

name_to_key = {
    "Intraday": "intraday",
    "Short Swing": "short_swing",
    "Mid Swing": "mid_swing",
    "Positional": "positional",
}

default_selected = [name_to_key[nav]] if nav in name_to_key else list(STRATEGY_SPECS.keys())
selected = st.multiselect(
    "Select strategies to run",
    options=list(STRATEGY_SPECS.keys()),
    default=default_selected,
)

c1, c2 = st.columns(2)
with c1:
    years = st.slider("Historical window (years)", min_value=1, max_value=7, value=5, step=1)
with c2:
    capital = st.number_input("Initial capital", min_value=10000.0, value=100000.0, step=10000.0)
transaction_cost_bps = st.slider("Transaction cost (bps)", 0.0, 50.0, 10.0, 1.0)
min_daily_tickers = st.slider("Minimum daily tickers required", 5, 50, 10, 1)

with st.expander("Model training (walk-forward)", expanded=False):
    walk_forward = st.checkbox("Use walk-forward (rolling) training", value=True)
    min_train_days = st.slider("Min training days before signals", 30, 400, 80, 10)
    retrain_every_days = st.slider("Retrain frequency (trading days)", 1, 20, 10, 1)
    label_horizon_days = st.slider("Label horizon (days)", 3, 15, 5, 1)

st.subheader("Strategy parameters (editable in UI)")
tab_intraday, tab_short, tab_mid, tab_pos = st.tabs(["Intraday", "Short Swing", "Mid Swing", "Positional"])

with tab_intraday:
    c1, c2 = st.columns(2)
    with c1:
        intraday_prob_min = st.slider("Intraday prob min", 0.40, 0.90, 0.55, 0.01)
        intraday_volume_spike = st.slider("Intraday volume spike multiple", 1.0, 4.0, 1.5, 0.1)
    with c2:
        intraday_max_positions = st.slider("Intraday max positions", 1, 10, STRATEGY_SPECS["intraday"].max_positions, 1)
        intraday_max_trades = st.slider("Intraday max trades/day", 1, 20, STRATEGY_SPECS["intraday"].max_trades_per_day, 1)

with tab_short:
    c1, c2, c3 = st.columns(3)
    with c1:
        short_prob_min = st.slider("Short Swing prob min", 0.40, 0.90, 0.60, 0.01)
        short_rsi_min = st.slider("Short Swing RSI min", 40.0, 70.0, 50.0, 1.0)
        short_rsi_max = st.slider("Short Swing RSI max", 45.0, 80.0, 65.0, 1.0)
    with c2:
        short_stop = st.slider("Short Swing stop loss (%)", -5.0, -0.5, -1.5, 0.1) / 100.0
        short_target = st.slider("Short Swing target (%)", 0.5, 10.0, 4.0, 0.5) / 100.0
        short_time_exit = st.slider("Short Swing time exit (days)", 2, 10, 5, 1)
    with c3:
        short_max_positions = st.slider("Short Swing max positions", 1, 20, STRATEGY_SPECS["short_swing"].max_positions, 1)
        short_max_trades = st.slider("Short Swing max trades/day", 1, 20, STRATEGY_SPECS["short_swing"].max_trades_per_day, 1)

with tab_mid:
    c1, c2, c3 = st.columns(3)
    with c1:
        mid_prob_min = st.slider("Mid Swing prob min", 0.40, 0.90, 0.65, 0.01)
        mid_rsi_min = st.slider("Mid Swing RSI min", 30.0, 60.0, 45.0, 1.0)
        mid_pullback_band = st.slider("Mid Swing pullback band vs MA20 (±%)", 0.5, 8.0, 3.0, 0.5) / 100.0
    with c2:
        mid_stop = st.slider("Mid Swing stop loss (%)", -8.0, -0.5, -2.0, 0.1) / 100.0
        mid_partial_at = st.slider("Mid Swing partial profit at (%)", 0.5, 10.0, 2.5, 0.5) / 100.0
        mid_partial_fraction = st.slider("Mid Swing partial fraction", 0.10, 0.90, 0.50, 0.05)
        mid_time_exit = st.slider("Mid Swing time exit (days)", 5, 25, 15, 1)
        mid_ma20_exit_only_if_return_below = st.slider("MA20 exit only if return below (%)", -2.0, 5.0, 1.0, 0.5) / 100.0
    with c3:
        mid_max_positions = st.slider("Mid Swing max positions", 1, 30, STRATEGY_SPECS["mid_swing"].max_positions, 1)
        mid_max_trades = st.slider("Mid Swing max trades/day", 1, 20, STRATEGY_SPECS["mid_swing"].max_trades_per_day, 1)

with tab_pos:
    c1, c2, c3 = st.columns(3)
    with c1:
        pos_prob_min = st.slider("Positional prob min", 0.40, 0.95, 0.65, 0.01)
        pos_rsi_min = st.slider("Positional RSI min", 40.0, 70.0, 50.0, 1.0)
        pos_rsi_max = st.slider("Positional RSI max", 45.0, 80.0, 60.0, 1.0)
    with c2:
        pos_stop = st.slider("Positional stop loss (%)", -15.0, -1.0, -4.0, 0.5) / 100.0
        pos_time_exit = st.slider("Positional time exit (days)", 10, 120, 60, 5)
    with c3:
        pos_max_positions = st.slider("Positional max positions", 1, 30, STRATEGY_SPECS["positional"].max_positions, 1)
        pos_max_trades = st.slider("Positional max trades/day", 1, 20, STRATEGY_SPECS["positional"].max_trades_per_day, 1)

st.subheader("Capital allocation per strategy (weights)")
aw1, aw2, aw3, aw4 = st.columns(4)
with aw1:
    w_intraday = st.slider("Intraday weight (%)", 0.0, 100.0, 5.0, 1.0)
with aw2:
    w_short = st.slider("Short Swing weight (%)", 0.0, 100.0, 20.0, 1.0)
with aw3:
    w_mid = st.slider("Mid Swing weight (%)", 0.0, 100.0, 50.0, 1.0)
with aw4:
    w_pos = st.slider("Positional weight (%)", 0.0, 100.0, 25.0, 1.0)
st.caption("Note: weights are normalized across the strategies you actually select to run.")

if st.button("Run Backtest", type="primary"):
    try:
        status_box = st.empty()
        progress_bar = st.progress(0.0, text="Starting backtest...")
        t0 = time.time()

        def _progress_cb(p: float, msg: str) -> None:
            elapsed = max(0.0, time.time() - t0)
            # Simple ETA based on linear remaining time.
            if p > 0.02:
                eta = max(0.0, elapsed * (1.0 - p) / max(p, 1e-9))
                text = f"{msg} | {p * 100:.0f}% | elapsed {elapsed:.0f}s | ETA ~{eta:.0f}s"
            else:
                text = f"{msg} | {p * 100:.0f}% | elapsed {elapsed:.0f}s"
            progress_bar.progress(p, text=text)
            status_box.info(f"Current step: {msg}")

        with st.spinner("Running multi-strategy backtest..."):
            strategy_params = {
                "intraday": {
                    "prob_min": float(intraday_prob_min),
                    "volume_spike_mult": float(intraday_volume_spike),
                    "max_positions": int(intraday_max_positions),
                    "max_trades_per_day": int(intraday_max_trades),
                },
                "short_swing": {
                    "prob_min": float(short_prob_min),
                    "rsi_min": float(short_rsi_min),
                    "rsi_max": float(short_rsi_max),
                    "stop_loss": float(short_stop),
                    "target": float(short_target),
                    "time_exit_days": int(short_time_exit),
                    "max_positions": int(short_max_positions),
                    "max_trades_per_day": int(short_max_trades),
                },
                "mid_swing": {
                    "prob_min": float(mid_prob_min),
                    "rsi_min": float(mid_rsi_min),
                    "pullback_band": float(mid_pullback_band),
                    "stop_loss": float(mid_stop),
                    "partial_at": float(mid_partial_at),
                    "partial_fraction": float(mid_partial_fraction),
                    "time_exit_days": int(mid_time_exit),
                    "ma20_exit_only_if_return_below": float(mid_ma20_exit_only_if_return_below),
                    "max_positions": int(mid_max_positions),
                    "max_trades_per_day": int(mid_max_trades),
                },
                "positional": {
                    "prob_min": float(pos_prob_min),
                    "rsi_min": float(pos_rsi_min),
                    "rsi_max": float(pos_rsi_max),
                    "stop_loss": float(pos_stop),
                    "time_exit_days": int(pos_time_exit),
                    "max_positions": int(pos_max_positions),
                    "max_trades_per_day": int(pos_max_trades),
                },
            }
            capital_weights = {
                "intraday": float(w_intraday) / 100.0,
                "short_swing": float(w_short) / 100.0,
                "mid_swing": float(w_mid) / 100.0,
                "positional": float(w_pos) / 100.0,
            }
            call_kwargs = {
                "selected_strategies": selected,
                "initial_capital": float(capital),
                "years": int(years),
                "transaction_cost_bps": float(transaction_cost_bps),
                "progress_callback": _progress_cb,
                "min_daily_tickers": int(min_daily_tickers),
                "strategy_params": strategy_params,
                "capital_weights": capital_weights,
                "walk_forward": bool(walk_forward),
                "min_train_days": int(min_train_days),
                "retrain_every_days": int(retrain_every_days),
                "label_horizon_days": int(label_horizon_days),
            }
            # If Streamlit hasn't reloaded the latest module yet, avoid hard-crashing on new kwargs.
            sig = inspect.signature(run_multi_strategy_backtest)
            filtered = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
            result = run_multi_strategy_backtest(**filtered)

        progress_bar.empty()
        elapsed_sec = time.time() - t0
        status_box.success(f"Run complete in {elapsed_sec:.1f}s")

        m = result["metrics"]
        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Total Return", f"{m['Total Return'] * 100:.2f}%")
        mc2.metric("Sharpe", f"{m['Sharpe Ratio']:.2f}")
        mc3.metric("Max Drawdown", f"{m['Max Drawdown'] * 100:.2f}%")
        mc4.metric("Win Rate", f"{m['Win Rate'] * 100:.2f}%")
        mc5.metric("Trade Count", f"{m['Trade Count']}")

        st.subheader("Combined Equity Curve")
        st.line_chart(result["equity_curve"].set_index("Date")[["PortfolioValue"]])

        run_params = result.get("effective_run_params") or {
            "selected_strategies": selected,
            "years": int(years),
            "initial_capital": float(capital),
            "transaction_cost_bps": float(transaction_cost_bps),
        }
        summary_text = build_multi_strategy_summary_text(
            run_params=run_params,
            result=result,
            runtime_seconds=elapsed_sec,
        )
        st.subheader("Copy-paste summary (params + results)")
        st.caption("Select all (Ctrl+A) then copy (Ctrl+C) for notes.")
        st.text_area(
            "Backtest summary — copy/paste",
            value=summary_text,
            height=360,
            key="multi_strategy_summary_copy_area",
            label_visibility="collapsed",
        )

        st.subheader("Per-Strategy Summary")
        st.dataframe(result["strategy_summary"], use_container_width=True, hide_index=True)

        cov = result.get("data_coverage") or {}
        if cov:
            loaded = int(cov.get("daily_loaded", 0))
            expected = int(cov.get("daily_expected", 0))
            if expected > 0 and loaded < expected:
                st.warning(f"Data coverage: daily OHLCV loaded for {loaded}/{expected} NIFTY50 tickers (Yahoo gaps can reduce coverage).")
            with st.expander("Data coverage details"):
                st.write(cov)

        st.subheader("Active Positions")
        st.dataframe(result["active_positions"], use_container_width=True, hide_index=True)

        with st.expander("Trade Log"):
            st.dataframe(result["trade_log"], use_container_width=True, hide_index=True)
    except Exception as exc:
        st.error(f"Run failed: {exc}")

