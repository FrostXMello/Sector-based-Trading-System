from __future__ import annotations

from dataclasses import dataclass

from typing import Callable

import time

import numpy as np
import pandas as pd

from data_engine import DataEngineConfig, NIFTY50_TICKERS, load_multi_timeframe_universe
from data_loader import DownloadConfig, download_stock_data
from execution.order_engine import execute_buy, execute_sell
from execution.position_manager import Position, mark_to_market
from execution.risk_engine import cap_position_value, exposure_limit_value
from feature_engineering import FEATURE_COLUMNS, add_relative_strength, build_features, build_labeled_dataset
from model import ModelConfig, fit_model_full
from sector_universe import STOCKS_BY_SECTOR
from strategies.intraday import intraday_entry_signal
from strategies.mid_swing import mid_swing_entry, mid_swing_exit
from strategies.positional import positional_entry, positional_exit
from strategies.short_swing import short_swing_entry, short_swing_exit


@dataclass(frozen=True)
class StrategySpec:
    key: str
    capital_weight: float
    max_positions: int
    max_trades_per_day: int


STRATEGY_SPECS: dict[str, StrategySpec] = {
    "intraday": StrategySpec("intraday", 0.05, 3, 5),
    "short_swing": StrategySpec("short_swing", 0.20, 6, 3),
    "mid_swing": StrategySpec("mid_swing", 0.50, 8, 3),
    "positional": StrategySpec("positional", 0.25, 6, 2),
}


def _ticker_sector_map() -> dict[str, str]:
    out: dict[str, str] = {}
    for s, names in STOCKS_BY_SECTOR.items():
        for t in names:
            out[t] = s
    return out


def _sector_rank_from_daily(
    daily_features: dict[str, pd.DataFrame],
    date: pd.Timestamp,
    *,
    top_n: int = 3,
) -> list[str]:
    t2s = _ticker_sector_map()
    rows: list[dict] = []
    for t, f in daily_features.items():
        s = t2s.get(t)
        if s is None:
            continue
        part = f[f["Date"] <= date]
        if len(part) < 12:
            continue
        close = part["Close"].astype(float)
        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
        ret_10d = float(close.iloc[-1] / close.iloc[-11] - 1.0) if len(close) >= 11 else 0.0
        rows.append({"Sector": s, "ret_5d": ret_5d, "ret_10d": ret_10d})
    if not rows:
        return []
    df = pd.DataFrame(rows).groupby("Sector", as_index=False).mean()
    df["score"] = 0.6 * df["ret_5d"] + 0.4 * df["ret_10d"]
    return df.sort_values("score", ascending=False)["Sector"].head(top_n).tolist()


def _sector_momentum_snapshot(
    daily_features: dict[str, pd.DataFrame],
    date: pd.Timestamp,
) -> pd.DataFrame:
    t2s = _ticker_sector_map()
    rows: list[dict] = []
    for t, f in daily_features.items():
        s = t2s.get(t)
        if s is None:
            continue
        part = f[f["Date"] <= date]
        if len(part) < 12:
            continue
        close = part["Close"].astype(float)
        ret_5d = float(close.iloc[-1] / close.iloc[-6] - 1.0) if len(close) >= 6 else 0.0
        ret_10d = float(close.iloc[-1] / close.iloc[-11] - 1.0) if len(close) >= 11 else 0.0
        rows.append({"Sector": s, "ret_5d": ret_5d, "ret_10d": ret_10d})
    if not rows:
        return pd.DataFrame(columns=["Sector", "ret_5d", "ret_10d"])
    return pd.DataFrame(rows).groupby("Sector", as_index=False).mean()


def run_multi_strategy_backtest(
    *,
    selected_strategies: list[str],
    initial_capital: float = 100000.0,
    years: int = 5,
    transaction_cost_bps: float = 10.0,
    progress_callback: Callable[[float, str], None] | None = None,
    min_daily_tickers: int = 10,
    strategy_params: dict[str, dict] | None = None,
    capital_weights: dict[str, float] | None = None,
    walk_forward: bool = True,
    min_train_days: int = 120,
    retrain_every_days: int = 10,
    label_horizon_days: int = 5,
    train_max_days: int | None = None,
    retrain_min_new_rows: int = 0,
) -> dict:
    def _progress(p: float, msg: str) -> None:
        if progress_callback is None:
            return
        p2 = float(max(0.0, min(1.0, p)))
        progress_callback(p2, str(msg))

    _progress(0.01, "Preparing run...")
    selected = [s for s in selected_strategies if s in STRATEGY_SPECS]
    if not selected:
        raise ValueError("Please select at least one strategy.")

    t0 = time.time()
    strategy_params = strategy_params or {}

    # Capital weights: allow UI override, normalize over selected.
    weights: dict[str, float] = {}
    if capital_weights:
        for k in selected:
            weights[k] = float(capital_weights.get(k, STRATEGY_SPECS[k].capital_weight))
    else:
        for k in selected:
            weights[k] = float(STRATEGY_SPECS[k].capital_weight)
    w_sum = sum(max(0.0, v) for v in weights.values())
    if w_sum <= 0:
        weights = {k: 1.0 / len(selected) for k in selected}
    else:
        weights = {k: max(0.0, v) / w_sum for k, v in weights.items()}

    _progress(0.05, "Downloading multi-timeframe universe (stocks)...")
    need_intraday = "intraday" in selected
    intraday_intervals = ("1m", "5m") if need_intraday else tuple()
    raw = load_multi_timeframe_universe(
        tickers=NIFTY50_TICKERS,
        cfg=DataEngineConfig(years=years, intraday_intervals=intraday_intervals, include_daily=True),
    )
    daily_prices = {t: frames["1d"] for t, frames in raw.items() if "1d" in frames}
    min_daily_tickers = int(max(5, min(50, int(min_daily_tickers))))
    missing_daily = [t for t in NIFTY50_TICKERS if t not in daily_prices]
    if len(daily_prices) < min_daily_tickers:
        raise RuntimeError(
            f"Insufficient daily data for NIFTY 50 universe: loaded {len(daily_prices)}/{len(NIFTY50_TICKERS)} tickers "
            f"(<{min_daily_tickers}). This is usually a Yahoo/yfinance availability issue. "
            f"Missing examples: {missing_daily[:8]}"
        )

    _progress(0.15, "Computing daily features...")
    daily_features = {t: build_features(df) for t, df in daily_prices.items()}

    _progress(0.20, "Downloading NIFTY calendar (^NSEI)...")
    nifty_df = download_stock_data("^NSEI", DownloadConfig(period=f"{max(1, int(years))}y", interval="1d"))
    nifty_df["Date"] = pd.to_datetime(nifty_df["Date"]).dt.normalize()
    _progress(0.24, "Computing relative strength vs NIFTY...")
    daily_features = {t: add_relative_strength(fdf, nifty_df) for t, fdf in daily_features.items()}
    nifty_feat = build_features(nifty_df)
    nifty_feat["Date"] = pd.to_datetime(nifty_feat["Date"]).dt.normalize()
    all_dates = sorted(pd.Series(nifty_df["Date"].unique()).dropna().tolist())
    expected_days = 252 * max(1, int(years))
    # Yahoo occasionally returns a slightly shorter window (e.g. 1y can be ~247).
    min_required_days = max(200, int(expected_days * 0.8))
    if len(all_dates) < min_required_days:
        raise RuntimeError(
            f"Not enough NIFTY calendar dates for robust simulation: {len(all_dates)} (<{min_required_days}). "
            "Try a larger 'History (years)' if this keeps happening (Yahoo data may be partial)."
        )

    # Shared model training dataset (labeled) across universe using daily data.
    _progress(0.28, "Building labeled training dataset for shared model...")
    train_frames: list[pd.DataFrame] = []
    for t, df in daily_prices.items():
        try:
            train_frames.append(
                build_labeled_dataset(df, ticker=t, horizon_days=int(label_horizon_days), threshold=0.01)
            )
        except Exception:
            continue
    if not train_frames:
        raise RuntimeError("Training dataset is empty after labeling. Universe downloads may have failed.")
    train_df = pd.concat(train_frames, ignore_index=True).sort_values("Date").reset_index(drop=True)
    train_df["Date"] = pd.to_datetime(train_df["Date"]).dt.normalize()
    train_dates_np = train_df["Date"].to_numpy(dtype="datetime64[ns]")

    _progress(0.55, "Initializing walk-forward model training...")
    model_cfg = ModelConfig(model_type="Random Forest", min_rows=200)
    current_model = None
    last_train_i: int | None = None
    last_train_end_idx = -1
    min_train_days = int(max(30, int(min_train_days)))
    retrain_every_days = int(max(1, int(retrain_every_days)))
    label_horizon_days = int(max(1, int(label_horizon_days)))
    retrain_min_new_rows = int(max(0, int(retrain_min_new_rows)))
    if train_max_days is not None:
        # Safety cap only; keep it large to preserve longer-term learning.
        train_max_days = int(max(300, int(train_max_days)))

    # Strategy-aware minimum bars required before evaluating entries.
    # Avoid a blunt 120-day global gate that delays otherwise-valid strategies.
    min_history_bars = 60
    if "positional" in selected:
        min_history_bars = max(min_history_bars, 110)  # MA100 + stabilization
    if "mid_swing" in selected:
        min_history_bars = max(min_history_bars, 55)   # MA50/MA20 plus pullback context
    if "short_swing" in selected:
        min_history_bars = max(min_history_bars, 55)   # MA50 + breakout lookback

    strategy_cash = {k: float(initial_capital) * float(weights[k]) for k in selected}
    strategy_positions: dict[str, dict[str, Position]] = {k: {} for k in selected}
    strategy_pnl = {k: 0.0 for k in selected}
    trade_count = {k: 0 for k in selected}
    wins = {k: 0 for k in selected}
    daily_rows: list[dict] = []
    logs: list[dict] = []
    t2s = _ticker_sector_map()
    top2_prev: list[str] = []

    total_days = len(all_dates)
    last_log_bucket = -1

    for i_day, d in enumerate(all_dates):
        if total_days > 0:
            p = 0.60 + 0.39 * (i_day / total_days)
        else:
            p = 0.98
        bucket = int((i_day / max(1, total_days)) * 20)
        if bucket != last_log_bucket:
            last_log_bucket = bucket
            _progress(p, f"Simulating calendar day {i_day + 1}/{total_days}...")

        date = pd.Timestamp(d).normalize()
        top3 = _sector_rank_from_daily(daily_features, date, top_n=3)
        top2 = top3[:2]
        sec_momo = _sector_momentum_snapshot(daily_features, date)
        sec_momo_map = {
            str(r["Sector"]): (float(r["ret_5d"]), float(r["ret_10d"]))
            for _, r in sec_momo.iterrows()
        }
        nifty_part = nifty_feat[nifty_feat["Date"] <= date]
        market_trend = False
        if not nifty_part.empty:
            nr = nifty_part.iloc[-1]
            if pd.notna(nr.get("MA100")) and pd.notna(nr.get("RSI14")):
                market_trend = bool(float(nr["Close"]) > float(nr["MA100"]) and float(nr["RSI14"]) > 55.0)
        prices_today: dict[str, float] = {}
        trades_today = {k: 0 for k in selected}

        # Walk-forward (rolling) retrain once per date (no lookahead):
        # train only on rows with labels fully known as-of this date, i.e. <= (date - label_horizon_days).
        if walk_forward:
            if i_day >= (min_train_days + label_horizon_days):
                cutoff_date = pd.Timestamp(all_dates[i_day - label_horizon_days]).normalize()
                retrain_due = (last_train_i is None) or ((i_day - last_train_i) >= retrain_every_days)
                if retrain_due:
                    _progress(0.56, f"Walk-forward retrain @ {str(date)[:10]} (cutoff {str(cutoff_date)[:10]})...")
                    cutoff_np = np.datetime64(cutoff_date, "ns")
                    end_idx = int(np.searchsorted(train_dates_np, cutoff_np, side="right"))
                    if end_idx >= model_cfg.min_rows:
                        # Expanding window by default: all history up to cutoff.
                        start_idx = 0
                        # Optional large cap for memory/runtime safety.
                        if train_max_days is not None:
                            window_start = pd.Timestamp(cutoff_date) - pd.Timedelta(days=int(train_max_days))
                            ws_np = np.datetime64(window_start.normalize(), "ns")
                            start_idx = int(np.searchsorted(train_dates_np, ws_np, side="left"))
                            if (end_idx - start_idx) < model_cfg.min_rows:
                                start_idx = max(0, end_idx - model_cfg.min_rows)
                        new_rows_since_last = end_idx - max(0, last_train_end_idx)
                        if (last_train_end_idx < 0) or (new_rows_since_last > retrain_min_new_rows):
                            train_slice = train_df.iloc[start_idx:end_idx]
                            current_model = fit_model_full(train_slice, FEATURE_COLUMNS, model_cfg)
                            last_train_i = i_day
                            last_train_end_idx = end_idx
            else:
                current_model = None
        else:
            if current_model is None:
                _progress(0.56, "Training single model on full history (no walk-forward)...")
                current_model = fit_model_full(train_df, FEATURE_COLUMNS, model_cfg)

        for t, f in daily_features.items():
            part = f[f["Date"] <= date]
            if part.empty:
                continue
            row = part.iloc[-1]
            prices_today[t] = float(row["Close"])

        # Exits first for each strategy
        for sk in selected:
            to_close: list[str] = []
            for t, p in strategy_positions[sk].items():
                px = prices_today.get(t, p.entry_price)
                f = daily_features[t]
                row = f[f["Date"] <= date].iloc[-1]
                ret = px / p.entry_price - 1.0
                p.hold_days += 1
                close_now = False
                exit_reason = ""
                if sk == "short_swing":
                    ss = strategy_params.get("short_swing", {})
                    if ret <= float(ss.get("stop_loss", -0.015)):
                        exit_reason = "stop_loss"
                    elif ret >= float(ss.get("target", 0.04)):
                        exit_reason = "target"
                    elif p.hold_days >= int(ss.get("time_exit_days", 5)):
                        exit_reason = "time_exit"
                    close_now = short_swing_exit(
                        ret,
                        p.hold_days,
                        stop_loss=float(ss.get("stop_loss", -0.015)),
                        target=float(ss.get("target", 0.04)),
                        time_exit_days=int(ss.get("time_exit_days", 5)),
                    )
                elif sk == "mid_swing":
                    ms = strategy_params.get("mid_swing", {})
                    if ret <= float(ms.get("stop_loss", -0.02)):
                        exit_reason = "stop_loss"
                    elif (px < float(row["MA20"])) and (ret < float(ms.get("ma20_exit_only_if_return_below", 0.03))):
                        exit_reason = "ma20_exit"
                    elif p.hold_days >= int(ms.get("time_exit_days", 15)):
                        exit_reason = "time_exit"
                    # Profit-protected trailing stop activates after +3% return.
                    trail_activate_ret = float(ms.get("trail_activate_ret", 0.03))
                    trail_pct = float(ms.get("trail_stop_pct", 0.03))
                    trail_stop = getattr(p, "trail_stop", None)
                    if ret >= trail_activate_ret:
                        new_trail = float(px) * (1.0 - trail_pct)
                        if trail_stop is None:
                            trail_stop = new_trail
                        else:
                            trail_stop = max(float(trail_stop), new_trail)
                        p.trail_stop = float(trail_stop)
                    if (trail_stop is not None) and (float(px) < float(trail_stop)):
                        close_now = True
                        exit_reason = "trailing_stop"
                    else:
                        close_now = mid_swing_exit(
                            px,
                            float(row["MA20"]),
                            ret,
                            p.hold_days,
                            stop_loss=float(ms.get("stop_loss", -0.02)),
                            time_exit_days=int(ms.get("time_exit_days", 15)),
                            ma20_exit_only_if_return_below=float(ms.get("ma20_exit_only_if_return_below", 0.03)),
                        )
                elif sk == "positional":
                    ps = strategy_params.get("positional", {})
                    if ret <= float(ps.get("stop_loss", -0.04)):
                        exit_reason = "stop_loss"
                    elif px < float(row["MA50"]):
                        exit_reason = "ma50_exit"
                    elif p.hold_days >= int(ps.get("time_exit_days", 60)):
                        exit_reason = "time_exit"
                    close_now = positional_exit(
                        px,
                        float(row["MA50"]),
                        ret,
                        p.hold_days,
                        stop_loss=float(ps.get("stop_loss", -0.04)),
                        time_exit_days=int(ps.get("time_exit_days", 60)),
                    )
                elif sk == "intraday":
                    close_now = True  # force day end exit for intraday book
                    exit_reason = "force_exit"
                if close_now:
                    p.exit_reason = exit_reason
                    to_close.append(t)
            for t in to_close:
                p = strategy_positions[sk][t]
                px = prices_today.get(t, p.entry_price)
                strategy_cash[sk], proceeds, exec_px = execute_sell(strategy_cash[sk], px, p.shares, transaction_cost_bps)
                strategy_pnl[sk] += proceeds - (p.shares * p.entry_price)
                if exec_px > p.entry_price:
                    wins[sk] += 1
                trade_count[sk] += 1
                logs.append(
                    {
                        "Date": date,
                        "Symbol": t,
                        "Strategy": sk,
                        "Side": "SELL",
                        "Action": "EXIT",
                        "exit_reason": str(getattr(p, "exit_reason", "")),
                        "shares": float(p.shares),
                        "entry_price": float(p.entry_price),
                        "exit_price": float(exec_px),
                        "pnl": float((exec_px - p.entry_price) * p.shares),
                        "holding_days": int(p.hold_days),
                        "entry_type": "pullback_breakout" if sk == "mid_swing" else sk,
                        "sector": str(t2s.get(t, "")),
                        "RSI_at_entry": float(getattr(p, "rsi_at_entry", float("nan"))),
                        "probability_at_entry": float(p.entry_prob),
                        "range_expansion_at_entry": float(getattr(p, "range_exp_at_entry", float("nan"))),
                        "breakout_level": float(getattr(p, "breakout_level", float("nan"))),
                    }
                )
                del strategy_positions[sk][t]

        # Entries: gather candidates first, then batch predict probabilities once/day.
        candidates: list[tuple[str, pd.Series, str, bool, bool]] = []
        for t, f in daily_features.items():
            part = f[f["Date"] <= date]
            if len(part) < min_history_bars:
                continue
            row = part.iloc[-1]
            if current_model is None:
                continue
            if any(
                pd.isna(row[c])
                for c in [
                    "MA20",
                    "MA50",
                    "RSI14",
                    "PrevClose",
                    "PrevClose2",
                    "MA20_prev",
                    "MA20_prev2",
                    "Prev5High",
                    "Prev3High",
                    "MA100",
                    "High20",
                    "Return_3d",
                    "RS_5",
                    "RS_10",
                ]
            ):
                continue
            sector = t2s.get(t, "")
            sector_top3 = sector in top3
            sector_top2_cons = sector in top2 and sector in top2_prev
            candidates.append((t, row, sector, sector_top3, sector_top2_cons))

        if current_model is None or not candidates:
            top2_prev = top2
            total_equity = 0.0
            for sk in selected:
                total_equity += mark_to_market(strategy_cash[sk], strategy_positions[sk], prices_today)
            daily_rows.append({"Date": date, "PortfolioValue": total_equity})
            continue

        X_today = pd.DataFrame(
            [{c: float(r[c]) for c in FEATURE_COLUMNS} for _, r, _, _, _ in candidates],
            columns=FEATURE_COLUMNS,
        )
        probs_today = current_model.predict_proba(X_today)[:, 1]

        for (t, row, sector, sector_top3, sector_top2_cons), prob in zip(candidates, probs_today):
            prob = float(prob)
            for sk in selected:
                spec = STRATEGY_SPECS[sk]
                sp = strategy_params.get(sk, {})
                max_positions = int(sp.get("max_positions", spec.max_positions))
                max_trades_per_day = int(sp.get("max_trades_per_day", spec.max_trades_per_day))
                if trades_today[sk] >= spec.max_trades_per_day:
                    continue
                if trades_today[sk] >= max_trades_per_day:
                    continue
                if len(strategy_positions[sk]) >= max_positions:
                    continue
                if t in strategy_positions[sk]:
                    continue
                enter = False
                if sk == "short_swing":
                    ss = strategy_params.get("short_swing", {})
                    enter = short_swing_entry(
                        row,
                        prob,
                        sector_top3,
                        prob_min=float(ss.get("prob_min", 0.60)),
                        rsi_min=float(ss.get("rsi_min", 50.0)),
                        rsi_max=float(ss.get("rsi_max", 65.0)),
                    )
                elif sk == "mid_swing":
                    ms = strategy_params.get("mid_swing", {})
                    sec5, sec10 = sec_momo_map.get(sector, (float("-inf"), float("-inf")))
                    sector_momo_ok = (sec5 > float(ms.get("sector_ret5_min", 0.01))) and (
                        sec10 > float(ms.get("sector_ret10_min", 0.02))
                    )
                    enter = mid_swing_entry(
                        row,
                        prob,
                        sector_top3 and market_trend and sector_momo_ok,
                        pullback_band=float(ms.get("pullback_band", 0.03)),
                        prob_min=float(ms.get("prob_min", 0.65)),
                        rsi_min=float(ms.get("rsi_min", 45.0)),
                    )
                elif sk == "positional":
                    ps = strategy_params.get("positional", {})
                    enter = positional_entry(
                        row,
                        prob,
                        sector_top2_cons,
                        prob_min=float(ps.get("prob_min", 0.65)),
                        rsi_min=float(ps.get("rsi_min", 50.0)),
                        rsi_max=float(ps.get("rsi_max", 60.0)),
                    )
                elif sk == "intraday":
                    intra = raw.get(t, {}).get("1m")
                    it = strategy_params.get("intraday", {})
                    enter = (
                        sector in top2
                        and intraday_entry_signal(
                            intra,
                            trade_date=date,
                            volume_spike_mult=float(it.get("volume_spike_mult", 1.5)),
                        )
                        and prob > float(it.get("prob_min", 0.55))
                    )
                if not enter:
                    continue

                equity = mark_to_market(strategy_cash[sk], strategy_positions[sk], prices_today)
                max_per_stock = exposure_limit_value(equity, 0.40)
                target_value = equity * (1.0 / max(1, max_positions))
                value = cap_position_value(target_value, strategy_cash[sk], max_per_stock)
                strategy_cash[sk], shares, exec_px = execute_buy(
                    strategy_cash[sk], float(row["Close"]), value, transaction_cost_bps
                )
                if shares <= 0:
                    continue
                strategy_positions[sk][t] = Position(
                    ticker=t,
                    strategy=sk,
                    shares=shares,
                    entry_price=float(exec_px),
                    entry_prob=prob,
                )
                # Attach diagnostics on the position object for later logging.
                setattr(strategy_positions[sk][t], "rsi_at_entry", float(row["RSI14"]))
                setattr(strategy_positions[sk][t], "range_exp_at_entry", float(row["RangeExpansion"]))
                setattr(strategy_positions[sk][t], "breakout_level", float(row["Prev5High"]))
                trades_today[sk] += 1
                logs.append(
                    {
                        "Date": date,
                        "Symbol": t,
                        "Strategy": sk,
                        "Side": "BUY",
                        "Action": "ENTRY",
                        "entry_type": "pullback_breakout" if sk == "mid_swing" else sk,
                        "sector": str(t2s.get(t, "")),
                        "entry_price": float(exec_px),
                        "RSI_at_entry": float(row["RSI14"]),
                        "probability_at_entry": float(prob),
                        "range_expansion_at_entry": float(row["RangeExpansion"]),
                        "breakout_level": float(row["Prev5High"]),
                    }
                )

        top2_prev = top2

        total_equity = 0.0
        for sk in selected:
            total_equity += mark_to_market(strategy_cash[sk], strategy_positions[sk], prices_today)
        daily_rows.append({"Date": date, "PortfolioValue": total_equity})

    eq = pd.DataFrame(daily_rows).sort_values("Date").reset_index(drop=True)
    ret = eq["PortfolioValue"].pct_change().dropna()
    metrics = {
        "Total Return": float(eq["PortfolioValue"].iloc[-1] / eq["PortfolioValue"].iloc[0] - 1.0),
        "Sharpe Ratio": float((ret.mean() / ret.std()) * (252 ** 0.5)) if len(ret) > 1 and ret.std() > 0 else 0.0,
        "Max Drawdown": float((eq["PortfolioValue"] / eq["PortfolioValue"].cummax() - 1.0).min()),
        "Win Rate": float(sum(wins.values()) / max(1, sum(trade_count.values()))),
        "Trade Count": int(sum(trade_count.values())),
    }
    strategy_table = pd.DataFrame(
        [
            {
                "Strategy": sk,
                "Cash": strategy_cash[sk],
                "OpenPositions": len(strategy_positions[sk]),
                "PnL": strategy_pnl[sk],
                "Trades": trade_count[sk],
                "WinRate": float(wins[sk] / max(1, trade_count[sk])),
            }
            for sk in selected
        ]
    )
    return {
        "equity_curve": eq,
        "metrics": metrics,
        "strategy_summary": strategy_table,
        "trade_log": pd.DataFrame(logs),
        "active_positions": pd.DataFrame(
            [{"Strategy": sk, "Ticker": p.ticker, "Shares": p.shares, "Entry": p.entry_price} for sk in selected for p in strategy_positions[sk].values()]
        ),
        "data_coverage": {
            "daily_loaded": int(len(daily_prices)),
            "daily_expected": int(len(NIFTY50_TICKERS)),
            "daily_missing": missing_daily,
            "intraday_enabled": bool(need_intraday),
        },
        "effective_run_params": {
            "selected_strategies": list(selected),
            "years": int(years),
            "initial_capital": float(initial_capital),
            "transaction_cost_bps": float(transaction_cost_bps),
            "min_daily_tickers": int(min_daily_tickers),
            "walk_forward": bool(walk_forward),
            "min_train_days": int(min_train_days),
            "retrain_every_days": int(retrain_every_days),
            "label_horizon_days": int(label_horizon_days),
            "train_max_days": None if train_max_days is None else int(train_max_days),
            "capital_weights_normalized": dict(weights),
            "strategy_params": dict(strategy_params),
        },
    }
