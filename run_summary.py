"""Build a plain-text backtest summary for Streamlit copy-paste notes."""

from __future__ import annotations

import math

import pandas as pd


def build_backtest_summary_text(
    *,
    run_params: dict,
    result: dict,
    runtime_seconds: float | None = None,
) -> str:
    m = result.get("metrics") or {}
    snap = result.get("latest_snapshot") or {}

    diag_lines = "  (none)"
    diag = snap.get("diagnostics")
    if isinstance(diag, pd.DataFrame) and len(diag):
        diag_lines = "\n".join(f"  - {row['Metric']}: {row['Value']}" for _, row in diag.iterrows())

    eq = result.get("equity_curve")
    bn = result.get("benchmark_curve")
    strat_ret_pct = float(m.get("Total Return", 0.0)) * 100.0
    end_eq = start_eq = float("nan")
    end_bench = start_bench = float("nan")
    bench_ret_pct = float("nan")
    if isinstance(eq, pd.DataFrame) and len(eq):
        try:
            start_eq = float(eq["PortfolioValue"].iloc[0])
            end_eq = float(eq["PortfolioValue"].iloc[-1])
        except (IndexError, KeyError, ValueError):
            pass
    if isinstance(bn, pd.DataFrame) and len(bn):
        try:
            start_bench = float(bn["PortfolioValue"].iloc[0])
            end_bench = float(bn["PortfolioValue"].iloc[-1])
            if start_bench > 0:
                bench_ret_pct = (end_bench / start_bench - 1.0) * 100.0
        except (IndexError, KeyError, ValueError):
            pass

    start_d = end_d = "—"
    if isinstance(eq, pd.DataFrame) and len(eq) and "Date" in eq.columns:
        start_d = str(eq["Date"].iloc[0])[:10]
        end_d = str(eq["Date"].iloc[-1])[:10]

    if math.isfinite(bench_ret_pct):
        bench_line = f"- NIFTY 50 benchmark (aligned window): {bench_ret_pct:.2f}%"
    else:
        bench_line = "- NIFTY 50 benchmark (aligned window): n/a"

    lines: list[str] = [
        "BACKTEST RUN SUMMARY",
        "====================",
        "",
        "### Run meta",
    ]
    if runtime_seconds is not None:
        lines.append(f"- Wall-clock runtime: {runtime_seconds:.1f}s")
    else:
        lines.append("- Wall-clock runtime: n/a")
    lines.extend(
        [
            f"- Backtest calendar: {start_d} → {end_d}",
            "",
            "### Parameters used",
        ]
    )
    for k in sorted(run_params.keys()):
        lines.append(f"- {k}: {run_params[k]}")

    market_note = (
        "- Market filter: NIFTY Close > MA50; optional RSI gate enabled"
        if run_params.get("use_market_rsi_filter")
        else "- Market filter: NIFTY Close > MA50 only"
    )
    lines.extend(
        [
            "",
            "### Strategy notes (this build)",
            market_note,
            "- Sectors: top N by risk-adjusted momentum score; no median cutoff",
            "- Entry: Price > MA50; (A) pullback bounce within ±pullback_band of MA20 with up-day close, OR (B) MA20 reclaim (prev close < prev MA20, today close > MA20); RSI14 > rsi_entry_min; model prob > prob_buy_min; sector already restricted to top N",
            "- Exits: stop-loss, conditional MA20 exit, prob exit, time exit, partial take-profit (see parameters)",
            "",
            "### Results",
            f"- Strategy total return: {strat_ret_pct:.2f}%",
            f"- Strategy portfolio (start → end): {start_eq:,.2f} → {end_eq:,.2f}" if math.isfinite(start_eq) else "- Strategy portfolio values: n/a",
            bench_line,
            f"- Benchmark portfolio (start → end): {start_bench:,.2f} → {end_bench:,.2f}" if math.isfinite(start_bench) else "- Benchmark portfolio values: n/a",
            f"- Sharpe ratio (daily, approx): {float(m.get('Sharpe Ratio', float('nan'))):.4f}",
            f"- Max drawdown: {float(m.get('Max Drawdown', 0.0)) * 100:.2f}%",
            f"- Win rate (daily step-up): {float(m.get('Win Rate', 0.0)) * 100:.2f}%",
            "",
            "### Diagnostics (cumulative)",
            diag_lines,
            "",
            "### Last snapshot",
            f"- Market filter (last evaluation): {'ON' if snap.get('market_on') else 'OFF'}",
            f"- Top sectors: {', '.join(snap.get('top_sectors') or []) or '—'}",
        ]
    )

    sel = snap.get("selected_stocks")
    if isinstance(sel, pd.DataFrame) and len(sel):
        lines.append("- Selected names (last evaluation):")
        for _, row in sel.iterrows():
            p = row.get("Probability", float("nan"))
            try:
                p_str = f"{float(p):.3f}"
            except (TypeError, ValueError):
                p_str = str(p)
            lines.append(
                f"  - {row.get('Ticker', '')} | sector={row.get('Sector', '')} | P={p_str} | {row.get('Signal', '')}"
            )
    else:
        lines.append("- Selected names (last evaluation): —")

    return "\n".join(lines)
