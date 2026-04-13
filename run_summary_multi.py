"""Build a plain-text multi-strategy backtest summary for copy/paste notes."""

from __future__ import annotations

import math

import pandas as pd


def build_multi_strategy_summary_text(
    *,
    run_params: dict,
    result: dict,
    runtime_seconds: float | None = None,
) -> str:
    m = result.get("metrics") or {}
    eq = result.get("equity_curve")

    start_d = end_d = "—"
    start_eq = end_eq = float("nan")
    if isinstance(eq, pd.DataFrame) and len(eq):
        try:
            start_d = str(eq["Date"].iloc[0])[:10]
            end_d = str(eq["Date"].iloc[-1])[:10]
            start_eq = float(eq["PortfolioValue"].iloc[0])
            end_eq = float(eq["PortfolioValue"].iloc[-1])
        except Exception:
            pass

    cov = result.get("data_coverage") or {}
    cov_line = "n/a"
    try:
        loaded = int(cov.get("daily_loaded", 0))
        expected = int(cov.get("daily_expected", 0))
        if expected > 0:
            cov_line = f"{loaded}/{expected} daily series loaded"
    except Exception:
        pass

    strat_table = result.get("strategy_summary")
    strat_lines = "  (n/a)"
    if isinstance(strat_table, pd.DataFrame) and len(strat_table):
        rows: list[str] = []
        for _, r in strat_table.iterrows():
            rows.append(
                f"  - {r.get('Strategy','')}: cash={float(r.get('Cash',0.0)):.2f}, "
                f"open={int(r.get('OpenPositions',0))}, pnl={float(r.get('PnL',0.0)):.2f}, "
                f"trades={int(r.get('Trades',0))}, winrate={float(r.get('WinRate',0.0)) * 100:.1f}%"
            )
        strat_lines = "\n".join(rows) if rows else "  (n/a)"

    lines: list[str] = [
        "MULTI-STRATEGY BACKTEST SUMMARY",
        "===============================",
        "",
        "### Run meta",
        f"- Wall-clock runtime: {runtime_seconds:.1f}s" if runtime_seconds is not None else "- Wall-clock runtime: n/a",
        f"- Backtest calendar: {start_d} → {end_d}",
        f"- Data coverage: {cov_line}",
        "",
        "### Parameters used",
    ]
    for k in sorted(run_params.keys()):
        lines.append(f"- {k}: {run_params[k]}")

    lines.extend(
        [
            "",
            "### Results",
            f"- Total return: {float(m.get('Total Return', 0.0)) * 100:.2f}%",
            f"- Portfolio (start → end): {start_eq:,.2f} → {end_eq:,.2f}" if math.isfinite(start_eq) else "- Portfolio values: n/a",
            f"- Sharpe ratio (daily, approx): {float(m.get('Sharpe Ratio', 0.0)):.4f}",
            f"- Max drawdown: {float(m.get('Max Drawdown', 0.0)) * 100:.2f}%",
            f"- Win rate (per closed trade, pooled): {float(m.get('Win Rate', 0.0)) * 100:.2f}%",
            f"- Trade count (total): {int(m.get('Trade Count', 0))}",
            "",
            "### Per-strategy snapshot",
            strat_lines,
        ]
    )

    return "\n".join(lines)

