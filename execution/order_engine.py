from __future__ import annotations


DEFAULT_SLIPPAGE = 0.001  # 0.1%


def execute_buy(
    cash: float,
    price: float,
    value: float,
    transaction_cost_bps: float,
    *,
    slippage: float = DEFAULT_SLIPPAGE,
) -> tuple[float, float, float]:
    if price <= 0 or value <= 0 or cash <= 0:
        return cash, 0.0, float(price)
    executed_price = float(price) * (1.0 + float(slippage))
    shares = value / executed_price
    gross = shares * executed_price
    total_cost = gross * (1.0 + transaction_cost_bps / 10000.0)
    if total_cost > cash:
        shares = cash / (executed_price * (1.0 + transaction_cost_bps / 10000.0))
        gross = shares * executed_price
        total_cost = gross * (1.0 + transaction_cost_bps / 10000.0)
    return cash - total_cost, shares, executed_price


def execute_sell(
    cash: float,
    price: float,
    shares: float,
    transaction_cost_bps: float,
    *,
    slippage: float = DEFAULT_SLIPPAGE,
) -> tuple[float, float, float]:
    if price <= 0 or shares <= 0:
        return cash, 0.0, float(price)
    executed_price = float(price) * (1.0 - float(slippage))
    gross = shares * executed_price
    proceeds = gross * (1.0 - transaction_cost_bps / 10000.0)
    return cash + proceeds, proceeds, executed_price
