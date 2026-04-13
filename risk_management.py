from __future__ import annotations

import numpy as np


def allocation_from_probability(
    probability: float,
    *,
    high_prob_cutoff: float = 0.75,
    alloc_high: float = 0.40,
    alloc_low: float = 0.30,
) -> float:
    return float(alloc_high) if float(probability) > float(high_prob_cutoff) else float(alloc_low)


def should_partial_book(
    entry_price: float,
    current_price: float,
    already_booked: bool,
    *,
    take_profit_pct: float = 0.03,
) -> bool:
    if already_booked:
        return False
    ret = current_price / entry_price - 1.0
    return bool(ret >= float(take_profit_pct))


def stop_loss_triggered(entry_price: float, current_price: float, stop_loss: float = -0.015) -> bool:
    ret = current_price / entry_price - 1.0
    return bool(ret <= float(stop_loss))


def apply_transaction_cost(notional: float, cost_bps: float) -> float:
    # bps = basis points (1 bps = 0.01%)
    return float(notional) * (1.0 - float(cost_bps) / 10000.0)


def bounded_probability(p: float) -> float:
    return float(np.clip(float(p), 0.0, 1.0))
