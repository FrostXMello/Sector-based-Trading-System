from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    ticker: str
    strategy: str
    shares: float
    entry_price: float
    entry_prob: float
    hold_days: int = 0
    partial_booked: bool = False
    trail_stop: float | None = None


def mark_to_market(cash: float, positions: dict[str, Position], prices: dict[str, float]) -> float:
    total = float(cash)
    for t, p in positions.items():
        px = float(prices.get(t, p.entry_price))
        total += p.shares * px
    return total
