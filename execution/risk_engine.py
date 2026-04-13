from __future__ import annotations


def cap_position_value(requested_value: float, cash: float, max_per_stock_value: float) -> float:
    return max(0.0, min(float(requested_value), float(cash), float(max_per_stock_value)))


def exposure_limit_value(total_equity: float, max_exposure_per_stock: float = 0.40) -> float:
    return max(0.0, float(total_equity) * float(max_exposure_per_stock))
