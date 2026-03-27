from __future__ import annotations

import numpy as np
import pandas as pd


def prob_to_signal(
    proba_buy: np.ndarray,
    *,
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
) -> np.ndarray:
    """
    Convert predicted probability of next-day return>0 into discrete trading signals.
    """
    if not (0 < sell_threshold < buy_threshold < 1):
        raise ValueError("Require 0 < sell_threshold < buy_threshold < 1")

    signals = np.array(["HOLD"] * len(proba_buy), dtype=object)
    signals[proba_buy > buy_threshold] = "BUY"
    signals[proba_buy < sell_threshold] = "SELL"
    return signals


def prob_to_signal_quantile(
    proba_buy: np.ndarray,
    *,
    buy_quantile: float = 0.7,
    sell_quantile: float = 0.3,
) -> np.ndarray:
    """
    Quantile-based signal mapping.

    Instead of using fixed probability cutoffs, we compute cutoffs from the
    predicted probability distribution for the test period:
    - BUY if proba >= q_buy
    - SELL if proba <= q_sell
    - otherwise HOLD

    This is useful when a model's probabilities are "compressed" (common for
    Logistic Regression), causing fixed thresholds to produce all HOLD.
    """
    if not (0 < sell_quantile < buy_quantile < 1):
        raise ValueError("Require 0 < sell_quantile < buy_quantile < 1")

    q_buy = float(np.quantile(proba_buy, buy_quantile))
    q_sell = float(np.quantile(proba_buy, sell_quantile))

    signals = np.array(["HOLD"] * len(proba_buy), dtype=object)
    signals[proba_buy >= q_buy] = "BUY"
    signals[proba_buy <= q_sell] = "SELL"
    return signals


def attach_signals(
    test_df: pd.DataFrame,
    test_proba: np.ndarray,
    *,
    threshold_mode: str = "fixed",
    buy_threshold: float = 0.6,
    sell_threshold: float = 0.4,
    buy_quantile: float = 0.7,
    sell_quantile: float = 0.3,
) -> pd.DataFrame:
    """
    Return a copy of `test_df` with `Proba` and `Signal` columns attached.
    """
    out = test_df.copy()
    out["Proba"] = test_proba

    proba = out["Proba"].values
    mode = threshold_mode.lower().strip()
    if mode == "fixed":
        out["Signal"] = prob_to_signal(proba, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    elif mode == "quantile":
        out["Signal"] = prob_to_signal_quantile(proba, buy_quantile=buy_quantile, sell_quantile=sell_quantile)
    else:
        raise ValueError("threshold_mode must be 'fixed' or 'quantile'")

    return out

