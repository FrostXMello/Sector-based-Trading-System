# Multi-Factor AI Financial Decision System

End-to-end demo pipeline:

1. Download 5 years of daily OHLCV data with `yfinance`
2. Build features (MA20/MA50, RSI14, daily returns, rolling volatility)
3. Train a classifier (Logistic Regression or Random Forest) to predict next-day return direction
4. Convert predicted probability to `BUY` / `SELL` / `HOLD`
5. Backtest a simple “all-in long / cash” strategy
6. Visualize results in a Streamlit dashboard

## Setup

```bash
pip install -r requirements.txt
```

## Run Streamlit UI

```bash
streamlit run app.py
```

## Notes

- Splits train/test chronologically to avoid leakage.
- The model predicts whether the *next* day's return is positive.
