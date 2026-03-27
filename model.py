from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils import set_seed


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "logreg"  # "logreg" or "rf"
    test_fraction: float = 0.2
    random_state: int = 42


@dataclass
class ModelResult:
    model: object
    test_df: pd.DataFrame
    test_proba: np.ndarray  # P(Target=1) for each row in test_df
    test_pred: np.ndarray   # predicted class labels
    test_accuracy: float


FEATURE_COLS = ["MA20", "MA50", "RSI14", "Daily_Return", "Rolling_Volatility_10"]


def train_model(model_df: pd.DataFrame, cfg: ModelConfig | None = None) -> ModelResult:
    """
    Train a classifier on historical features and return next-day probabilities
    for the out-of-sample (test) period.

    Uses a chronological split (first 80% train, last 20% test) to avoid leakage.
    """
    if cfg is None:
        cfg = ModelConfig()

    if cfg.test_fraction <= 0 or cfg.test_fraction >= 1:
        raise ValueError("test_fraction must be between 0 and 1")

    if not all(c in model_df.columns for c in FEATURE_COLS + ["Target"]):
        missing = [c for c in FEATURE_COLS + ["Target"] if c not in model_df.columns]
        raise ValueError(f"model_df missing columns: {missing}")

    data = model_df.sort_values("Date").reset_index(drop=True)
    n = len(data)
    if n < 200:
        raise ValueError(f"Not enough data to train (rows={n}). Try another ticker.")

    split_idx = int((1 - cfg.test_fraction) * n)
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    X_train = train_df[FEATURE_COLS].values
    y_train = train_df["Target"].values
    X_test = test_df[FEATURE_COLS].values
    y_test = test_df["Target"].values

    set_seed(cfg.random_state)

    if cfg.model_type.lower() == "logreg":
        clf = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=cfg.random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    elif cfg.model_type.lower() in ("rf", "randomforest", "random_forest"):
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    else:
        raise ValueError("model_type must be 'logreg' or 'rf'")

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, pred))

    return ModelResult(
        model=clf,
        test_df=test_df,
        test_proba=proba,
        test_pred=pred,
        test_accuracy=acc,
    )

