from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score

@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "Random Forest"
    test_fraction: float = 0.2
    random_state: int = 42
    min_rows: int = 200


def _build_model(cfg: ModelConfig):
    token = cfg.model_type.lower().strip().replace("_", " ").replace("-", " ")
    if token in {"rf", "random forest"}:
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=5,
            random_state=cfg.random_state,
            n_jobs=-1,
            class_weight="balanced",
        )
    if token in {"gb", "gradient boosting"}:
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=cfg.random_state,
        )
    raise ValueError("model_type must be 'Random Forest' or 'Gradient Boosting'")


def train_model(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    cfg: ModelConfig | None = None,
) -> tuple[object, pd.DataFrame, float]:
    """
    Chronological split training. Returns model, test dataframe with probabilities, and accuracy.
    """
    if cfg is None:
        cfg = ModelConfig()
    if cfg.test_fraction <= 0 or cfg.test_fraction >= 1:
        raise ValueError("test_fraction must be between 0 and 1")

    needed = list(feature_cols) + ["Date", "Target"]
    miss = [c for c in needed if c not in dataset.columns]
    if miss:
        raise ValueError(f"Dataset missing columns: {miss}")

    data = dataset.sort_values("Date").reset_index(drop=True)
    if len(data) < cfg.min_rows:
        raise ValueError(f"Not enough rows to train ({len(data)} < {cfg.min_rows}).")

    split_idx = int((1 - cfg.test_fraction) * len(data))
    train_df = data.iloc[:split_idx].copy()
    test_df = data.iloc[split_idx:].copy()

    model = _build_model(cfg)
    model.fit(train_df[feature_cols], train_df["Target"])

    test_df["Probability"] = model.predict_proba(test_df[feature_cols])[:, 1]
    test_df["Pred"] = (test_df["Probability"] >= 0.5).astype(int)
    acc = float(accuracy_score(test_df["Target"], test_df["Pred"]))
    return model, test_df, acc


def fit_model_full(
    dataset: pd.DataFrame,
    feature_cols: list[str],
    cfg: ModelConfig | None = None,
) -> object:
    """
    Fit a model on ALL provided rows (no test split).

    Used for walk-forward/rolling training in backtests where the model is retrained
    repeatedly on the expanding history up to each evaluation date.
    """
    if cfg is None:
        cfg = ModelConfig()
    needed = list(feature_cols) + ["Target"]
    miss = [c for c in needed if c not in dataset.columns]
    if miss:
        raise ValueError(f"Dataset missing columns: {miss}")
    data = dataset.reset_index(drop=True)
    if len(data) < cfg.min_rows:
        raise ValueError(f"Not enough rows to train ({len(data)} < {cfg.min_rows}).")
    model = _build_model(cfg)
    model.fit(data[feature_cols], data["Target"])
    return model

