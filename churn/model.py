"""Model pipeline, evaluation and threshold policy."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline

from .features import build_preprocessor

RANDOM_STATE = 42


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_iter=300,
                    learning_rate=0.06,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate(y_true: np.ndarray, proba: np.ndarray) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }


def choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    offer_cost: float = 50.0,
    retained_value: float = 500.0,
    max_flag_rate: float = 0.2,
) -> dict:
    """Pick the probability threshold with the best expected value.

    Every customer flagged as a churn risk gets a retention offer that costs
    offer_cost. A correctly flagged churner who stays is worth
    retained_value. The retention team can only contact a share of the
    customer base, so thresholds flagging more than max_flag_rate are out.
    The best remaining threshold maximises
    TP * (retained_value - offer_cost) - FP * offer_cost.
    """
    best = {"threshold": 0.5, "expected_value": float("-inf"), "flag_rate": 0.0}
    for t in np.arange(0.05, 0.95, 0.01):
        flag = proba >= t
        flag_rate = float(flag.mean())
        if flag_rate > max_flag_rate:
            continue
        tp = int(np.sum(flag & (y_true == 1)))
        fp = int(np.sum(flag & (y_true == 0)))
        ev = tp * (retained_value - offer_cost) - fp * offer_cost
        if ev > best["expected_value"]:
            best = {
                "threshold": round(float(t), 2),
                "expected_value": float(ev),
                "flag_rate": round(flag_rate, 4),
            }
    return best
