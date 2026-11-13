import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from churn.features import FEATURES, clean, encode_target
from churn.model import RANDOM_STATE, choose_threshold, evaluate, make_pipeline


def split():
    df = clean(pd.read_csv("data/telco_churn_sample.csv"))
    X, y = df[FEATURES], encode_target(df)
    return train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)


def test_pipeline_beats_baseline_auc():
    X_train, X_test, y_train, y_test = split()
    pipe = make_pipeline().fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba)
    assert 0.0 <= proba.min() and proba.max() <= 1.0
    assert metrics["roc_auc"] > 0.75  # majority baseline is 0.5
    assert metrics["pr_auc"] > y_test.mean()


def test_training_is_deterministic():
    X_train, X_test, y_train, y_test = split()
    p1 = make_pipeline().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    p2 = make_pipeline().fit(X_train, y_train).predict_proba(X_test)[:, 1]
    np.testing.assert_allclose(p1, p2)


def test_choose_threshold_in_range_and_scores_offer():
    y = np.array([1, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.4, 0.3, 0.2, 0.1, 0.05])
    best = choose_threshold(y, proba)
    assert 0.0 < best["threshold"] < 1.0
    assert best["expected_value"] > 0
