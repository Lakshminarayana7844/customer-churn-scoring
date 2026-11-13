"""Train and evaluate the churn model on the committed sample data."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .features import FEATURES, clean, encode_target
from .model import RANDOM_STATE, choose_threshold, evaluate, make_pipeline

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "telco_churn_sample.csv"


def train(df: pd.DataFrame) -> dict:
    df = clean(df)
    X = df[FEATURES]
    y = encode_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    pipe = make_pipeline()
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, proba)
    policy = choose_threshold(y_test, proba)
    # Final model refit on all rows for serving.
    pipe.fit(X, y)
    return {"pipeline": pipe, "metrics": metrics, "policy": policy, "rows": len(df)}


def main() -> None:
    result = train(pd.read_csv(DATA_PATH))
    print(json.dumps({"rows": result["rows"], "metrics": result["metrics"], "policy": result["policy"]}, indent=2))


if __name__ == "__main__":
    main()
