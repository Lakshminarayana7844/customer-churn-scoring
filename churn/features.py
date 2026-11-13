"""Feature definitions and cleaning for the Telco churn data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
TARGET = "Churn"
ID_COLUMN = "customerID"
FEATURES = CATEGORICAL + NUMERIC


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with consistent types.

    TotalCharges arrives as text with blanks for new customers; coerce to
    float and fill blanks with MonthlyCharges (first invoice not issued yet).
    """
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"])
    out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    for col in CATEGORICAL:
        out[col] = out[col].astype(str)
    return out


def encode_target(df: pd.DataFrame) -> np.ndarray:
    """Map Churn Yes/No to 1/0."""
    return (df[TARGET] == "Yes").astype(int).to_numpy()


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
            ("num", StandardScaler(), NUMERIC),
        ]
    )
