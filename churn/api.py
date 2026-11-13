"""FastAPI scoring service for customer churn.

The model is trained at startup from the committed sample data. Training on
1100 rows takes under a second, keeps the image self-contained and makes the
served model exactly reproducible from the repo contents.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from .train import DATA_PATH, train


@asynccontextmanager
async def lifespan(app: FastAPI):
    _state.update(train(pd.read_csv(DATA_PATH)))
    yield


app = FastAPI(title="Customer Churn Scoring", version="1.0.0", lifespan=lifespan)


class Customer(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: float = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: Optional[float] = None


class Score(BaseModel):
    churn_probability: float
    churn_flag: bool
    threshold: float


_state: dict = {}


def _score_rows(df: pd.DataFrame) -> List[Score]:
    pipe = _state["pipeline"]
    threshold = _state["policy"]["threshold"]
    work = df.copy()
    work["TotalCharges"] = work["TotalCharges"].fillna(work["MonthlyCharges"])
    proba = pipe.predict_proba(work)[:, 1]
    return [
        Score(
            churn_probability=round(float(p), 4),
            churn_flag=bool(p >= threshold),
            threshold=threshold,
        )
        for p in proba
    ]


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "rows_trained": _state.get("rows"),
        "metrics": _state.get("metrics"),
        "policy": _state.get("policy"),
    }


@app.post("/score", response_model=Score)
def score(customer: Customer) -> Score:
    return _score_rows(pd.DataFrame([customer.model_dump()]))[0]


@app.post("/score/batch", response_model=List[Score])
def score_batch(customers: List[Customer]) -> List[Score]:
    return _score_rows(pd.DataFrame([c.model_dump() for c in customers]))
