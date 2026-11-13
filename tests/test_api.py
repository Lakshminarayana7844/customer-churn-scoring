from fastapi.testclient import TestClient

from churn.api import app

PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_health():
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["rows_trained"] > 0
        assert body["metrics"]["roc_auc"] > 0.75


def test_score_single():
    with TestClient(app) as client:
        r = client.post("/score", json=PAYLOAD)
        assert r.status_code == 200
        body = r.json()
        assert 0.0 <= body["churn_probability"] <= 1.0
        assert isinstance(body["churn_flag"], bool)
        assert 0.0 < body["threshold"] < 1.0


def test_score_batch_and_missing_total_charges():
    payload2 = dict(PAYLOAD, TotalCharges=None)
    with TestClient(app) as client:
        r = client.post("/score/batch", json=[PAYLOAD, payload2])
        assert r.status_code == 200
        assert len(r.json()) == 2


def test_score_rejects_bad_payload():
    with TestClient(app) as client:
        r = client.post("/score", json={"gender": "Female"})
        assert r.status_code == 422
