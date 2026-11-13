# Customer Churn Scoring

Predict which customers are likely to churn, and decide who is worth a
retention offer. End-to-end machine learning project: data preparation,
a scikit-learn training pipeline, honest evaluation, a cost-based decision
policy, and a FastAPI scoring service with tests, Docker and CI.

## Problem

Acquiring a new customer costs far more than keeping an existing one. A
retention team needs two things: a ranked list of customers by churn risk,
and a rule for who to contact, since contacting everyone wastes budget.
This project delivers both as a small, reproducible service.

## Data

IBM Telco Customer Churn sample data (7,043 customers, 21 columns: account,
services, contract, charges and a churn label). The full file is mirrored at
`https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv`.

To keep the repository light, `data/telco_churn_sample.csv` holds a
stratified 1,100-row sample with the original churn rate (~26.5%).
`python scripts/prepare_data.py` regenerates it exactly from the public
source (seed fixed). Training on the full dataset works without any code
change: point `churn/train.py` at the full CSV.

## Approach

- Cleaning: `TotalCharges` arrives as text with blanks for brand-new
  customers; it is coerced to numeric and blanks fall back to
  `MonthlyCharges` (`churn/features.py`).
- Features: 15 categorical (one-hot) + 4 numeric (scaled) in a
  `ColumnTransformer` inside a single sklearn `Pipeline`, so train and
  serve apply identical transformations.
- Model: `HistGradientBoostingClassifier` (300 iterations, learning rate
  0.06, L2 1.0, fixed seed - training is deterministic).
- Split: 75/25 stratified. Metrics are computed on the held-out 25%, then
  the final model is refit on all 1,100 rows for serving.

## Results (held-out test set, 275 rows)

| Metric | Value | Baseline |
|---|---|---|
| ROC-AUC | 0.845 | 0.500 (random) |
| PR-AUC | 0.642 | 0.265 (churn rate) |
| Brier score | 0.145 | 0.195 (always predict base rate) |

## Decision policy

Accuracy is not the business goal; saving the right customers is.
`churn/model.py` scores each candidate threshold by expected value:

```
expected value = TP * (retained_value - offer_cost) - FP * offer_cost
```

with offer_cost = 50 (a retention call plus discount) and
retained_value = 500 (a year of margin). The retention team can contact at
most 20% of the customer base, so thresholds flagging more are excluded.
On the test set the best policy is:

- threshold 0.64, flagging 19.6% of customers
- expected value 15,800 per 275 customers scored (vs 0 for doing nothing)

Both numbers move with the cost inputs; tune them in
`choose_threshold(offer_cost=..., retained_value=..., max_flag_rate=...)`.

## API

`churn/api.py` serves the model. It trains at startup from the committed
sample (under a second, fully reproducible).

```bash
pip install -r requirements.txt
uvicorn churn.api:app --reload
```

- `GET /health` - status, training rows, metrics, active policy
- `POST /score` - one customer, returns `churn_probability`, `churn_flag`
  and the `threshold` used
- `POST /score/batch` - a list of customers, same shape back

```bash
curl -X POST localhost:8000/score -H 'Content-Type: application/json' -d '{
  "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
  "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service",
  "InternetService": "DSL", "OnlineSecurity": "No", "OnlineBackup": "Yes",
  "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
  "StreamingMovies": "No", "Contract": "Month-to-month",
  "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85, "TotalCharges": 29.85}'
```

## Docker

```bash
docker build -t churn-scoring .
docker run -p 8000:8000 churn-scoring
```

The build runs `python -m churn.train` once to print the metrics into the
build log, then serves with uvicorn.

## Tests and CI

11 tests cover cleaning (blank charges, types), the feature pipeline
(shape, encoding), the model (beats baseline AUC, deterministic training,
threshold policy) and the API (health, single and batch scoring, invalid
payload returns 422).

```bash
python -m pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs the same suite on every
push and pull request.

## Layout

```
churn/            features, model, training, API
scripts/          prepare_data.py regenerates the committed sample
data/             telco_churn_sample.csv (1,100 rows)
tests/            pytest suite
Dockerfile        self-contained scoring image
.github/          CI workflow
```

## Limitations

- Trained on a 1,100-row sample of a 2019-era sample dataset; real churn
  drivers drift and the model needs periodic retraining on current data.
- The expected-value policy uses flat costs; real retention programs price
  offers per customer segment.
- Probabilities rank risk well but are not calibrated to true churn odds;
  add isotonic or Platt calibration before using them as literal
  probabilities.
