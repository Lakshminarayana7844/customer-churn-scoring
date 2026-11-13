"""Build the committed churn sample from the public IBM Telco churn dataset.

Source: IBM Telco Customer Churn sample data (7043 customers), mirrored at
https://raw.githubusercontent.com/Giskard-AI/examples/main/datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv

The full file is ~1 MB. To keep this repository light we commit a
stratified 1100-row sample with the same churn rate (~26.5%).
Re-running this script regenerates data/telco_churn_sample.csv exactly.
"""
from pathlib import Path

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/Giskard-AI/examples/main/"
    "datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv"
)
SAMPLE_ROWS = 1100
SEED = 42


def build_sample(full: pd.DataFrame, rows: int = SAMPLE_ROWS, seed: int = SEED) -> pd.DataFrame:
    frac = rows / len(full)
    parts = [g.sample(frac=frac, random_state=seed) for _, g in full.groupby("Churn")]
    sample = pd.concat(parts)
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    full = pd.read_csv(SOURCE_URL)
    sample = build_sample(full)
    out = Path(__file__).resolve().parents[1] / "data" / "telco_churn_sample.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    print(f"wrote {len(sample)} rows to {out}")


if __name__ == "__main__":
    main()
