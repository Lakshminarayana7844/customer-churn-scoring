import pandas as pd

from churn.features import FEATURES, build_preprocessor, clean, encode_target


def sample_df():
    return pd.read_csv("data/telco_churn_sample.csv")


def test_clean_total_charges_blanks_become_numeric():
    df = sample_df()
    cleaned = clean(df)
    assert cleaned["TotalCharges"].dtype == float
    assert cleaned["TotalCharges"].notna().all()


def test_clean_keeps_all_feature_columns():
    cleaned = clean(sample_df())
    for col in FEATURES:
        assert col in cleaned.columns


def test_encode_target_binary():
    y = encode_target(clean(sample_df()))
    assert set(y) <= {0, 1}
    assert 0.2 < y.mean() < 0.35  # churn rate stays near 26.5%


def test_preprocessor_expands_categoricals():
    cleaned = clean(sample_df())
    prep = build_preprocessor()
    Xt = prep.fit_transform(cleaned[FEATURES])
    assert Xt.shape[0] == len(cleaned)
    assert Xt.shape[1] > len(FEATURES)
