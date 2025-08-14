import pandas as pd

def test_data_basic():
    df = pd.read_csv("data/transactions.csv", nrows=1000)
    assert df["amount"].ge(0).all()
    assert df["merchant_cat"].notna().all()
    assert set(df["is_fraud"].unique()).issubset({0,1})
