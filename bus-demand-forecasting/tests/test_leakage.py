import pandas as pd

from busdemand import leakage_audit as L


def test_no_forbidden_features():
    assert L.check_forbidden_inputs()["passed"]


def test_chronological_split_is_ordered():
    dates = pd.Series(pd.date_range("2022-01-01", "2025-12-31"))
    r = L.check_split(dates)
    assert r["passed"] and r["train"][1] < r["validation"][0] and r["validation"][1] < r["test"][0]


def test_future_invariance_on_real_clean_data():
    from busdemand import config as C
    clean = pd.read_csv(C.CLEAN_TRIPS, parse_dates=["departure_datetime", "date"])
    assert L.check_future_invariance(clean, n_samples=8)["passed"]


def test_no_sensitive_columns():
    assert L.check_no_sensitive_columns(["route_id", "passengers", "capacity_seats"])["passed"]
    assert not L.check_no_sensitive_columns(["route_id", "passenger_phone"])["passed"]
