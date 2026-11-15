import numpy as np
import pandas as pd

from busdemand.evaluation import RouteTimeBaseline, chronological_masks, high_demand_flag_metrics, regression_metrics, wape


def test_wape_and_metrics_known_values():
    y, p = np.array([10, 20, 30.0]), np.array([12, 18, 30.0])
    assert abs(wape(y, p) - 4 / 60) < 1e-12
    m = regression_metrics(y, p)
    assert abs(m["MAE"] - 4 / 3) < 1e-12 and abs(m["RMSE"] - np.sqrt(8 / 3)) < 1e-12 and abs(m["Bias"]) < 1e-12


def test_masks_partition_and_order():
    d = pd.Series(pd.date_range("2025-01-01", "2025-12-31"))
    tr, va, te = chronological_masks(d, "2025-06-30", "2025-09-30")
    assert (tr.astype(int) + va.astype(int) + te.astype(int) == 1).all()
    assert d[tr].max() < d[va].min() and d[va].max() < d[te].min()


def test_baseline_fallbacks():
    df = pd.DataFrame({"route_id": ["A"] * 4, "slot": ["08:00"] * 4,
                       "departure_datetime": pd.to_datetime(["2025-01-06 08:00", "2025-01-13 08:00", "2025-01-07 08:00", "2025-01-08 08:00"]),
                       "passengers": [10.0, 20.0, 30.0, 40.0]})
    b = RouteTimeBaseline().fit(df)
    q = pd.DataFrame({"route_id": ["A", "A", "Z"], "departure_datetime": pd.to_datetime(["2025-01-20 08:00", "2025-01-19 08:00", "2025-01-20 08:00"])})
    p = b.predict(q)
    assert p[0] == 15.0          # Monday 08:00 mean of 10 and 20
    assert p[1] == 25.0          # Sunday unseen -> route/slot mean
    assert p[2] == 25.0          # unknown route -> global mean


def test_flag_metrics():
    r = high_demand_flag_metrics([45, 45, 10, 10], [44, 20, 44, 10], [45, 45, 45, 45])
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 1
