import numpy as np
import pandas as pd

from busdemand import config as C
from busdemand.features import FEATURES, HistoryIndex, build_features, calendar_features


def test_lags_use_14_21_28_35_days_back(small_history):
    idx = HistoryIndex(small_history)
    target = pd.Timestamp("2025-03-20")                       # a Thursday, plenty of history
    tday = HistoryIndex.day_number(target)
    f = idx.history_features("R01", "08:00", tday)
    for k, col in enumerate(["lag1", "lag2", "lag3", "lag4"]):
        d = target - pd.Timedelta(days=14 + 7 * k)
        assert f[col] == float(d.toordinal() % 50), col
    assert f["_cutoff_day"] == tday - (C.HORIZON_DAYS + 1)


def test_features_do_not_change_when_future_data_changes(small_history):
    target = pd.Timestamp("2025-03-20")
    tday = HistoryIndex.day_number(target)
    a = HistoryIndex(small_history).history_features("R01", "08:00", tday)
    changed = small_history.copy()
    changed.loc[changed["date"] > target - pd.Timedelta(days=C.HORIZON_DAYS + 1), "passengers"] = 9999.0
    b = HistoryIndex(changed).history_features("R01", "08:00", tday)
    assert a == b


def test_build_features_shape_and_determinism(small_history):
    idx = HistoryIndex(small_history)
    df = pd.DataFrame({"route_id": ["R01", "R01"], "departure_datetime": pd.to_datetime(["2025-03-20 08:00", "2025-03-22 17:00"]),
                       "distance_km": [250.0, 250.0], "capacity_seats": [45.0, 45.0]})
    X1, X2 = build_features(df, idx), build_features(df, idx)
    assert list(X1[FEATURES].columns) == FEATURES
    pd.testing.assert_frame_equal(X1, X2)
    assert X1.loc[0, "is_peak_hour"] == 1 and X1.loc[1, "is_weekend"] == 1


def test_calendar_features_season_and_peak():
    f = calendar_features(pd.Series(pd.to_datetime(["2025-07-15 07:30", "2025-12-25 23:00"])))
    assert f.loc[0, "season"] == "monsoon" and f.loc[0, "is_peak_hour"] == 1
    assert f.loc[1, "season"] == "winter" and f.loc[1, "is_holiday"] == 1 and f.loc[1, "is_peak_hour"] == 0


def test_missing_history_gives_nan_not_crash(small_history):
    idx = HistoryIndex(small_history)
    f = idx.history_features("R01", "13:00", HistoryIndex.day_number(pd.Timestamp("2025-03-20")))   # slot never observed
    assert np.isnan(f["lag1"]) and f["n_lags_available"] == 0
