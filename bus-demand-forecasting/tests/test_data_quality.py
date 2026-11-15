import numpy as np
import pandas as pd

from busdemand.data_quality import clean_trips


def _raw():
    base = dict(route_id="R01", origin="A", destination="B")
    rows = [
        dict(trip_id="t1", distance_km=250.0, departure_datetime="2025-01-01 08:00:00", capacity_seats=45.0, passengers=30.0, **base),
        dict(trip_id="t1", distance_km=250.0, departure_datetime="2025-01-01 08:00:00", capacity_seats=45.0, passengers=30.0, **base),  # duplicate
        dict(trip_id="t2", distance_km=np.nan, departure_datetime="2025-01-02 08:00:00", capacity_seats=np.nan, passengers=20.0, **base),  # missing dist+cap
        dict(trip_id="t3", distance_km=250.0, departure_datetime="2025-01-03 08:00:00", capacity_seats=45.0, passengers=-1.0, **base),   # negative
        dict(trip_id="t4", distance_km=250.0, departure_datetime="2025-01-04 08:00:00", capacity_seats=45.0, passengers=99.0, **base),   # > capacity
        dict(trip_id="t5", distance_km=250.0, departure_datetime="2025-01-05 08:00:00", capacity_seats=45.0, passengers=np.nan, **base), # missing target
        dict(trip_id="t6", distance_km=250.0, departure_datetime="2025-01-06 08:00:00", capacity_seats=45.0, passengers=45.0, **base),   # sold out
    ]
    return pd.DataFrame(rows)


def test_cleaning_rules():
    clean, rep = clean_trips(_raw())
    assert rep["duplicates_dropped"] == 1
    assert clean["trip_id"].is_unique
    assert clean[["distance_km", "capacity_seats"]].isna().sum().sum() == 0
    assert clean.loc[clean.trip_id == "t2", "capacity_seats"].iloc[0] == 45.0     # filled from route+slot mode
    assert rep["impossible_passengers_set_nan"] == 2                               # -1 and 99
    assert not clean.loc[clean.trip_id.isin(["t3", "t4", "t5"]), "valid_target"].any()
    assert clean.loc[clean.trip_id == "t6", "sold_out"].iloc[0]
    assert (clean["passengers"].dropna() <= clean.loc[clean.passengers.notna(), "capacity_seats"]).all()


def test_no_valid_row_is_dropped_by_outlier_rule():
    clean, rep = clean_trips(_raw())
    assert rep["rows_clean"] == len(clean) == 6   # only the duplicate was removed
