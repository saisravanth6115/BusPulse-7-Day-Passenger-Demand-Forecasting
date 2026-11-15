"""Data-quality checks and cleaning for the raw trip export.

Rules (each one is counted in the quality report):
  1. parse datetimes; drop exact duplicate trips (same trip_id)
  2. distance_km missing  -> filled from the route master table
  3. capacity_seats missing -> filled with the most common capacity of the same route+slot
  4. passengers < 0 or > capacity_seats (physically impossible) -> set to NaN
  5. passengers NaN -> row kept (so the calendar is complete) but flagged valid_target=False
     and excluded from model training / evaluation
  6. outliers are FLAGGED (robust z-score within route+slot+weekday) but NOT removed:
     they are mostly genuine disruptions that the model must be evaluated on.
  7. sold_out flag (passengers == capacity): demand is censored at capacity.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C


def clean_trips(raw: pd.DataFrame, routes: pd.DataFrame | None = None):
    """Return (clean_df, report_dict)."""
    rep: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()
    df["departure_datetime"] = pd.to_datetime(df["departure_datetime"], errors="coerce")
    rep["bad_datetime_rows_dropped"] = int(df["departure_datetime"].isna().sum())
    df = df.dropna(subset=["departure_datetime"])

    rep["missing_before"] = {c: int(df[c].isna().sum()) for c in ["distance_km", "capacity_seats", "passengers"]}

    n0 = len(df)
    df = df.drop_duplicates(subset="trip_id", keep="first")
    rep["duplicates_dropped"] = int(n0 - len(df))

    df = df.sort_values("departure_datetime").reset_index(drop=True)
    df["slot"] = df["departure_datetime"].dt.strftime("%H:%M")
    df["date"] = df["departure_datetime"].dt.normalize()

    # 2. distance from route master (mode per route)
    dist_map = df.groupby("route_id")["distance_km"].agg(lambda s: s.mode().iloc[0])
    miss = df["distance_km"].isna()
    df.loc[miss, "distance_km"] = df.loc[miss, "route_id"].map(dist_map)
    rep["distance_filled"] = int(miss.sum())

    # 3. capacity: most frequent capacity for same route+slot (fall back to route)
    cap_slot = df.groupby(["route_id", "slot"])["capacity_seats"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    cap_route = df.groupby("route_id")["capacity_seats"].agg(lambda s: s.mode().iloc[0])
    miss = df["capacity_seats"].isna()
    keys = list(zip(df.loc[miss, "route_id"], df.loc[miss, "slot"]))
    fill = pd.Series([cap_slot.get(k, np.nan) for k in keys], index=df.index[miss])
    fill = fill.fillna(df.loc[miss, "route_id"].map(cap_route))
    df.loc[miss, "capacity_seats"] = fill
    rep["capacity_filled"] = int(miss.sum())

    # 4. impossible passenger counts
    impossible = (df["passengers"] < 0) | (df["passengers"] > df["capacity_seats"])
    rep["impossible_passengers_set_nan"] = int(impossible.sum())
    df.loc[impossible, "passengers"] = np.nan

    # 5. valid target flag
    df["valid_target"] = df["passengers"].notna()
    rep["rows_without_valid_target"] = int((~df["valid_target"]).sum())

    # 6. outlier flag (robust z-score inside route+slot+weekday group) - flagged, never dropped
    g = df.groupby(["route_id", "slot", df["date"].dt.dayofweek])["passengers"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median()).replace(0, np.nan)
    rz = 0.6745 * (df["passengers"] - med) / mad
    df["outlier_flag"] = (rz.abs() > 3.5).fillna(False)
    rep["outliers_flagged_not_removed"] = int(df["outlier_flag"].sum())

    # 7. censoring flag
    df["sold_out"] = (df["passengers"] == df["capacity_seats"]) & df["valid_target"]
    rep["sold_out_share"] = round(float(df.loc[df.valid_target, "sold_out"].mean()), 4)

    df["occupancy"] = df["passengers"] / df["capacity_seats"]
    rep["rows_clean"] = int(len(df))
    rep["date_range"] = [str(df["date"].min().date()), str(df["date"].max().date())]
    rep["missing_after"] = {c: int(df[c].isna().sum()) for c in ["distance_km", "capacity_seats", "passengers"]}
    keep = ["trip_id", "route_id", "origin", "destination", "distance_km", "departure_datetime", "date", "slot",
            "capacity_seats", "passengers", "occupancy", "valid_target", "outlier_flag", "sold_out"]
    return df[keep], rep


def run() -> dict:
    raw = pd.read_csv(C.RAW_TRIPS)
    clean, rep = clean_trips(raw)
    C.DATA_PROC.mkdir(parents=True, exist_ok=True)
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    clean.to_csv(C.CLEAN_TRIPS, index=False)
    C.QUALITY_JSON.write_text(json.dumps(rep, indent=2))
    return rep


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
