"""Leakage-safe feature engineering - used IDENTICALLY by training and by the API.

Forecast framing
----------------
* Prediction unit : one scheduled trip (route + departure date/time)
* Target          : passengers boarded (<= seat capacity)
* Horizon         : forecast issued at 00:00, HORIZON_DAYS (=7) days before departure date
* Information set : only trips that departed on or before  departure_date - (HORIZON_DAYS + 1)

History features are looked up through `HistoryIndex.history_features`, which computes a
`cutoff` date and never reads anything after it. A unit test proves this by corrupting all
data after the cutoff and checking the features do not change.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import config as C
from .holidays_cal import holiday_table

NUM_FEATURES = [
    "hour_decimal", "is_weekend", "is_peak_hour",
    "is_holiday", "is_major_festival", "days_to_next_holiday", "days_since_prev_holiday", "next_holiday_major",
    "doy_sin", "doy_cos",
    "distance_km", "capacity_seats",
    "lag1", "lag2", "lag3", "lag4", "lag_mean4", "lag_occ_mean4",
    "route_mean_28d", "route_occ_28d", "n_lags_available",
]
CAT_FEATURES = ["route_id", "dow", "season"]
FEATURES = NUM_FEATURES + CAT_FEATURES

# columns that must NEVER be used as inputs (would leak the answer or are unknown at forecast time)
FORBIDDEN_FEATURES = ["passengers", "occupancy", "sold_out", "valid_target", "outlier_flag",
                      "rain", "weather", "disruption", "demand", "trip_id"]

_EPOCH = dt.date(1970, 1, 1).toordinal()


def _season(month: pd.Series) -> pd.Series:
    return month.map(lambda m: "winter" if m in (12, 1, 2) else "summer" if m in (3, 4, 5, 6)
                     else "monsoon" if m in (7, 8, 9) else "post_monsoon")


def calendar_features(departure: pd.Series) -> pd.DataFrame:
    """Features derived only from the scheduled departure timestamp and the public holiday calendar."""
    d = pd.to_datetime(departure).reset_index(drop=True)
    hour = d.dt.hour + d.dt.minute / 60.0
    out = pd.DataFrame({
        "hour_decimal": hour,
        "dow": d.dt.dayofweek.astype(str),
        "is_weekend": (d.dt.dayofweek >= 5).astype(int),
        "is_peak_hour": d.dt.hour.isin(C.PEAK_HOURS).astype(int),
        "season": _season(d.dt.month),
        "doy_sin": np.sin(2 * np.pi * d.dt.dayofyear / 365.25),
        "doy_cos": np.cos(2 * np.pi * d.dt.dayofyear / 365.25),
    })
    return pd.concat([out, holiday_table(d)], axis=1)


class HistoryIndex:
    """Fast lookup of already-observed trips. Keys use integer day numbers."""

    def __init__(self, hist: pd.DataFrame):
        h = hist[hist["passengers"].notna()]
        days = (pd.to_datetime(h["date"]).values.astype("datetime64[D]").astype(int) + _EPOCH).tolist()
        pax = h["passengers"].astype(float).tolist()
        cap = h["capacity_seats"].astype(float).tolist()
        self.trip: dict = {}
        self.route_day: dict = {}
        for r, s, d, p, c in zip(h["route_id"].tolist(), h["slot"].tolist(), days, pax, cap):
            self.trip[(r, s, d)] = (p, c)
            a = self.route_day.get((r, d), (0.0, 0.0, 0))
            self.route_day[(r, d)] = (a[0] + p, a[1] + c, a[2] + 1)
        self.max_day = max(days) if days else None
        self.min_day = min(days) if days else None

    @staticmethod
    def day_number(date) -> int:
        return pd.Timestamp(date).date().toordinal()

    def cutoff_day(self, target_day: int, horizon: int = C.HORIZON_DAYS) -> int:
        """Last day whose trips are allowed to be read."""
        cutoff = target_day - (horizon + 1)
        if self.max_day is not None:
            cutoff = min(cutoff, self.max_day)
        return cutoff

    def history_features(self, route_id: str, slot: str, target_day: int, horizon: int = C.HORIZON_DAYS) -> dict:
        cutoff = self.cutoff_day(target_day, horizon)
        # latest date <= cutoff that falls on the same weekday as the target
        back = ((cutoff + 6) % 7 - (target_day + 6) % 7) % 7
        anchor = cutoff - back
        lags, occs = [], []
        for k in range(4):
            v = self.trip.get((route_id, slot, anchor - 7 * k))
            lags.append(v[0] if v else np.nan)
            occs.append(v[0] / v[1] if v else np.nan)
        s_p = s_c = 0.0
        n = 0
        for d in range(cutoff - 27, cutoff + 1):
            v = self.route_day.get((route_id, d))
            if v:
                s_p += v[0]; s_c += v[1]; n += v[2]
        avail = int(np.sum(~np.isnan(lags)))
        return {
            "lag1": lags[0], "lag2": lags[1], "lag3": lags[2], "lag4": lags[3],
            "lag_mean4": float(np.nanmean(lags)) if avail else np.nan,
            "lag_occ_mean4": float(np.nanmean(occs)) if avail else np.nan,
            "route_mean_28d": s_p / n if n else np.nan,
            "route_occ_28d": s_p / s_c if s_c else np.nan,
            "n_lags_available": avail,
            "_cutoff_day": cutoff,
        }

    def last_observed_date(self) -> pd.Timestamp:
        return pd.Timestamp(dt.date.fromordinal(self.max_day))


def build_features(df: pd.DataFrame, index: HistoryIndex, horizon: int = C.HORIZON_DAYS) -> pd.DataFrame:
    """df needs: route_id, departure_datetime, distance_km, capacity_seats. Returns FEATURES (+ helper cols)."""
    base = df.reset_index(drop=True)
    cal = calendar_features(base["departure_datetime"])
    dep = pd.to_datetime(base["departure_datetime"])
    slots = dep.dt.strftime("%H:%M").tolist()
    days = (dep.dt.normalize().values.astype("datetime64[D]").astype(int) + _EPOCH).tolist()
    rows = [index.history_features(r, s, d, horizon) for r, s, d in zip(base["route_id"].tolist(), slots, days)]
    hist = pd.DataFrame(rows)
    X = pd.concat([base[["route_id", "distance_km", "capacity_seats"]].astype({"distance_km": float, "capacity_seats": float}),
                   cal, hist], axis=1)
    return X
