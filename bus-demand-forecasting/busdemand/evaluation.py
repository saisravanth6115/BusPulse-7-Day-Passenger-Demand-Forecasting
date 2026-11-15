"""Chronological splitting, metrics and the historical route/time baselines."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


# ----------------------------------------------------------------- splits
def chronological_masks(dates: pd.Series, train_end=C.TRAIN_END, val_end=C.VAL_END):
    """Boolean masks for train / validation / test. No shuffling, no overlap."""
    d = pd.to_datetime(dates).reset_index(drop=True)
    train = d <= pd.Timestamp(train_end)
    val = (d > pd.Timestamp(train_end)) & (d <= pd.Timestamp(val_end))
    test = d > pd.Timestamp(val_end)
    return train.values, val.values, test.values


# ----------------------------------------------------------------- metrics
def wape(y, p) -> float:
    """Weighted absolute percentage error = sum|y-p| / sum|y|. Robust when some y are small."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.abs(y - p).sum() / np.abs(y).sum())


def regression_metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "WAPE": wape(y, p),
        "Bias": float(np.mean(p - y)),          # >0 = over-forecast, <0 = under-forecast
        "n": int(len(y)),
    }


def high_demand_flag_metrics(y, p, cap, actual_thr=C.HIGH_OCC, pred_thr=C.HIGH_OCC) -> dict:
    """Capacity-aware business metric.

    Event to catch  : actual occupancy >= actual_thr (bus (nearly) full -> passengers may be turned away).
    Flag raised when: predicted occupancy >= pred_thr.
    precision = of flagged trips, how many really filled up; recall = of trips that filled up, how many we flagged.
    """
    y, p, cap = (np.asarray(a, float) for a in (y, p, cap))
    actual = (y / cap) >= actual_thr
    pred = (p / cap) >= pred_thr
    tp = int((actual & pred).sum()); fp = int((~actual & pred).sum()); fn = int((actual & ~pred).sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else float("nan")
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn, "actual_positive": int(actual.sum()),
            "flagged": int(pred.sum())}


def breakdown(df: pd.DataFrame, by: str, y="y", p="pred") -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by, observed=True):
        m = regression_metrics(g[y], g[p])
        m[by] = key
        rows.append(m)
    return pd.DataFrame(rows).set_index(by)[["n", "MAE", "RMSE", "WAPE", "Bias"]]


# ----------------------------------------------------------------- baselines
class RouteTimeBaseline:
    """Historical average by route x departure slot x weekday (fallbacks: route x slot, then route).

    This is exactly what a planner would do with a spreadsheet, so any ML model must beat it.
    """

    def fit(self, df: pd.DataFrame):
        d = df.copy()
        d["dow"] = pd.to_datetime(d["departure_datetime"]).dt.dayofweek
        self.m_full = d.groupby(["route_id", "slot", "dow"])["passengers"].mean().to_dict()
        self.m_slot = d.groupby(["route_id", "slot"])["passengers"].mean().to_dict()
        self.m_route = d.groupby("route_id")["passengers"].mean().to_dict()
        self.global_mean = float(d["passengers"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        dow = pd.to_datetime(df["departure_datetime"]).dt.dayofweek
        slot = pd.to_datetime(df["departure_datetime"]).dt.strftime("%H:%M")
        out = []
        for r, s, w in zip(df["route_id"], slot, dow):
            out.append(self.m_full.get((r, s, w), self.m_slot.get((r, s), self.m_route.get(r, self.global_mean))))
        return np.asarray(out, float)


def seasonal_naive(X: pd.DataFrame) -> np.ndarray:
    """Second baseline: mean of the same slot on the last 4 same-weekdays that were observable at forecast time."""
    p = X["lag_mean4"].copy()
    p = p.fillna(X["route_mean_28d"])
    return p.fillna(p.mean()).values
