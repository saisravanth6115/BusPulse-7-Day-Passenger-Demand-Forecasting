"""Model explanations that a reviewer (and you) can understand in two minutes.

Global : permutation importance on the held-out test set
         (how much does MAE get worse if one input column is shuffled?).
Local  : "replace-by-typical-value" attribution for ONE prediction
         (how many passengers does the forecast change if this input is replaced by the
          training-set typical value?). It is an approximation (no interaction handling),
          deliberately simple and fully transparent.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from . import config as C
from .features import CAT_FEATURES, FEATURES, NUM_FEATURES

FRIENDLY = {
    "lag1": "same-slot demand 2 weeks ago", "lag2": "same-slot demand 3 weeks ago",
    "lag3": "same-slot demand 4 weeks ago", "lag4": "same-slot demand 5 weeks ago",
    "lag_mean4": "average same-slot demand (last 4 same weekdays)",
    "lag_occ_mean4": "average same-slot occupancy (last 4 same weekdays)",
    "route_mean_28d": "route's average demand, last 28 days",
    "route_occ_28d": "route's average occupancy, last 28 days",
    "n_lags_available": "number of usable history points",
    "hour_decimal": "departure time of day", "is_peak_hour": "peak-hour departure",
    "is_weekend": "weekend", "dow": "day of week", "season": "season",
    "is_holiday": "public holiday", "is_major_festival": "major festival day",
    "days_to_next_holiday": "days until next holiday", "days_since_prev_holiday": "days since last holiday",
    "next_holiday_major": "next holiday is a major festival",
    "doy_sin": "time of year (sin)", "doy_cos": "time of year (cos)",
    "distance_km": "route distance", "capacity_seats": "bus capacity (seats)", "route_id": "route",
}


def reference_row(X_train: pd.DataFrame) -> dict:
    ref = {}
    for c in NUM_FEATURES:
        ref[c] = float(X_train[c].median())
    for c in CAT_FEATURES:
        ref[c] = X_train[c].mode().iloc[0]
    return ref


def global_importance(pipeline, X_test: pd.DataFrame, y_test, n_repeats: int = 3, max_rows: int = 6000) -> pd.DataFrame:
    rng = np.random.default_rng(C.SEED)
    idx = rng.choice(len(X_test), size=min(max_rows, len(X_test)), replace=False)
    r = permutation_importance(pipeline, X_test.iloc[idx][FEATURES], np.asarray(y_test)[idx], n_repeats=n_repeats,
                               scoring="neg_mean_absolute_error", random_state=C.SEED, n_jobs=1)
    df = pd.DataFrame({"feature": FEATURES, "mae_increase": r.importances_mean, "std": r.importances_std})
    df["meaning"] = df["feature"].map(FRIENDLY)
    return df.sort_values("mae_increase", ascending=False).reset_index(drop=True)


def explain_prediction(pipeline, x_row: pd.DataFrame, ref: dict, top_k: int = 6) -> list[dict]:
    """x_row: single-row DataFrame with FEATURES. Returns the top_k drivers (signed passenger impact)."""
    base = x_row[FEATURES].iloc[[0]]
    variants = [base]
    for f in FEATURES:
        v = base.copy()
        v[f] = ref[f]
        variants.append(v)
    preds = pipeline.predict(pd.concat(variants, ignore_index=True))
    full = float(preds[0])
    out = []
    for f, p in zip(FEATURES, preds[1:]):
        out.append({"feature": f, "meaning": FRIENDLY.get(f, f), "value": _py(base[f].iloc[0]),
                    "typical_value": _py(ref[f]), "impact_passengers": round(full - float(p), 2)})
    out.sort(key=lambda d: abs(d["impact_passengers"]), reverse=True)
    return out[:top_k]


def _py(v):
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 3)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v
