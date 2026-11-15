"""Candidate models, hyper-parameter grids and prediction-interval calibration."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config as C
from .features import CAT_FEATURES, NUM_FEATURES


def make_preprocessor(scale: bool = False) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
    ])


def make_model(name: str, **params) -> Pipeline:
    if name == "ridge":
        est, scale = Ridge(random_state=C.SEED, **params), True
    elif name == "random_forest":
        est, scale = RandomForestRegressor(n_estimators=80, max_samples=0.35, n_jobs=-1, random_state=C.SEED, **params), False
    elif name == "hist_gradient_boosting":
        est, scale = HistGradientBoostingRegressor(max_iter=250, random_state=C.SEED, **params), False
    else:
        raise ValueError(name)
    return Pipeline([("prep", make_preprocessor(scale)), ("model", est)])


def _grid(**kw):
    keys = list(kw)
    return [dict(zip(keys, v)) for v in itertools.product(*kw.values())]


# Small grids on purpose: tuned on the VALIDATION period only, never on test.
GRIDS = {
    "ridge": _grid(alpha=[0.1, 1.0, 10.0, 100.0]),
    "random_forest": _grid(min_samples_leaf=[5, 20], max_features=[0.5]),
    "hist_gradient_boosting": _grid(loss=["squared_error", "poisson"], learning_rate=[0.05, 0.1], max_leaf_nodes=[15, 31]),
}


def clip_to_capacity(pred, capacity):
    return np.clip(np.asarray(pred, float), 0, np.asarray(capacity, float))


# ----------------------------------------------------- uncertainty (split conformal)
def conformal_widths(y_val, pred_val, is_peak_val, levels=(0.8, 0.9)) -> dict:
    """Half-widths of prediction intervals from absolute validation residuals, separately for peak/off-peak.

    width = the ceil((n+1)*level)/n empirical quantile of |y - pred| (split-conformal).
    """
    res = np.abs(np.asarray(y_val, float) - np.asarray(pred_val, float))
    peak = np.asarray(is_peak_val).astype(int)
    out = {}
    for lv in levels:
        w = {}
        for grp in (0, 1):
            r = res[peak == grp]
            q = min(1.0, math.ceil((len(r) + 1) * lv) / len(r))
            w[str(grp)] = float(np.quantile(r, q))
        out[str(lv)] = w
    return out


def interval(pred, capacity, is_peak, widths: dict, level: float = C.INTERVAL_LEVEL, scale: float = 1.0):
    w = np.where(np.asarray(is_peak).astype(int) == 1, widths[str(level)]["1"], widths[str(level)]["0"]) * scale
    pred = np.asarray(pred, float)
    return np.clip(pred - w, 0, capacity), np.clip(pred + w, 0, capacity)
