"""Automated leakage / sensitive-data audit (acceptance test #5).

Run directly:  python -m busdemand.leakage_audit
Also called by the training pipeline and by tests/test_leakage.py.

Checks
  1. no forbidden / target-derived column is a model input
  2. chronological split: train < validation < test, no overlap
  3. history features only read data on or before the cutoff (future values are corrupted and
     the features must not change)
  4. no personal / sensitive columns in the dataset (only aggregate trip-level counts)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config as C
from .evaluation import chronological_masks
from .features import FEATURES, FORBIDDEN_FEATURES, HistoryIndex, build_features

SENSITIVE_HINTS = ["name", "phone", "email", "aadhaar", "passenger_id", "card", "address", "gender", "age"]


def check_forbidden_inputs() -> dict:
    bad = [f for f in FEATURES if any(f == b or f.startswith(b) for b in FORBIDDEN_FEATURES)]
    return {"passed": not bad, "violations": bad, "n_features": len(FEATURES)}


def check_split(dates: pd.Series) -> dict:
    tr, va, te = chronological_masks(dates)
    d = pd.to_datetime(dates).reset_index(drop=True)
    ok_overlap = not ((tr & va).any() or (tr & te).any() or (va & te).any())
    ok_order = d[tr].max() < d[va].min() and d[va].max() < d[te].min()
    return {"passed": bool(ok_overlap and ok_order),
            "train": [str(d[tr].min().date()), str(d[tr].max().date())],
            "validation": [str(d[va].min().date()), str(d[va].max().date())],
            "test": [str(d[te].min().date()), str(d[te].max().date())],
            "rows": {"train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum())}}


def check_future_invariance(clean: pd.DataFrame, n_samples: int = 25, seed: int = C.SEED) -> dict:
    """Corrupt every observation after each sample's cutoff; its features must stay identical."""
    rng = np.random.default_rng(seed)
    ok = clean[clean["valid_target"]]
    sample = ok.iloc[rng.choice(len(ok), size=n_samples, replace=False)]
    idx_true = HistoryIndex(clean)
    failures = 0
    for _, row in sample.iterrows():
        target_day = HistoryIndex.day_number(row["date"])
        cutoff = idx_true.cutoff_day(target_day)
        corrupted = clean.copy()
        days = pd.to_datetime(corrupted["date"]).dt.date.map(lambda d: d.toordinal())
        mask = days > cutoff
        corrupted.loc[mask, "passengers"] = 9999.0
        f_a = idx_true.history_features(row["route_id"], row["slot"], target_day)
        f_b = HistoryIndex(corrupted).history_features(row["route_id"], row["slot"], target_day)
        if any(not (f_a[k] == f_b[k] or (np.isnan(f_a[k]) and np.isnan(f_b[k]))) for k in f_a):
            failures += 1
    return {"passed": failures == 0, "samples_checked": n_samples, "failures": failures}


def check_no_sensitive_columns(columns) -> dict:
    hits = [c for c in columns if any(h in c.lower() for h in SENSITIVE_HINTS)]
    return {"passed": not hits, "sensitive_like_columns": hits, "columns": list(columns)}


def run(clean: pd.DataFrame | None = None, write: bool = True) -> dict:
    if clean is None:
        clean = pd.read_csv(C.CLEAN_TRIPS, parse_dates=["departure_datetime", "date"])
    modelled = clean[clean["valid_target"]]
    res = {
        "forbidden_inputs": check_forbidden_inputs(),
        "chronological_split": check_split(modelled["date"]),
        "future_invariance": check_future_invariance(clean),
        "sensitive_data": check_no_sensitive_columns(clean.columns),
    }
    res["all_passed"] = all(v["passed"] for v in res.values() if isinstance(v, dict))
    if write:
        C.REPORTS.mkdir(parents=True, exist_ok=True)
        (C.REPORTS / "leakage_audit.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    out = run()
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["all_passed"] else 1)
