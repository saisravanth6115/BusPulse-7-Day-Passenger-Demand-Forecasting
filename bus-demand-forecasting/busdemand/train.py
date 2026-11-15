"""End-to-end, repeatable training pipeline.

    python -m busdemand.train

Stages
  1. clean raw export + data-quality report
  2. leakage-safe features (forecast issued 7 days before departure)
  3. chronological split: train / validation (tuning) / test (held out, used once)
  4. baselines (route/time average, seasonal naive) + 3 candidate models, tuned on VALIDATION only
  5. refit on train+validation, evaluate ONCE on test, error analysis (route, peak/off-peak, ...)
  6. prediction intervals (split conformal), explanations, leakage audit
  7. save artifact + metrics + figures
"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd

from . import config as C
from . import data_quality, explain, leakage_audit
from .evaluation import (RouteTimeBaseline, breakdown, chronological_masks, high_demand_flag_metrics,
                         regression_metrics, seasonal_naive)
from .features import FEATURES, HistoryIndex, build_features
from .models import GRIDS, clip_to_capacity, conformal_widths, interval, make_model

METHODS = ["baseline_route_time", "baseline_seasonal_naive", "ridge", "random_forest", "hist_gradient_boosting"]


def mark_disruptions(df: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    """Ground-truth disruption flag (synthetic generator knowledge). Used ONLY for error analysis."""
    flag = pd.Series(False, index=df.index)
    for _, e in events.iterrows():
        flag |= (df["route_id"] == e.route_id) & (df["date"] >= pd.Timestamp(e.start_date)) & (df["date"] <= pd.Timestamp(e.end_date))
    return flag


def _tbl(df: pd.DataFrame) -> dict:
    return json.loads(df.round(4).to_json(orient="index"))


def main():
    t0 = time.time()
    C.MODELS.mkdir(exist_ok=True)
    C.REPORTS.mkdir(exist_ok=True)
    C.FIGURES.mkdir(parents=True, exist_ok=True)

    print("[1/7] cleaning raw data ...")
    quality = data_quality.run()
    clean = pd.read_csv(C.CLEAN_TRIPS, parse_dates=["departure_datetime", "date"])
    routes = pd.read_csv(C.ROUTES_CSV)
    events = pd.read_csv(C.EVENTS_CSV, parse_dates=["start_date", "end_date"])

    print("[2/7] building leakage-safe features ...")
    index = HistoryIndex(clean)
    start = clean["date"].min()
    model_df = clean[clean["valid_target"] & (clean["date"] >= start + pd.Timedelta(days=C.WARMUP_DAYS))].reset_index(drop=True)
    X = build_features(model_df, index)
    y = model_df["passengers"].values.astype(float)
    cap = model_df["capacity_seats"].values.astype(float)
    peak = X["is_peak_hour"].values

    print("[3/7] chronological split ...")
    tr, va, te = chronological_masks(model_df["date"])
    trva = tr | va
    print(f"      train={tr.sum():,}  validation={va.sum():,}  test={te.sum():,}")

    print("[4/7] baselines + tuning on validation ...")
    val_pred, val_metrics, tuning_log, best_params = {}, {}, [], {}
    bl = RouteTimeBaseline().fit(model_df[tr])
    val_pred["baseline_route_time"] = clip_to_capacity(bl.predict(model_df[va]), cap[va])
    val_pred["baseline_seasonal_naive"] = clip_to_capacity(seasonal_naive(X[va]), cap[va])
    for family, grid in GRIDS.items():
        best = (np.inf, None, None)
        for params in grid:
            m = make_model(family, **params).fit(X.loc[tr, FEATURES], y[tr])
            p = clip_to_capacity(m.predict(X.loc[va, FEATURES]), cap[va])
            mae = regression_metrics(y[va], p)["MAE"]
            tuning_log.append({"model": family, "params": params, "val_MAE": round(mae, 4)})
            print(f"      {family:24s} {params}  val MAE={mae:.3f}")
            if mae < best[0]:
                best = (mae, params, p)
        best_params[family] = best[1]
        val_pred[family] = best[2]
    for k in METHODS:
        val_metrics[k] = regression_metrics(y[va], val_pred[k])
    best_family = min(GRIDS, key=lambda f: val_metrics[f]["MAE"])
    print(f"      selected on validation MAE: {best_family} {best_params[best_family]}")

    print("[5/7] refit on train+validation, evaluate ONCE on test ...")
    test_pred, fitted = {}, {}
    bl_final = RouteTimeBaseline().fit(model_df[trva])
    test_pred["baseline_route_time"] = clip_to_capacity(bl_final.predict(model_df[te]), cap[te])
    test_pred["baseline_seasonal_naive"] = clip_to_capacity(seasonal_naive(X[te]), cap[te])
    for family in GRIDS:
        m = make_model(family, **best_params[family]).fit(X.loc[trva, FEATURES], y[trva])
        fitted[family] = m
        test_pred[family] = clip_to_capacity(m.predict(X.loc[te, FEATURES]), cap[te])
    test_metrics = {k: regression_metrics(y[te], test_pred[k]) for k in METHODS}
    flag_metrics = {k: {"HIGH (pred occ>=90%)": high_demand_flag_metrics(y[te], test_pred[k], cap[te], C.HIGH_OCC, C.HIGH_OCC),
                        "WATCH+HIGH (pred occ>=75%)": high_demand_flag_metrics(y[te], test_pred[k], cap[te], C.HIGH_OCC, C.WATCH_OCC)}
                    for k in METHODS}
    best_pipe = fitted[best_family]

    print("[6/7] intervals, error analysis, explanations, leakage audit ...")
    widths = conformal_widths(y[va], val_pred[best_family], peak[va])
    lo80, hi80 = interval(test_pred[best_family], cap[te], peak[te], widths, 0.8)
    lo90, hi90 = interval(test_pred[best_family], cap[te], peak[te], widths, 0.9)
    cov80 = float(np.mean((y[te] >= lo80) & (y[te] <= hi80)))
    cov90 = float(np.mean((y[te] >= lo90) & (y[te] <= hi90)))

    T = model_df[te].reset_index(drop=True)
    Xt = X[te].reset_index(drop=True)
    out = pd.concat([T[["trip_id", "route_id", "departure_datetime", "date", "slot", "capacity_seats"]],
                     Xt[["is_peak_hour", "is_holiday", "is_major_festival", "is_weekend", "lag_mean4", "n_lags_available"]]], axis=1)
    out["y"] = y[te]
    for k in METHODS:
        out[f"pred_{k}"] = test_pred[k]
    out["pred"] = test_pred[best_family]
    out["pi80_low"], out["pi80_high"] = lo80, hi80
    out["abs_err"] = (out["y"] - out["pred"]).abs()
    out["disruption"] = mark_disruptions(out, events).values
    out["peak_label"] = np.where(out["is_peak_hour"] == 1, "peak", "off-peak")
    out["day_type"] = np.where(out["is_holiday"] == 1, "holiday", np.where(out["is_weekend"] == 1, "weekend", "weekday"))
    out["disruption_label"] = np.where(out["disruption"], "disruption", "normal")
    out.to_csv(C.TEST_PREDS, index=False)

    def per_method(by):
        res = {}
        for k in METHODS:
            d = out.rename(columns={f"pred_{k}": "p"})[[by, "y", "p"]]
            res[k] = _tbl(breakdown(d, by, "y", "p"))
        return res

    importance = explain.global_importance(best_pipe, Xt, y[te])
    ref = explain.reference_row(X[tr])
    worst = int(out.loc[~out["disruption"], "abs_err"].idxmax())
    worst_dis = int(out.loc[out["disruption"], "abs_err"].idxmax()) if out["disruption"].any() else None
    typical = int((out["abs_err"] - out["abs_err"].median()).abs().idxmin())
    examples = []
    for tag, i in [("typical_prediction", typical), ("largest_error_without_disruption", worst), ("largest_error_during_disruption", worst_dis)]:
        if i is None:
            continue
        examples.append({
            "tag": tag, "trip_id": out.loc[i, "trip_id"], "route_id": out.loc[i, "route_id"],
            "departure": str(out.loc[i, "departure_datetime"]), "capacity": int(out.loc[i, "capacity_seats"]),
            "actual": float(out.loc[i, "y"]), "predicted": round(float(out.loc[i, "pred"]), 1),
            "baseline_prediction": round(float(out.loc[i, "pred_baseline_route_time"]), 1),
            "disruption_known_to_generator": bool(out.loc[i, "disruption"]),
            "drivers": explain.explain_prediction(best_pipe, Xt.iloc[[i]], ref),
        })

    audit = leakage_audit.run(clean)

    print("[7/7] saving artifact + metrics ...")
    hist = clean.loc[clean["valid_target"], ["route_id", "slot", "date", "passengers", "capacity_seats"]]
    hist.to_csv(C.MODELS / "history.csv.gz", index=False)
    routes_meta = routes.to_dict(orient="records")
    artifact = {
        "pipeline": best_pipe, "model_name": best_family, "params": best_params[best_family],
        "features": FEATURES, "interval_widths": widths, "reference_row": ref,
        "baseline": bl_final, "routes": routes_meta,
        "horizon_days": C.HORIZON_DAYS, "seed": C.SEED,
        "train_window": [str(model_df.loc[tr, "date"].min().date()), str(model_df.loc[trva, "date"].max().date())],
        "test_metrics": test_metrics[best_family], "baseline_test_metrics": test_metrics["baseline_route_time"],
    }
    joblib.dump(artifact, C.ARTIFACT, compress=3)

    base_w = test_metrics["baseline_route_time"]["WAPE"]
    best_w = test_metrics[best_family]["WAPE"]
    metrics = {
        "framing": {
            "decision": "How many buses / which bus size to schedule for each departure",
            "prediction_unit": "one scheduled trip (route + departure date/time)",
            "target": "passengers boarded (<= seat capacity)",
            "horizon_days": C.HORIZON_DAYS,
            "forecast_issued": "00:00, 7 days before departure date; only trips up to departure_date-8 are visible",
            "high_flag": C.HIGH_OCC, "watch_flag": C.WATCH_OCC, "low_flag": C.LOW_OCC,
        },
        "data": {"quality": quality, "modelling_rows": int(len(model_df)),
                 "split_rows": {"train": int(tr.sum()), "validation": int(va.sum()), "test": int(te.sum())},
                 "routes": int(routes.shape[0]), "features": FEATURES},
        "validation": val_metrics, "test": test_metrics, "test_flag_metrics": flag_metrics,
        "best_model": best_family, "best_params": best_params, "tuning_log": tuning_log,
        "improvement_vs_baseline": {"wape_baseline": base_w, "wape_best": best_w,
                                    "relative_wape_reduction_pct": round(100 * (base_w - best_w) / base_w, 1)},
        "by_route": per_method("route_id"), "by_peak": per_method("peak_label"),
        "by_day_type": per_method("day_type"), "by_disruption": per_method("disruption_label"),
        "interval": {"level_main": C.INTERVAL_LEVEL, "widths": widths,
                     "test_coverage_80": cov80, "test_coverage_90": cov90,
                     "avg_width_80": float(np.mean(hi80 - lo80))},
        "importance": json.loads(importance.head(12).round(4).to_json(orient="records")),
        "examples": examples, "leakage_audit": audit,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    C.METRICS_JSON.write_text(json.dumps(metrics, indent=2, default=float))

    try:
        from . import make_figures
        make_figures.main()
    except Exception as exc:  # figures must never break training
        print("figure generation skipped:", exc)

    print(f"\nDONE in {time.time() - t0:.0f}s. Best model: {best_family}")
    for k in METHODS:
        m = test_metrics[k]
        print(f"  TEST {k:26s} MAE={m['MAE']:.3f}  RMSE={m['RMSE']:.3f}  WAPE={m['WAPE']:.3%}")


if __name__ == "__main__":
    main()
