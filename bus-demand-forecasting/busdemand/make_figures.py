"""All figures used in the report and notebooks. Run automatically at the end of training.

    python -m busdemand.make_figures
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})
COL = {"baseline_route_time": "#9aa0a6", "baseline_seasonal_naive": "#c0c4c8", "ridge": "#4c78a8",
       "random_forest": "#59a14f", "hist_gradient_boosting": "#e15759"}
LABEL = {"baseline_route_time": "Baseline: route/time avg", "baseline_seasonal_naive": "Baseline: seasonal naive",
         "ridge": "Ridge", "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting"}


def _save(name):
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(C.FIGURES / name, bbox_inches="tight")
    plt.close()


def eda_figures(clean: pd.DataFrame):
    ok = clean[clean["valid_target"]].copy()
    ok["hour"] = ok["departure_datetime"].dt.hour
    ok["dow"] = ok["date"].dt.dayofweek
    ok["month"] = ok["date"].dt.month

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].hist(ok["passengers"], bins=50, color="#4c78a8")
    ax[0].set(title="Passengers per trip", xlabel="passengers", ylabel="trips")
    ax[1].hist(ok["occupancy"], bins=40, color="#59a14f")
    ax[1].axvline(C.HIGH_OCC, color="r", ls="--", label="high-demand flag (90%)")
    ax[1].set(title="Seat occupancy (note the spike at 100% = sold out / censored)", xlabel="occupancy")
    ax[1].legend()
    _save("eda_target_distribution.png")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ok.groupby("hour")["passengers"].mean().plot(ax=ax[0], marker="o", color="#4c78a8")
    ax[0].set(title="Mean passengers by departure hour", xlabel="hour")
    ok.groupby("dow")["passengers"].mean().plot(kind="bar", ax=ax[1], color="#f28e2b")
    ax[1].set(title="By weekday (0=Mon)", xlabel="")
    ok.groupby("month")["passengers"].mean().plot(ax=ax[2], marker="o", color="#e15759")
    ax[2].set(title="By month (seasonality)", xlabel="month")
    _save("eda_patterns.png")

    piv = ok.pivot_table(index="route_id", columns="dow", values="passengers", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.imshow(piv.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax.set_yticks(range(len(piv)), piv.index)
    ax.set_title("Mean passengers: route x weekday")
    ax.grid(False)
    plt.colorbar(im, ax=ax)
    _save("eda_route_weekday_heatmap.png")

    daily = ok.groupby("date")["passengers"].sum()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    daily.plot(ax=ax, color="#4c78a8", lw=0.7)
    daily.rolling(28, center=True).mean().plot(ax=ax, color="#e15759", lw=2, label="28-day mean")
    ax.set(title="Total daily passengers (all routes): trend, seasonality, festival spikes", xlabel="", ylabel="passengers/day")
    ax.legend()
    _save("eda_daily_total.png")

    q = json.loads(C.QUALITY_JSON.read_text())
    items = {"duplicates": q["duplicates_dropped"], "missing capacity": q["missing_before"]["capacity_seats"],
             "missing distance": q["missing_before"]["distance_km"], "missing passengers": q["missing_before"]["passengers"],
             "impossible passengers": q["impossible_passengers_set_nan"], "outliers (flagged, kept)": q["outliers_flagged_not_removed"]}
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(list(items), list(items.values()), color="#b07aa1")
    ax.set(title="Data-quality issues found in the raw export (rows)")
    for i, v in enumerate(items.values()):
        ax.text(v, i, f" {v}", va="center")
    _save("eda_data_quality.png")


def result_figures(m: dict, preds: pd.DataFrame):
    methods = list(LABEL)
    # overall metrics
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for a, met in zip(ax, ["MAE", "RMSE", "WAPE"]):
        vals = [m["test"][k][met] for k in methods]
        a.bar([LABEL[k] for k in methods], vals, color=[COL[k] for k in methods])
        a.set_title(f"Test {met} (lower is better)")
        plt.setp(a.get_xticklabels(), rotation=35, ha="right")
        for i, v in enumerate(vals):
            a.text(i, v, f"{v:.3f}" if met != "WAPE" else f"{v:.1%}", ha="center", va="bottom", fontsize=8)
    _save("res_overall_metrics.png")

    # per route WAPE: baseline vs best
    best = m["best_model"]
    routes = sorted(m["by_route"][best])
    x = np.arange(len(routes)); w = 0.38
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.bar(x - w / 2, [m["by_route"]["baseline_route_time"][r]["WAPE"] for r in routes], w, label=LABEL["baseline_route_time"], color=COL["baseline_route_time"])
    ax.bar(x + w / 2, [m["by_route"][best][r]["WAPE"] for r in routes], w, label=LABEL[best], color=COL[best])
    ax.set_xticks(x, routes)
    ax.set(title="Test WAPE by route", ylabel="WAPE")
    ax.legend()
    _save("res_wape_by_route.png")

    # peak / off-peak
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    groups = ["off-peak", "peak"]
    x = np.arange(2); w = 0.16
    for i, k in enumerate(methods):
        ax.bar(x + (i - 2) * w, [m["by_peak"][k][g]["MAE"] for g in groups], w, label=LABEL[k], color=COL[k])
    ax.set_xticks(x, groups)
    ax.set(title="Test MAE: peak vs off-peak departures", ylabel="MAE (passengers)")
    ax.legend(fontsize=7)
    _save("res_peak_offpeak.png")

    # actual vs predicted
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, col, t in [(ax[0], "pred_baseline_route_time", "Baseline"), (ax[1], "pred", f"Best model ({LABEL[best]})")]:
        a.scatter(preds["y"], preds[col], s=4, alpha=0.3, color="#4c78a8")
        lim = [0, max(preds["y"].max(), preds[col].max())]
        a.plot(lim, lim, "r--", lw=1)
        a.set(title=t, xlabel="actual passengers", ylabel="predicted passengers")
    _save("res_actual_vs_pred.png")

    # time series for one route / slot across the test period
    rid = preds.groupby("route_id")["y"].sum().idxmax()
    slot = preds[preds.route_id == rid].groupby("slot")["y"].sum().idxmax()
    g = preds[(preds.route_id == rid) & (preds.slot == slot)].sort_values("date")
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(g["date"], g["y"], color="k", lw=1.2, label="actual")
    ax.plot(g["date"], g["pred"], color=COL[best], lw=1.6, label="model forecast")
    ax.plot(g["date"], g["pred_baseline_route_time"], color="#9aa0a6", lw=1.2, ls="--", label="baseline")
    ax.fill_between(g["date"], g["pi80_low"], g["pi80_high"], color=COL[best], alpha=0.15, label="80% interval")
    ax.set(title=f"Route {rid}, {slot} departure - test period (Oct-Dec)", ylabel="passengers")
    ax.legend(ncol=4, fontsize=8)
    _save("res_timeseries_example.png")

    # residuals + disruption effect
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    err = preds["pred"] - preds["y"]
    ax[0].hist(err, bins=50, color="#4c78a8")
    ax[0].axvline(0, color="r")
    ax[0].set(title="Forecast error (pred - actual)", xlabel="passengers")
    dd = preds.groupby("disruption_label")["abs_err"].mean()
    ax[1].bar(dd.index, dd.values, color=["#59a14f", "#e15759"][: len(dd)])
    ax[1].set(title="Mean absolute error: normal vs disruption days", ylabel="MAE")
    _save("res_errors.png")

    # importance
    imp = pd.DataFrame(m["importance"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.barh(imp["meaning"], imp["mae_increase"], color="#f28e2b")
    ax.set(title="Permutation importance (test MAE increase when shuffled)", xlabel="MAE increase")
    _save("res_importance.png")

    # tuning
    tl = pd.DataFrame(m["tuning_log"])
    fig, ax = plt.subplots(figsize=(7, 3.4))
    best_by = tl.groupby("model")["val_MAE"].min()
    ax.bar([LABEL[k] for k in best_by.index], best_by.values, color=[COL[k] for k in best_by.index])
    ax.set(title="Validation MAE of tuned models (model selection)", ylabel="MAE")
    _save("res_validation_tuning.png")


def architecture_figure():
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
    def box(x, y, w, h, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=color, ec="#333", lw=1))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.5)
    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))
    # row 1: offline pipeline
    box(0.2, 3.9, 1.9, 0.9, "data_generator.py\nsynthetic trips\n(seeded)", "#dbe9f6")
    box(2.5, 3.9, 1.9, 0.9, "data_quality.py\nclean + QA report", "#dbe9f6")
    box(4.8, 3.9, 2.2, 0.9, "features.py\nlag + calendar features\n(7-day horizon, no leakage)", "#fde9c9")
    box(7.4, 3.9, 2.0, 0.9, "train.py\nbaselines + 3 models\nchronological split", "#dff0d8")
    box(9.8, 3.9, 2.0, 0.9, "models/\ndemand_model.joblib\nhistory.csv.gz", "#f8d7da")
    for a, b in [(2.1, 2.5), (4.4, 4.8), (7.0, 7.4), (9.4, 9.8)]:
        arrow(a, 4.35, b, 4.35)
    # row 2: serving
    box(9.8, 2.2, 2.0, 0.9, "predict.py\nDemandPredictor\n(same feature code)", "#fde9c9")
    arrow(10.8, 3.9, 10.8, 3.1)
    box(6.6, 0.7, 2.4, 0.9, "api/main.py (FastAPI)\n/predict /routes /model-info", "#e8dff5")
    box(9.5, 0.7, 2.4, 0.9, "app/streamlit_app.py\ninteractive planner UI", "#e8dff5")
    arrow(10.3, 2.2, 8.2, 1.6); arrow(11.0, 2.2, 10.9, 1.6)
    # row 2 left: evidence
    box(0.2, 2.2, 2.4, 0.9, "leakage_audit.py\n+ 30 pytest tests", "#fff3cd")
    box(3.0, 2.2, 3.2, 0.9, "reports/: metrics.json, figures,\ntest_predictions.csv, report", "#fff3cd")
    arrow(8.4, 3.9, 5.2, 3.1)
    box(0.2, 0.7, 6.0, 0.9, "Planner (human) reviews forecast, interval, capacity flag and warnings, then decides.\nThe tool never changes a schedule by itself.", "#f1f1f1")
    arrow(6.6, 1.15, 6.25, 1.15)
    ax.set_title("System architecture", fontsize=11)
    _save("architecture.png")


def main():
    clean = pd.read_csv(C.CLEAN_TRIPS, parse_dates=["departure_datetime", "date"])
    eda_figures(clean)
    architecture_figure()
    if C.METRICS_JSON.exists() and C.TEST_PREDS.exists():
        m = json.loads(C.METRICS_JSON.read_text())
        preds = pd.read_csv(C.TEST_PREDS, parse_dates=["date", "departure_datetime"])
        result_figures(m, preds)
    print("figures written to", C.FIGURES)


if __name__ == "__main__":
    main()
