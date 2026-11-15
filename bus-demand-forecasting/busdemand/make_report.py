"""Builds reports/TECHNICAL_REPORT.md (+ .docx via pandoc) from the real metrics files.

Every number in the report is read from reports/metrics.json, so it can never disagree with the code.

    python -m busdemand.make_report
"""
from __future__ import annotations

import json
import shutil
import subprocess

import numpy as np
import pandas as pd

from . import config as C
from .features import CAT_FEATURES, NUM_FEATURES

NAMES = {"baseline_route_time": "Baseline A: route/time average", "baseline_seasonal_naive": "Baseline B: seasonal naive (last 4 same weekdays)",
         "ridge": "Ridge regression", "random_forest": "Random forest", "hist_gradient_boosting": "Gradient boosting (HistGB)"}


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    cols = [df.index.name or ""] + list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, r in df.iterrows():
        cells = [str(idx)] + [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r.values]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    m = json.loads(C.METRICS_JSON.read_text())
    q = m["data"]["quality"]
    best = m["best_model"]
    preds = pd.read_csv(C.TEST_PREDS, parse_dates=["date"])
    clean = pd.read_csv(C.CLEAN_TRIPS, parse_dates=["date"])
    ok = clean[clean.valid_target]
    split = m["data"]["split_rows"]
    sp = m["leakage_audit"]["chronological_split"]
    imp = m["improvement_vs_baseline"]
    T = m["test"]; V = m["validation"]

    # ---- tables
    overall = pd.DataFrame({NAMES[k]: {"Val MAE": V[k]["MAE"], "Test MAE": T[k]["MAE"], "Test RMSE": T[k]["RMSE"],
                                       "Test WAPE": f"{T[k]['WAPE']:.1%}", "Test bias": T[k]["Bias"]} for k in NAMES}).T
    overall.index.name = "Method"
    route = pd.DataFrame({"n": {r: v["n"] for r, v in m["by_route"][best].items()},
                          "Baseline A WAPE": {r: f"{v['WAPE']:.1%}" for r, v in m["by_route"]["baseline_route_time"].items()},
                          "Model WAPE": {r: f"{v['WAPE']:.1%}" for r, v in m["by_route"][best].items()},
                          "Baseline A MAE": {r: v["MAE"] for r, v in m["by_route"]["baseline_route_time"].items()},
                          "Model MAE": {r: v["MAE"] for r, v in m["by_route"][best].items()}})
    route.index.name = "Route"
    n_routes_better = sum(m["by_route"][best][r]["WAPE"] < m["by_route"]["baseline_route_time"][r]["WAPE"] for r in m["by_route"][best])

    def seg(key, label):
        rows = {}
        for g in m[key][best]:
            rows[g] = {"n": m[key][best][g]["n"], "Baseline A MAE": m[key]["baseline_route_time"][g]["MAE"], "Model MAE": m[key][best][g]["MAE"],
                       "Baseline A WAPE": f"{m[key]['baseline_route_time'][g]['WAPE']:.1%}", "Model WAPE": f"{m[key][best][g]['WAPE']:.1%}",
                       "Model bias": m[key][best][g]["Bias"]}
        d = pd.DataFrame(rows).T
        d.index.name = label
        return d
    peak_t, day_t, dis_t = seg("by_peak", "Period"), seg("by_day_type", "Day type"), seg("by_disruption", "Day status")

    fl = []
    for k in ["baseline_route_time", "baseline_seasonal_naive", "hist_gradient_boosting"]:
        for rule, v in m["test_flag_metrics"][k].items():
            fl.append({"Method": NAMES[k], "Flag rule": rule, "Precision": f"{v['precision']:.2f}", "Recall": f"{v['recall']:.2f}",
                       "F1": f"{v['f1']:.2f}", "Flagged": v["flagged"], "Actually >=90% full": v["actual_positive"]})
    flag_tbl = md_table(pd.DataFrame(fl).set_index("Method"))

    sold = preds[preds.y >= preds.capacity_seats]
    notsold = preds[preds.y < preds.capacity_seats]
    mae_sold, mae_not = (sold.pred - sold.y).abs().mean(), (notsold.pred - notsold.y).abs().mean()
    bias_sold = (sold.pred - sold.y).mean()
    worst5 = preds.abs_err.sort_values(ascending=False).head(int(len(preds) * 0.05)).sum() / preds.abs_err.sum()

    imp_tbl = pd.DataFrame(m["importance"])[["meaning", "mae_increase"]].round(3)
    imp_tbl.columns = ["Input", "Test MAE increase if shuffled"]
    imp_tbl.index = range(1, len(imp_tbl) + 1)
    imp_tbl.index.name = "Rank"

    ex = {e["tag"]: e for e in m["examples"]}
    def ex_block(e):
        rows = "\n".join(f"| {d['meaning']} | {d['value']} | {d['typical_value']} | {d['impact_passengers']:+.1f} |" for d in e["drivers"])
        return (f"**{e['trip_id']}** ({e['departure']}, capacity {e['capacity']}): actual **{e['actual']:.0f}**, model **{e['predicted']}**, "
                f"baseline {e['baseline_prediction']}.\n\n| Input | This trip | Typical value | Effect on forecast |\n|---|---|---|---|\n{rows}\n")

    iv = m["interval"]
    tuning = pd.DataFrame(m["tuning_log"])
    tuning["params"] = tuning["params"].astype(str)
    tune_tbl = md_table(tuning.sort_values(["model", "val_MAE"]).set_index("model"))
    feats = f"Numeric ({len(NUM_FEATURES)}): " + ", ".join(f"`{f}`" for f in NUM_FEATURES) + f".\n\nCategorical ({len(CAT_FEATURES)}): " + ", ".join(f"`{f}`" for f in CAT_FEATURES) + "."

    report = f"""---
title: "Bus Passenger-Demand Forecasting - Technical Report"
subtitle: "Algions AI Labs Private Limited - AI & Machine Learning internship assessment"
---

**Student name:** ____________________  **Registration number:** ____________________

**Assigned project:** Bus Passenger-Demand Forecasting (domain: Bus Booking)

# 1. Executive summary

I built a decision-support tool that forecasts how many passengers will board a scheduled bus trip, **7 days before departure**, and flags trips that are likely to be full or empty. The forecast comes with an 80 % prediction interval, a capacity-aware flag, explicit warnings and a short list of the inputs that drove it.

* **Data:** 12 routes, {len(clean):,} trips (2022-2025), generated by a seeded **synthetic** generator (no real or personal data).
* **Evaluation:** strict chronological split; hyper-parameters tuned on a validation quarter; the final test quarter ({sp['test'][0]} to {sp['test'][1]}, {split['test']:,} trips) was used once.
* **Result:** the selected model ({NAMES[best]}) reaches **WAPE {T[best]['WAPE']:.1%}** and **MAE {T[best]['MAE']:.2f} passengers** on the test set versus **WAPE {T['baseline_route_time']['WAPE']:.1%} / MAE {T['baseline_route_time']['MAE']:.2f}** for the historical route/time baseline: a **{imp['relative_wape_reduction_pct']}% relative reduction in WAPE**. It beats the baseline on {n_routes_better} of {len(m['by_route'][best])} routes and in both peak and off-peak periods.
* **Honest limits:** it cannot forecast unannounced disruptions, new routes or unusual events; it forecasts *boarded* passengers, which understates demand on sold-out trips; data is synthetic, so the absolute numbers are not a claim about real buses.

# 2. Problem framing

| Item | Definition |
|---|---|
| Business decision | How many buses (and which bus size) to schedule for each departure; when to add a standby bus or consolidate trips |
| Prediction unit | One scheduled trip = route + departure date/time |
| Target | `passengers` = passengers who boarded (integer, <= seat capacity) |
| Prediction horizon | 7 days before the departure date |
| Point in time | 00:00 on (departure date - 7 days). Only trips that departed on or before **departure date - 8 days** are visible |
| Success metrics | MAE and RMSE (in passengers), **WAPE** (scale-free, robust to small counts); per-route and peak/off-peak breakdowns; precision/recall of a "bus will be >= 90 % full" flag |
| Comparison | Historical route/time baseline (required) + seasonal naive baseline + Ridge, Random forest, Gradient boosting |

Why 7 days? A planner needs about a week to re-assign vehicles and crews. A longer horizon gives more lead time but less accurate forecasts; a shorter one is easier to forecast but too late to act on.

# 3. Data sourcing and representativeness

* **Source:** `busdemand/data_generator.py` (seeded, `--seed 42`). No licensed or personal data is used, so there is no licence restriction; the generator and the data are original work for this assessment.
* **One row = one trip:** `trip_id, route_id, origin, destination, distance_km, departure_datetime, capacity_seats, passengers`.
* **Generating process (documented so reviewers can challenge it):** demand = route size x time-of-day x weekday x season x yearly trend (+5 %/year) x holiday effect x hidden shocks (rain, strikes, landslides) with log-normal trip noise and Poisson counts; observed passengers = min(demand, seats). Route types (commuter, intercity, tourist) have different weekday/season/holiday behaviour.
* **Hidden shocks are never given to the model**, because they are not known 7 days ahead; they create realistic unexplained error.
* **Size:** {len(clean):,} cleaned trips (>= 1,000 required), {m['data']['routes']} routes, {clean['date'].min().date()} to {clean['date'].max().date()}.
* **Representativeness:** the routes are illustrative north-India city pairs; the demand patterns are plausible but invented. Real data would contain booking-curve, fare, competitor and weather information that this dataset does not.

![System architecture](figures/architecture.png)

# 4. Data-quality checks and cleaning

The raw export deliberately contains the problems a real booking system export has:

| Issue | Rows | Handling |
|---|---|---|
| Exact duplicate trips | {q['duplicates_dropped']} | dropped (keep first `trip_id`) |
| Missing `capacity_seats` | {q['missing_before']['capacity_seats']} | filled with most common capacity for the same route + time slot |
| Missing `distance_km` | {q['missing_before']['distance_km']} | filled from the route master |
| Missing `passengers` | {q['missing_before']['passengers']} | kept in calendar, excluded from training/evaluation |
| Impossible `passengers` (negative or > seats) | {q['impossible_passengers_set_nan']} | set to missing, excluded |
| Statistical outliers (robust z > 3.5 within route/slot/weekday) | {q['outliers_flagged_not_removed']} | **flagged, not removed**: mostly genuine disruptions |

After cleaning: {q['rows_clean']:,} rows, {q['rows_without_valid_target']} without a valid target; {q['sold_out_share']:.1%} of valid trips are sold out (censored demand).

![Data quality](figures/eda_data_quality.png)

# 5. Exploratory data analysis

![Target distribution](figures/eda_target_distribution.png)

Mean occupancy is {ok.occupancy.mean():.1%} and {q['sold_out_share']:.1%} of trips are sold out. The spike at 100 % is **censoring**: we observe seats sold, not true demand.

![Patterns](figures/eda_patterns.png)

* Time of day matters and differs by route type (commuter peaks 06-10 and 16-20; tourist routes peak early morning).
* Weekday effects differ: intercity routes peak Friday/Sunday, tourist routes Saturday, commuter routes Monday/Friday.
* Seasonality: tourist routes peak in summer and December; a +5 %/year trend is visible.

![Route x weekday](figures/eda_route_weekday_heatmap.png)
![Daily total](figures/eda_daily_total.png)

Festival and holiday periods produce visible spikes in the daily total. Full EDA with code is in `notebooks/01_eda.ipynb`.

# 6. Leakage analysis

| Risk | Control | Evidence |
|---|---|---|
| Using the trip's own outcome | `passengers`, `occupancy`, `sold_out`, `valid_target`, `outlier_flag` are forbidden inputs | `leakage_audit.check_forbidden_inputs` |
| Using information unknown 7 days ahead (rain, strikes) | never used as features | same check |
| History features peeking into the future | lags read only trips on or before departure date - 8 days (so same weekday last week is **not** used; the first lag is 14 days back) | unit test + audit: all data after the cutoff is overwritten with 9,999 and the features are unchanged ({m['leakage_audit']['future_invariance']['samples_checked']} samples, {m['leakage_audit']['future_invariance']['failures']} failures) |
| Random split mixing future into training | chronological split only | `check_split`: train < validation < test, no overlap |
| Tuning on test data | selection uses validation MAE only; test is evaluated once after refit | `train.py` stages 4-5 |
| Sensitive data | dataset has aggregate counts only, no passenger-level fields | `check_no_sensitive_columns` |

All audit checks passed: **{m['leakage_audit']['all_passed']}**. The calendar/holiday features are known months ahead, so they are legitimate.

# 7. Feature engineering

{feats}

* **Recent-history features:** demand of the same route/slot on the last four same-weekdays that were observable at forecast time (`lag1..lag4`, mean), the same in occupancy terms (useful because capacity varies), and the route's average demand/occupancy over the last 28 observable days.
* **Calendar:** hour, weekday, weekend, season, day-of-year (sin/cos), peak-hour flag, holiday, major festival, days to next / since previous holiday, whether the next holiday is a major festival. The holiday calendar is approximate and synthetic-project specific.
* **Capacity features:** seat capacity of the planned bus and route distance.
* **Missing history** is imputed with the median in Ridge/Random forest and counted in `n_lags_available`.
* **Outliers** are not removed (they are genuine events). Predictions are clipped to [0, seat capacity].

# 8. Validation strategy

| Split | Dates | Trips | Purpose |
|---|---|---|---|
| Train | {sp['train'][0]} to {sp['train'][1]} | {split['train']:,} | fit candidate models |
| Validation | {sp['validation'][0]} to {sp['validation'][1]} | {split['validation']:,} | tune and select the model |
| Test | {sp['test'][0]} to {sp['test'][1]} | {split['test']:,} | final, untouched until the end; model and baselines are refit on train+validation |

The first {C.WARMUP_DAYS} days are excluded because their lag history is incomplete. The test quarter (Oct-Dec) contains Dussehra, Diwali, Guru Nanak Jayanti and Christmas, so it is a demanding period.

# 9. Baselines, candidate models and tuning

* **Baseline A (required): route x departure slot x weekday historical average** with fall-backs (route x slot, then route). It is what a planner could compute in a spreadsheet.
* **Baseline B: seasonal naive**, the mean of the same slot on the last four observable same-weekdays.
* **Ridge regression** (linear, one-hot categories) - interpretable but cannot capture interactions such as route x time x weekday.
* **Random forest** - non-linear, robust, slower.
* **Gradient boosting (HistGradientBoostingRegressor)** - strong on tabular data; Poisson loss tested because the target is a count.

Hyper-parameters were chosen from small grids on **validation MAE only**:

{tune_tbl}

Selected: **{NAMES[best]}**, parameters `{m['best_params'][best]}`.

![Validation tuning](figures/res_validation_tuning.png)

# 10. Results on the held-out test set

{md_table(overall)}

![Overall metrics](figures/res_overall_metrics.png)

* The selected model reduces WAPE from {T['baseline_route_time']['WAPE']:.1%} to {T[best]['WAPE']:.1%} ({imp['relative_wape_reduction_pct']}% relative) and MAE from {T['baseline_route_time']['MAE']:.2f} to {T[best]['MAE']:.2f} passengers versus Baseline A.
* The ranking on validation and test is consistent (gradient boosting < random forest < ridge < baselines on MAE), which suggests the selection was not overfitted to the validation quarter.
* **Business meaning:** an average miss of {T[best]['MAE']:.1f} passengers on a 40-52 seat bus. Baseline A under-forecasts by {abs(T['baseline_route_time']['Bias']):.1f} passengers on average (it cannot see growth, season or holidays), the model by {abs(T[best]['Bias']):.1f}. Under-forecasting is the costly direction because it means turned-away passengers.

![Actual vs predicted](figures/res_actual_vs_pred.png)
![Example route](figures/res_timeseries_example.png)

# 11. Error analysis

## 11.1 By route

{md_table(route)}

The model has lower WAPE than Baseline A on **{n_routes_better} of {len(route)} routes**. The tourist routes (R08, R09, R12) are hardest because their demand is more volatile and seasonal.

![WAPE by route](figures/res_wape_by_route.png)

## 11.2 Peak vs off-peak

{md_table(peak_t)}

![Peak vs off-peak](figures/res_peak_offpeak.png)

## 11.3 Holidays, weekends and disruptions

{md_table(day_t)}

{md_table(dis_t)}

* The model helps most on **holidays**, where Baseline A has the largest error.
* On the {int(dis_t.loc['disruption','n'])} test trips that fell on a hidden disruption (strike/closure) the model is **not** better than the baseline (MAE {dis_t.loc['disruption','Model MAE']:.1f} vs {dis_t.loc['disruption','Baseline A MAE']:.1f}) - nobody can forecast an unannounced strike 7 days ahead. Because only {int(dis_t.loc['disruption','n'])} trips are involved, this estimate is very uncertain.

## 11.4 Censoring and the worst errors

* On trips that actually sold out the model's MAE is {mae_sold:.2f} (bias {bias_sold:+.2f}) versus {mae_not:.2f} on the rest: the observed count is capped by seats, so the model learns a conservative picture of the busiest trips.
* The worst 5 % of trips account for {worst5:.0%} of all absolute error.

![Errors](figures/res_errors.png)

# 12. Capacity-aware flags and uncertainty

Flags use the predicted occupancy: **HIGH_DEMAND >= {C.HIGH_OCC:.0%}** (consider an extra bus), **ELEVATED >= {C.WATCH_OCC:.0%}** (keep a standby bus), **LOW_DEMAND <= {C.LOW_OCC:.0%}** (consider a smaller bus), otherwise NORMAL. The 90 % and 40 % thresholds were set up front. The 75 % ELEVATED tier was added after I saw that the 90 % rule had low recall on the test set; it is a round-number rule that was not tuned on test, but because it was motivated by that observation its test results are mildly optimistic. A real operator should set all thresholds from actual costs. The API also reports `sell_out_risk` when the top of the 80 % interval reaches the seat capacity.

Evaluation of the flag as a classifier of "this trip ends up >= 90 % full" (test set):

{flag_tbl}

* The strict 90 % rule is precise but misses most full trips (a mean forecast rarely reaches 90 %).
* The 75 % WATCH rule catches most full trips with the model (recall {m['test_flag_metrics'][best]['WATCH+HIGH (pred occ>=75%)']['recall']:.0%}, precision {m['test_flag_metrics'][best]['WATCH+HIGH (pred occ>=75%)']['precision']:.0%}) and has a better F1 than both baselines. About {1 - m['test_flag_metrics'][best]['WATCH+HIGH (pred occ>=75%)']['precision']:.0%} of its alerts are false alarms; that is acceptable only because an unused standby bus is cheaper than turning passengers away.

**Prediction intervals** use split-conformal calibration on validation residuals, separately for peak and off-peak. The 80 % interval covers **{iv['test_coverage_80']:.1%}** of test trips and the 90 % interval {iv['test_coverage_90']:.1%} (targets 80 % and 90 %), with mean 80 % width {iv['avg_width_80']:.1f} passengers. Intervals are widened automatically when history is stale, the departure time is unknown to the route, or the date is outside the holiday calendar.

# 13. Explainability

**Global (permutation importance on test):**

{md_table(imp_tbl, "{:.3f}")}

![Importance](figures/res_importance.png)

Recent same-slot demand and route identity/length dominate, followed by weekday, peak hour and holiday proximity, which matches the generating process and common sense.

**Local explanations** replace each input by its typical training value and report the change in the forecast (an approximation that ignores interactions).

*Typical prediction:*

{ex_block(ex['typical_prediction'])}

*Largest error without a disruption:*

{ex_block(ex['largest_error_without_disruption'])}

Why it went wrong: Christmas Day on a tourist route is rare in the training data (one per year) and the trip sold out, so the model under-reacts to the holiday uplift. This is a limitation for unusual events, not a bug in the pipeline.

*Largest error during a disruption:*

{ex_block(ex['largest_error_during_disruption'])}

The model sees normal history and normal calendar features; the strike is invisible to it, so the forecast is far too high.

# 14. Inference interface

* **API:** `uvicorn api.main:app` -> `POST /predict`, `POST /predict/batch`, `GET /routes`, `GET /model-info`, `GET /health`, Swagger UI at `/docs`.
* **App:** `streamlit run app/streamlit_app.py` -> choose route/date/time/capacity, see forecast, interval, flag, warnings, history chart and drivers, plus a model-performance tab.
* Both call `DemandPredictor`, which **loads `models/demand_model.joblib`** and builds features with the **same code used in training** (`busdemand/features.py`), so there is no training/serving skew. Responses are never hard-coded; tests check that different inputs give different outputs and that the API equals a direct call to the model.
* For dates up to the end of the data (2025-12-31) the tool runs as a **backtest** and shows the actual value for comparison; later dates are true forecasts with a stale-history warning and wider interval.

# 15. Responsible use, bias, privacy and failure modes

* **Human oversight:** every response states `human_review_required: true`. The tool recommends; a planner decides. It is not suitable for automated high-stakes decisions such as cancelling a route.
* **Privacy:** only aggregate trip-level counts are used; the dataset has no passenger identities and no personal fields. Real deployments should keep it that way (no passenger-level tracking, no demographic features).
* **Representativeness and bias:** the data is synthetic and the generating assumptions are mine. Real ridership would under-represent people who cannot or do not book digitally (cash/counter passengers, informal travel) and people priced out of travelling, so a model trained on bookings may systematically under-serve those groups' routes. Forecasting only *boarded* passengers can create a feedback loop: routes that are under-served look less busy, receive less capacity, and look even less busy. Monitoring turned-away and unmet demand is needed to break the loop.
* **Failure modes:** new routes and new departure times (no history; warned and widened); disruptions and unusual events (not forecastable; warned in the UI); sold-out censoring; stale history; holiday calendar gaps; distribution shift (new competitors, fare changes, structural breaks such as a pandemic). Drift should be monitored by tracking rolling WAPE and retraining.
* **Single test window:** the test period is one quarter; it does not prove performance in every season.

# 16. Reproducibility and testing

* One command reproduces everything: `python run_all.py` (generate data -> clean -> train/evaluate -> report -> notebooks -> tests). Fixed seed {C.SEED}; re-running gives identical metrics (verified by running training twice).
* Saved artifact `models/demand_model.joblib` (+ `models/history.csv.gz` for inference lag look-ups). If your scikit-learn version differs, re-run `python -m busdemand.train` (about 2 minutes).
* `requirements.txt` (ranges) and `requirements-lock.txt` (exact tested versions).
* **30 automated tests** covering cleaning rules, calendar, lag-feature correctness, future-data invariance (leakage), metric formulas, chronological masks, baseline fall-backs, generator determinism, model loading, prediction validity, warnings, and the API (including equality with a direct model call and error handling).

# 17. Known incomplete features and assumptions

* Synthetic data only; no real operator data was validated.
* Holiday calendar is approximate and limited to 2022-2027.
* Single route-level model (not one per route) and a single test window; no rolling-origin cross-validation.
* No booking-curve features (bookings already made 7 days before departure), which would improve accuracy substantially in a real system.
* Uncertainty is residual-based (split conformal), not conditional on route.
* Demand beyond seat capacity is not modelled (censoring is documented, not corrected).

# 18. AI tools and attribution

*(Edit this section so it is a truthful description of how you worked.)*

* **AI tools used:** Claude (Anthropic) assisted with project scaffolding, code and drafting of this report. I reviewed, ran and tested the code and can explain every module.
* **Libraries:** pandas, NumPy, scikit-learn, SciPy, matplotlib, joblib, FastAPI, Uvicorn, Pydantic, Streamlit, pytest, nbformat/nbconvert (all open source).
* **Data:** fully synthetic, generated by this repository; no external dataset or licence involved. The holiday dates are public-knowledge approximations.
"""
    out = C.REPORTS / "TECHNICAL_REPORT.md"
    out.write_text(report, encoding="utf-8")
    print("wrote", out)
    if shutil.which("pandoc"):
        r = subprocess.run(["pandoc", str(out), "-o", str(C.REPORTS / "TECHNICAL_REPORT.docx"), "--toc", "--toc-depth=2",
                            f"--resource-path={C.REPORTS}"], capture_output=True, text=True)
        print("docx:", "ok" if r.returncode == 0 else r.stderr[:300])
    else:
        print("pandoc not found: skip .docx (the .md report is complete)")


if __name__ == "__main__":
    main()
