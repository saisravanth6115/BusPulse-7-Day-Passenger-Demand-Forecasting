"""Builds and executes the two notebooks. Run from repo root:  python scripts/build_notebooks.py"""
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display, Image
from busdemand import config as C
pd.set_option("display.width", 140); pd.set_option("display.max_columns", 30)
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": .25})"""

# ------------------------------------------------------------------ notebook 1
n1 = nbf.v4.new_notebook()
n1.cells = [
md("""# 01 - Problem framing, data and EDA
**Project:** Bus Passenger-Demand Forecasting (Algions AI Labs internship assessment)

## Framing
| Item | Definition |
|---|---|
| Business decision | How many buses / which bus size to schedule for each departure |
| Prediction unit | one scheduled trip = route + departure date/time |
| Target | `passengers` boarded (<= seat capacity) |
| Horizon / point in time | forecast issued **00:00, 7 days before the departure date**; only trips up to *departure date - 8 days* are visible |
| Data | **synthetic** generator (`busdemand/data_generator.py`), 12 routes, 2022-2025, no personal data |

The notebook walks through data quality, cleaning, exploratory analysis and the leakage audit."""),
code(SETUP),
md("## 1. Raw data"),
code("""raw = pd.read_csv(C.RAW_TRIPS)
print(raw.shape)
display(raw.head())
display(raw.dtypes.to_frame("dtype").T)
display(raw.isna().sum().to_frame("missing").T)"""),
md("## 2. Data-quality checks and cleaning\nThe raw export deliberately contains duplicates, missing values and impossible values (more passengers than seats, negative counts)."),
code("""from busdemand.data_quality import clean_trips
clean, rep = clean_trips(raw)
display(pd.Series({k: v for k, v in rep.items() if not isinstance(v, dict)}).to_frame("value"))
display(pd.DataFrame({"before": rep["missing_before"], "after": rep["missing_after"]}))
Image(str(C.FIGURES / "eda_data_quality.png"))"""),
md("""Cleaning rules: drop duplicate `trip_id`; fill `distance_km` from the route master; fill `capacity_seats` with the most common capacity of the same route+slot; set impossible passenger counts to missing; rows without a valid target are excluded from modelling (they stay in the calendar). **Outliers are flagged, not removed** - most are genuine disruptions that the forecast tool must be honest about."""),
md("## 3. Target distribution and censoring"),
code("""ok = clean[clean.valid_target]
print("mean passengers:", ok.passengers.mean().round(1), "| mean occupancy:", ok.occupancy.mean().round(3), "| sold-out share:", ok.sold_out.mean().round(3))
Image(str(C.FIGURES / "eda_target_distribution.png"))"""),
md("""**Observation:** the spike at 100 % occupancy is *censoring*: when a bus sells out we only see the seat count, not the true demand. The model therefore forecasts **boarded passengers**, and the app reports a sell-out-risk flag instead of pretending to know unconstrained demand."""),
md("## 4. Patterns: time of day, weekday, season, route"),
code("""Image(str(C.FIGURES / "eda_patterns.png"))"""),
code("""Image(str(C.FIGURES / "eda_route_weekday_heatmap.png"))"""),
code("""Image(str(C.FIGURES / "eda_daily_total.png"))"""),
md("## 5. Holiday effect"),
code("""from busdemand.holidays_cal import holiday_table
t = holiday_table(ok["date"])
ok2 = ok.reset_index(drop=True).assign(**t.to_dict("list"))
grp = np.where(ok2.is_holiday == 1, "holiday", np.where(ok2.days_to_next_holiday == 1, "day before holiday", np.where(ok2.days_since_prev_holiday == 1, "day after holiday", "normal day")))
display(ok2.assign(day_type=grp).groupby("day_type").occupancy.agg(["mean", "count"]).round(3))"""),
md("## 6. Outliers vs. known disruptions"),
code("""events = pd.read_csv(C.EVENTS_CSV, parse_dates=["start_date", "end_date"])
flag = pd.Series(False, index=clean.index)
for _, e in events.iterrows():
    flag |= (clean.route_id == e.route_id) & (clean.date >= e.start_date) & (clean.date <= e.end_date)
print("trips on disruption days (generator ground truth):", int(flag.sum()))
print("flagged outliers:", int(clean.outlier_flag.sum()), "| of which on disruption days:", int((clean.outlier_flag & flag).sum()))
display(events.head())"""),
md("""## 7. Leakage analysis
* **Not used as inputs:** `passengers` of the trip itself, occupancy, sold-out flag, rain/weather, disruption events (none of these are known 7 days ahead).
* **History features** (`lag1..lag4`, `lag_mean4`, `route_mean_28d`, ...) read only trips on or before `departure_date - 8 days`.
* Calendar and holiday features are known months in advance, so they are legitimate.
* Chronological split: train <= 2025-06-30 < validation <= 2025-09-30 < test.

Automated audit (also run in `pytest`):"""),
code("""from busdemand import leakage_audit
audit = leakage_audit.run(clean, write=False)
display(pd.Series({k: v["passed"] for k, v in audit.items() if isinstance(v, dict)}).to_frame("passed"))
print(json.dumps(audit["chronological_split"], indent=1))"""),
md("## 8. Feature list"),
code("""from busdemand.features import NUM_FEATURES, CAT_FEATURES
print("numeric:", NUM_FEATURES); print("categorical:", CAT_FEATURES)"""),
md("""## Take-aways
1. Demand is driven by **route, time of day, weekday, season and holidays**, and is strongly autocorrelated at the same route/slot/weekday.
2. About **7 %** of trips are sold out, so observed counts under-state demand on the busiest trips.
3. Data issues were found and handled with documented, reversible rules; outliers are kept because they represent real disruptions."""),
]
# ------------------------------------------------------------------ notebook 2
n2 = nbf.v4.new_notebook()
n2.cells = [
md("""# 02 - Models, held-out evaluation, error analysis and explanation
Everything here is read from files produced by `python -m busdemand.train` (nothing is typed in by hand)."""),
code(SETUP + """
m = json.loads(C.METRICS_JSON.read_text())
preds = pd.read_csv(C.TEST_PREDS, parse_dates=["date", "departure_datetime"])
print("best model:", m["best_model"], m["best_params"][m["best_model"]])"""),
md("## 1. Split strategy"),
code("""print(json.dumps(m["leakage_audit"]["chronological_split"], indent=1))"""),
md("## 2. Validation (used for tuning/selection) vs. TEST (used once)"),
code("""val = pd.DataFrame(m["validation"]).T[["MAE", "RMSE", "WAPE"]].add_prefix("val_")
tst = pd.DataFrame(m["test"]).T[["MAE", "RMSE", "WAPE", "Bias"]].add_prefix("test_")
display(val.join(tst).round(3))
print("WAPE reduction vs route/time baseline on test:", m["improvement_vs_baseline"]["relative_wape_reduction_pct"], "%")
Image(str(C.FIGURES / "res_overall_metrics.png"))"""),
md("### Tuning log (validation only)"),
code("""display(pd.DataFrame(m["tuning_log"]).sort_values(["model", "val_MAE"]).reset_index(drop=True))"""),
md("## 3. Error analysis - per route"),
code("""best = m["best_model"]
r = pd.DataFrame(m["by_route"][best]).T[["n", "MAE", "WAPE", "Bias"]].join(pd.DataFrame(m["by_route"]["baseline_route_time"]).T[["MAE", "WAPE"]].add_prefix("baseline_"))
display(r.round(3))
Image(str(C.FIGURES / "res_wape_by_route.png"))"""),
md("## 4. Peak vs off-peak, day type, disruptions"),
code("""for key in ["by_peak", "by_day_type", "by_disruption"]:
    print(key)
    display(pd.concat({k: pd.DataFrame(m[key][k]).T[["n", "MAE", "WAPE", "Bias"]] for k in ["baseline_route_time", best]}, axis=1).round(3))
Image(str(C.FIGURES / "res_peak_offpeak.png"))"""),
md("""**Reading the bias column:** negative bias means under-forecasting. The baseline under-forecasts by ~4 passengers (it cannot see growth, seasonality or holidays); the model cuts the bias to ~1.3. Under-forecasting is the costly direction for capacity planning, so we also look at flags below."""),
md("## 5. Capacity-aware flags (business metric)"),
code("""rows = []
for k, d in m["test_flag_metrics"].items():
    for rule, v in d.items():
        rows.append({"method": k, "flag rule": rule, "precision": v["precision"], "recall": v["recall"], "f1": v["f1"], "flagged": v["flagged"], "actually full": v["actual_positive"]})
display(pd.DataFrame(rows).round(3))"""),
md("""*Event to catch = a trip that actually ends up >= 90 % full.* The 75 % WATCH rule is fixed a priori (not tuned on test). It trades some false alarms for much higher recall, which is the right trade-off when an empty extra bus costs less than turning passengers away."""),
md("## 6. Uncertainty: 80 % prediction interval (split conformal)"),
code("""iv = m["interval"]
print("target coverage 80% -> empirical TEST coverage:", round(iv["test_coverage_80"], 3), "| 90% ->", round(iv["test_coverage_90"], 3), "| mean width (80%):", round(iv["avg_width_80"], 1), "passengers")
Image(str(C.FIGURES / "res_timeseries_example.png"))"""),
md("## 7. What drives the forecast? (global)"),
code("""display(pd.DataFrame(m["importance"])[["meaning", "mae_increase"]].round(3))
Image(str(C.FIGURES / "res_importance.png"))"""),
md("## 8. Explaining individual predictions and errors"),
code("""for e in m["examples"]:
    print(f"\\n=== {e['tag']} | {e['trip_id']} | actual={e['actual']} predicted={e['predicted']} baseline={e['baseline_prediction']} | disruption: {e['disruption_known_to_generator']}")
    display(pd.DataFrame(e["drivers"])[["meaning", "value", "typical_value", "impact_passengers"]])"""),
md("""**How to read this:** *impact_passengers* = change in the forecast if that input were replaced by its typical training value (an approximation without interaction effects). For the largest non-disruption error (Christmas Day on the Chandigarh-Shimla tourist route) the model forecasts ~21 but 45 board: holidays on tourist routes are rare in training (one Christmas per year) and the trip is sold out, so the model under-reacts. For the disruption example (R02, a strike day) the model sees normal history and cannot know about the strike."""),
md("## 9. Worst errors table"),
code("""cols = ["trip_id", "y", "pred", "pred_baseline_route_time", "abs_err", "day_type", "disruption_label"]
display(preds.sort_values("abs_err", ascending=False)[cols].head(10).round(1))
print("Share of total absolute error from the worst 5% of trips:", round(preds.abs_err.sort_values(ascending=False).head(int(len(preds) * .05)).sum() / preds.abs_err.sum(), 3))"""),
md("## 10. Live inference through the saved artifact"),
code("""from busdemand.predict import DemandPredictor
p = DemandPredictor()
r = p.predict("R01", "2025-12-24 18:00")
print({k: r[k] for k in ["route", "predicted_passengers", "interval_low", "interval_high", "predicted_occupancy", "capacity_flag", "sell_out_risk", "actual_passengers_for_comparison", "warnings"]})"""),
md("""## Conclusions
* The tuned gradient-boosting model beats the historical route/time baseline on every metric overall and on every route and in peak and off-peak periods.
* Error is largest for rare events (festivals on tourist routes) and unforecastable disruptions: the tool says so via warnings, wider intervals and the human-review requirement.
* Limits: synthetic data, single test window (Oct-Dec 2025), 6 disruption trips in test (too few for a firm estimate), no real booking-curve features."""),
]

for name, nb in [("01_eda.ipynb", n1), ("02_model_evaluation.ipynb", n2)]:
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    ExecutePreprocessor(timeout=300, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT / "notebooks")}})
    nbf.write(nb, ROOT / "notebooks" / name)
    print("built", name)
