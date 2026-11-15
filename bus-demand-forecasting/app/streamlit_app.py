"""Interactive planner app (loads the saved model; no hard-coded predictions).

Run:  streamlit run app/streamlit_app.py
"""
import json
import sys
from datetime import date, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from busdemand import config as C  # noqa: E402
from busdemand.predict import LIMITATIONS, get_predictor  # noqa: E402

st.set_page_config(page_title="Bus Demand Forecast", page_icon="🚌", layout="wide")
pred = get_predictor()
routes = {r["route_id"]: r for r in pred.list_routes()}

st.title("🚌 Bus Passenger-Demand Forecast")
st.caption("Decision support for capacity planning: forecast issued 7 days before departure. "
           "A human planner must review every forecast. Trained on synthetic data.")

tab_pred, tab_perf = st.tabs(["Forecast a trip", "Model performance"])

with tab_pred:
    left, right = st.columns([1, 2])
    with left:
        rid = st.selectbox("Route", list(routes), format_func=lambda r: f"{r} - {routes[r]['origin']} to {routes[r]['destination']}")
        d = st.date_input("Departure date", value=date(2025, 12, 24))
        slot_choice = st.selectbox("Scheduled departure time", routes[rid]["slots"] + ["custom time"])
        if slot_choice == "custom time":
            t = st.time_input("Custom time", value=time(10, 15))
        else:
            hh, mm = map(int, slot_choice.split(":"))
            t = time(hh, mm)
        cap = st.number_input("Seats on planned bus", 10, 80, int(routes[rid]["default_capacity"]))
        go = st.button("Forecast demand", type="primary")
        st.info(f"History in the system ends on **{pred.last_observed_date()}**. Dates up to then run as a *backtest* "
                "(actual value shown for comparison); later dates are true forecasts with wider uncertainty.")

    with right:
        if go:
            dep = pd.Timestamp.combine(d, t) if hasattr(pd.Timestamp, "combine") else pd.Timestamp(f"{d} {t}")
            try:
                r = pred.predict(rid, dep, int(cap), explain=True)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()
            colour = {"HIGH_DEMAND": "red", "ELEVATED": "orange", "NORMAL": "green", "LOW_DEMAND": "blue"}[r["capacity_flag"]]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Predicted passengers", f"{r['predicted_passengers']:.0f}")
            c2.metric(f"{int(r['interval_level'] * 100)}% interval", f"{r['interval_low']:.0f} - {r['interval_high']:.0f}")
            c3.metric("Predicted occupancy", f"{r['predicted_occupancy']:.0%}")
            c4.metric("Baseline (route/time avg)", f"{r['baseline_route_time_prediction']:.0f}")
            st.markdown(f"### Capacity flag: :{colour}[{r['capacity_flag']}]  {'⚠️ sell-out risk' if r['sell_out_risk'] else ''}")
            st.write(r["advice"])
            if "actual_passengers_for_comparison" in r:
                st.success(f"Backtest: actual boarded passengers were **{r['actual_passengers_for_comparison']:.0f}** "
                           f"(forecast error {r['predicted_passengers'] - r['actual_passengers_for_comparison']:+.1f}).")
            for w in r["warnings"]:
                st.warning(w)

            # chart: same-slot history + forecast
            slot = dep.strftime("%H:%M")
            h = pred.recent_history(rid, slot, dep - pd.Timedelta(days=C.HORIZON_DAYS + 1), n=16)
            fig, ax = plt.subplots(figsize=(8, 3.2))
            if len(h):
                ax.plot(h["date"], h["passengers"], marker="o", color="#4c78a8", label="observed, same route & time")
            ax.errorbar([dep.normalize()], [r["predicted_passengers"]],
                        yerr=[[r["predicted_passengers"] - r["interval_low"]], [r["interval_high"] - r["predicted_passengers"]]],
                        fmt="s", color="#e15759", capsize=6, label="forecast + interval")
            if "actual_passengers_for_comparison" in r:
                ax.scatter([dep.normalize()], [r["actual_passengers_for_comparison"]], color="k", marker="x", s=70, label="actual")
            ax.axhline(r["capacity_seats"], color="grey", ls="--", label="capacity")
            ax.set_ylabel("passengers")
            ax.legend(fontsize=8)
            st.pyplot(fig)

            st.subheader("Why this forecast? (top drivers)")
            drv = pd.DataFrame(r["top_drivers"])[["meaning", "value", "typical_value", "impact_passengers"]]
            drv.columns = ["input", "this trip", "typical value", "effect on forecast (passengers)"]
            drv[["this trip", "typical value"]] = drv[["this trip", "typical value"]].astype(str)
            st.dataframe(drv, hide_index=True, width="stretch")
            st.caption("Effect = change in forecast if this input were replaced by its typical training value.")
        else:
            st.write("Choose a route, date and time, then press **Forecast demand**.")

    with st.expander("Limitations & responsible use", expanded=False):
        for x in LIMITATIONS:
            st.write("- " + x)

with tab_perf:
    if C.METRICS_JSON.exists():
        m = json.loads(C.METRICS_JSON.read_text())
        st.subheader("Held-out TEST metrics (Oct-Dec 2025)")
        tbl = pd.DataFrame(m["test"]).T[["MAE", "RMSE", "WAPE", "Bias"]].round(3)
        st.dataframe(tbl)
        st.write(f"Best model: **{m['best_model']}** - WAPE reduced by **{m['improvement_vs_baseline']['relative_wape_reduction_pct']}%** vs the route/time baseline.")
        for f in ["res_overall_metrics.png", "res_wape_by_route.png", "res_peak_offpeak.png", "res_timeseries_example.png", "res_importance.png"]:
            if (C.FIGURES / f).exists():
                st.image(str(C.FIGURES / f))
    else:
        st.warning("Run the training pipeline first: python -m busdemand.train")
