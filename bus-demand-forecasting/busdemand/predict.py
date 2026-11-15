"""Inference: loads the SAVED model artifact and returns real predictions (nothing is hard-coded).

    from busdemand.predict import DemandPredictor
    p = DemandPredictor()
    p.predict("R01", "2025-12-24 18:00")
"""
from __future__ import annotations

import datetime as dt
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from . import config as C
from .explain import explain_prediction
from .features import FEATURES, HistoryIndex, build_features
from .holidays_cal import calendar_covered
from .models import clip_to_capacity, interval

LIMITATIONS = [
    "Decision-support only: a planner must review every forecast before changing a schedule.",
    "Forecasts passengers who BOARDED (capped at seats). On sold-out trips true demand is higher than what was observed.",
    "New routes or new departure times have no history: the model cannot forecast them reliably.",
    "Unannounced disruptions (strikes, landslides, breakdowns, bad weather) cannot be predicted 7 days ahead.",
    "Unusual events (concerts, exams, elections, new competitors) are not in the data.",
    "Trained on SYNTHETIC data: use real operator data and re-validate before any operational use.",
]


class DemandPredictor:
    def __init__(self, artifact_path=C.ARTIFACT, history_path=C.MODELS / "history.csv.gz"):
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {artifact_path}. Run:  python -m busdemand.train")
        self.art = joblib.load(artifact_path)
        hist = pd.read_csv(history_path, parse_dates=["date"])
        self.hist = hist
        self.index = HistoryIndex(hist)
        self.routes = {r["route_id"]: r for r in self.art["routes"]}
        self.horizon = self.art["horizon_days"]

    # ------------------------------------------------------------------ helpers
    def list_routes(self) -> list[dict]:
        out = []
        for r in self.routes.values():
            out.append({"route_id": r["route_id"], "origin": r["origin"], "destination": r["destination"],
                        "distance_km": r["distance_km"], "default_capacity": int(r["default_capacity"]),
                        "slots": r["slots"].split("|")})
        return out

    def last_observed_date(self) -> str:
        return str(self.index.last_observed_date().date())

    def recent_history(self, route_id: str, slot: str, before: str | pd.Timestamp, n: int = 12) -> pd.DataFrame:
        """Last n observed trips of the same route+slot strictly before `before` (for charts)."""
        h = self.hist[(self.hist.route_id == route_id) & (self.hist.slot == slot) & (self.hist.date < pd.Timestamp(before))]
        return h.sort_values("date").tail(n)[["date", "passengers", "capacity_seats"]]

    # ------------------------------------------------------------------ main API
    def predict(self, route_id: str, departure, capacity: int | None = None, explain: bool = True) -> dict:
        if route_id not in self.routes:
            raise ValueError(f"Unknown route '{route_id}'. New routes are not supported (no history). Known: {sorted(self.routes)}")
        dep = pd.Timestamp(departure)
        if pd.isna(dep):
            raise ValueError("departure must be a valid date-time, e.g. 2025-12-24 18:00")
        route = self.routes[route_id]
        cap = int(capacity) if capacity is not None else int(route["default_capacity"])
        if not 10 <= cap <= 80:
            raise ValueError("capacity must be between 10 and 80 seats")

        row = pd.DataFrame({"route_id": [route_id], "departure_datetime": [dep],
                            "distance_km": [float(route["distance_km"])], "capacity_seats": [float(cap)]})
        X = build_features(row, self.index, self.horizon)
        raw = float(self.art["pipeline"].predict(X[FEATURES])[0])
        pred = float(clip_to_capacity(raw, cap))
        is_peak = int(X["is_peak_hour"].iloc[0])

        # ---- reliability warnings + interval widening
        warnings, scale = [], 1.0
        slot = dep.strftime("%H:%M")
        target_day = HistoryIndex.day_number(dep)
        last_day = self.index.max_day
        known_slots = route["slots"].split("|")
        if slot not in known_slots:
            warnings.append(f"Departure time {slot} is not a scheduled slot for {route_id} (known: {', '.join(known_slots)}). "
                            "No same-slot history: forecast is much less reliable.")
            scale = max(scale, 1.5)
        n_lags = int(X["n_lags_available"].iloc[0])
        if slot in known_slots and n_lags < 2:
            warnings.append(f"Only {n_lags} of 4 same-weekday history points available: reliability reduced.")
            scale = max(scale, 1.25)
        age = target_day - (self.horizon + 1) - last_day
        mode = "backtest" if target_day <= last_day else "forecast"
        if age > 0:
            warnings.append(f"Observed history ends on {self.last_observed_date()}; this departure is {age} days past the data the "
                            "model normally sees (it uses the latest available weeks). Retrain with fresh data for reliable forecasts.")
            scale = max(scale, 1.0 + min(age / 28.0, 1.0))
        if not calendar_covered(dep):
            warnings.append("Holiday calendar does not cover this date: holiday effects are ignored.")
            scale = max(scale, 1.25)
        if int(X["is_major_festival"].iloc[0]) or (int(X["next_holiday_major"].iloc[0]) and int(X["days_to_next_holiday"].iloc[0]) <= 2):
            warnings.append("Major festival period: demand is more volatile and the model has few comparable examples.")
            scale = max(scale, 1.25)

        lo, hi = interval(pred, cap, is_peak, self.art["interval_widths"], C.INTERVAL_LEVEL, scale)
        lo, hi = float(np.atleast_1d(lo)[0]), float(np.atleast_1d(hi)[0])

        # ---- capacity-aware flag
        occ = pred / cap
        if occ >= C.HIGH_OCC:
            level, advice = "HIGH_DEMAND", "Likely (almost) full. Consider an extra bus or a larger vehicle."
        elif occ >= C.WATCH_OCC:
            level, advice = "ELEVATED", "Demand above normal. Keep a standby bus and monitor bookings."
        elif occ <= C.LOW_OCC:
            level, advice = "LOW_DEMAND", "Likely under-filled. Consider a smaller bus or consolidating trips."
        else:
            level, advice = "NORMAL", "Normal load. Keep the planned capacity."
        sell_out_risk = hi >= cap

        base = float(clip_to_capacity(self.art["baseline"].predict(row.assign(slot=slot))[0], cap))
        result = {
            "route_id": route_id, "route": f"{route['origin']} - {route['destination']}",
            "departure": str(dep), "capacity_seats": cap, "mode": mode,
            "forecast_issued_for": f"{self.horizon} days before departure",
            "predicted_passengers": round(pred, 1),
            "interval_low": round(lo, 1), "interval_high": round(hi, 1), "interval_level": C.INTERVAL_LEVEL,
            "predicted_occupancy": round(occ, 3),
            "capacity_flag": level, "sell_out_risk": bool(sell_out_risk), "advice": advice,
            "baseline_route_time_prediction": round(base, 1),
            "warnings": warnings, "limitations": LIMITATIONS, "human_review_required": True,
            "model": {"name": self.art["model_name"], "train_window": self.art["train_window"],
                      "test_MAE": round(self.art["test_metrics"]["MAE"], 3), "test_WAPE": round(self.art["test_metrics"]["WAPE"], 4)},
        }
        if mode == "backtest":
            v = self.index.trip.get((route_id, slot, target_day))
            if v:
                result["actual_passengers_for_comparison"] = v[0]
        if explain:
            result["top_drivers"] = explain_prediction(self.art["pipeline"], X, self.art["reference_row"])
        return result


@lru_cache(maxsize=1)
def get_predictor() -> DemandPredictor:
    return DemandPredictor()
