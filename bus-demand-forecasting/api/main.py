"""FastAPI inference service. Every response comes from the SAVED model artifact.

Start:   uvicorn api.main:app --reload
Docs:    http://127.0.0.1:8000/docs   (interactive Swagger UI - good for the demo video)
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from busdemand import config as C
from busdemand.predict import LIMITATIONS, get_predictor

app = FastAPI(
    title="Bus Passenger-Demand Forecast API",
    version="1.0.0",
    description="Decision-support forecasts of passengers per scheduled bus trip, 7 days ahead. "
                "Human review required. Trained on synthetic data.",
)


class PredictRequest(BaseModel):
    route_id: str = Field(..., examples=["R01"], description="Route id from GET /routes")
    departure: datetime = Field(..., examples=["2025-12-24T18:00:00"], description="Scheduled departure date-time")
    capacity_seats: Optional[int] = Field(None, ge=10, le=80, examples=[45], description="Seats on the planned bus (default: route standard)")
    explain: bool = Field(True, description="Include the top drivers of this forecast")


class BatchRequest(BaseModel):
    items: List[PredictRequest] = Field(..., max_length=200)


@app.get("/health")
def health():
    p = get_predictor()
    return {"status": "ok", "model": p.art["model_name"], "history_ends": p.last_observed_date()}


@app.get("/routes")
def routes():
    return get_predictor().list_routes()


@app.get("/model-info")
def model_info():
    p = get_predictor()
    info = {"model": p.art["model_name"], "params": p.art["params"], "train_window": p.art["train_window"],
            "horizon_days": p.horizon, "seed": p.art["seed"], "test_metrics": p.art["test_metrics"],
            "baseline_test_metrics": p.art["baseline_test_metrics"], "limitations": LIMITATIONS}
    if C.METRICS_JSON.exists():
        m = json.loads(C.METRICS_JSON.read_text())
        info["improvement_vs_baseline"] = m["improvement_vs_baseline"]
        info["interval"] = {k: v for k, v in m["interval"].items() if k != "widths"}
    return info


@app.post("/predict")
def predict(req: PredictRequest):
    try:
        return get_predictor().predict(req.route_id, req.departure, req.capacity_seats, req.explain)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/predict/batch")
def predict_batch(req: BatchRequest):
    out = []
    pred = get_predictor()
    for it in req.items:
        try:
            r = pred.predict(it.route_id, it.departure, it.capacity_seats, False)
            out.append({k: r[k] for k in ["route_id", "departure", "capacity_seats", "predicted_passengers", "interval_low",
                                          "interval_high", "predicted_occupancy", "capacity_flag", "sell_out_risk", "warnings"]})
        except ValueError as exc:
            out.append({"route_id": it.route_id, "departure": str(it.departure), "error": str(exc)})
    return out
