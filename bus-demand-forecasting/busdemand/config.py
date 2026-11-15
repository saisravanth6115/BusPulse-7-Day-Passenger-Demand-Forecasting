"""Central configuration: paths, seeds, split dates and business thresholds.

Everything that defines the *problem framing* lives here so that the report,
the training pipeline, the API and the tests all use the same numbers.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

RAW_TRIPS = DATA_RAW / "bus_trips_raw.csv"
ROUTES_CSV = DATA_RAW / "routes.csv"
EVENTS_CSV = DATA_RAW / "disruption_events.csv"
CLEAN_TRIPS = DATA_PROC / "trips_clean.csv.gz"
QUALITY_JSON = REPORTS / "data_quality_report.json"
METRICS_JSON = REPORTS / "metrics.json"
TEST_PREDS = REPORTS / "test_predictions.csv"
ARTIFACT = MODELS / "demand_model.joblib"

SEED = 42

# ---- synthetic data window ------------------------------------------------
START_DATE = "2022-01-01"
END_DATE = "2025-12-31"

# ---- chronological split (no shuffling, no overlap) -----------------------
TRAIN_END = "2025-06-30"   # train      : start .. 2025-06-30
VAL_END = "2025-09-30"     # validation : 2025-07-01 .. 2025-09-30 (model selection / tuning)
#                            test       : 2025-10-01 .. 2025-12-31 (touched once, at the very end)

# ---- problem framing ---------------------------------------------------------
# Forecast is issued at 00:00 on (departure_date - HORIZON_DAYS).
# The last *fully observed* day is therefore departure_date - (HORIZON_DAYS + 1).
HORIZON_DAYS = 7
WARMUP_DAYS = 42           # first days have incomplete lag history -> not used for modelling

# ---- capacity-aware flags (planning thresholds) ------------------------------
HIGH_OCC = 0.90            # predicted occupancy >= 90 %  -> consider an extra bus
WATCH_OCC = 0.75           # predicted occupancy >= 75 %  -> keep a standby bus / monitor bookings (round-number rule, not tuned; added after seeing the 90% rule had low recall - disclosed in report)
LOW_OCC = 0.40             # predicted occupancy <= 40 %  -> consider consolidating / smaller bus
INTERVAL_LEVEL = 0.80      # prediction interval coverage target

# ---- peak definition (by scheduled departure hour) ---------------------------
PEAK_HOURS = tuple(range(6, 10)) + tuple(range(16, 21))   # 06-09:59 and 16-20:59
