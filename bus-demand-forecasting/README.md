# Bus Passenger-Demand Forecasting

Algions AI Labs internship assessment - AI & Machine Learning - **Bus Booking domain**.
Forecasts passengers per scheduled bus trip **7 days before departure**, with a prediction interval, capacity-aware flags, warnings and explanations. Decision support only: a human planner reviews every forecast.

| | |
|---|---|
| Student | _your name / registration number_ |
| Data | **Synthetic** (generator in `busdemand/data_generator.py`); no real or personal data, no licence issues |
| Target | passengers boarded (<= seats) per trip (route + departure date/time) |
| Best model | Gradient boosting; test WAPE 16.5 % vs 20.9 % for the route/time baseline (see `reports/TECHNICAL_REPORT.md`) |

## 1. Setup (Python 3.10+)
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt        # or requirements-lock.txt for exact tested versions
```

## 2. Reproduce everything (one command, ~4 min)
```bash
python run_all.py                      # add --skip-notebooks to save a minute
```
or step by step:
```bash
python -m busdemand.data_generator     # 1. synthetic raw data -> data/raw/ (seed 42)
python -m busdemand.train              # 2. clean -> features -> chronological split -> baselines + 3 models
                                       #    -> test evaluation -> intervals -> explanations -> leakage audit
                                       #    -> models/demand_model.joblib, reports/metrics.json, figures
python -m busdemand.make_report        # 3. reports/TECHNICAL_REPORT.md (+ .docx)
python scripts/build_notebooks.py      # 4. execute notebooks/01_eda.ipynb and 02_model_evaluation.ipynb
python -m pytest -q                    # 5. 30 tests (preprocessing, leakage, metrics, model, API)
python -m busdemand.leakage_audit      # leakage / sensitive-data audit on its own
```
Re-running gives identical metrics (fixed seeds). The committed artifact was built with the versions in `requirements-lock.txt`; if loading fails with a scikit-learn version error, run step 2 again.

## 3. Use the model
**API** (Swagger UI at http://127.0.0.1:8000/docs):
```bash
uvicorn api.main:app --reload
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"route_id":"R01","departure":"2025-12-24T18:00:00","capacity_seats":45}'
```
**Interactive app:**
```bash
streamlit run app/streamlit_app.py
```
Dates up to 2025-12-31 run as a *backtest* (actual value shown); later dates are real forecasts with a stale-history warning and wider interval. Both interfaces load `models/demand_model.joblib` - nothing is hard-coded.

## 4. Problem framing
* **Decision:** how many buses / which size per departure.
* **Point in time:** 00:00, 7 days before departure date; only trips up to departure date - 8 days are visible (first lag is 14 days back).
* **Metrics:** MAE, RMSE, WAPE; per-route and peak/off-peak; precision/recall of "bus >= 90 % full" flag.
* **Split:** train <= 2025-06-30 < validation <= 2025-09-30 < test (Oct-Dec 2025, used once).

## 5. Repository map
```
busdemand/   data_generator, data_quality, holidays_cal, features, evaluation, models, explain,
             leakage_audit, train, predict, make_figures, make_report
api/main.py  FastAPI service          app/streamlit_app.py  interactive UI
notebooks/   01_eda, 02_model_evaluation (executed)
tests/       30 pytest tests          models/  saved artifact + inference history
reports/     TECHNICAL_REPORT.md/.docx, metrics.json, figures/, leakage_audit.json, data_quality_report.json
data/        raw (messy synthetic export), processed (cleaned)
```

## 6. Limitations (short)
Synthetic data; cannot forecast new routes, unannounced disruptions or unusual events; forecasts boarded (not latent) demand; single test window; approximate holiday calendar. Details, bias/privacy discussion and failure modes: report sections 15 and 17.

No secrets are needed; `.env.example` is a placeholder only.
