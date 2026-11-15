# Your step-by-step guide to 100/100

Do these in order. Tick each box.

## Part A - Run it on your own computer (30 min)
- [ ] Install Python 3.10+ and VS Code. Unzip the project, open the folder in VS Code.
- [ ] Terminal: `python -m venv .venv`, activate it, `pip install -r requirements.txt`
- [ ] `python run_all.py` (about 4 min). Everything must say passed / DONE.
- [ ] `uvicorn api.main:app --reload` -> open http://127.0.0.1:8000/docs, try POST /predict.
- [ ] `streamlit run app/streamlit_app.py` -> try a route and date. Stop both when done.

## Part B - Make it yours (1-2 hours, DO NOT SKIP)
The assignment says you must explain and modify your own code in a live interview, and that AI output without understanding earns nothing.
- [ ] Put your name and registration number in: README.md, reports/TECHNICAL_REPORT.md (top), then run `python -m busdemand.make_report` to rebuild the .docx.
- [ ] Read these files in this order and be able to explain each in 2 minutes: `config.py` -> `data_generator.py` -> `data_quality.py` -> `features.py` -> `evaluation.py` -> `models.py` -> `train.py` -> `predict.py` -> `api/main.py`.
- [ ] Try one change yourself and re-run: e.g. change `HORIZON_DAYS` to 3 in config.py and see what happens to the metrics, or add a route to `ROUTES` in `data_generator.py`.
- [ ] Section 18 of the report ("AI tools and attribution") and the form's AI declaration must be truthful. It already says Claude helped with scaffolding, code and drafting; edit it to match how you really worked.

## Part C - GitHub (20 min)
- [ ] Create a GitHub repo (public is easiest, so reviewers can open it). Then in the project folder:
```
git init
git add .
git commit -m "Bus passenger-demand forecasting: final"
git branch -M main
git remote add origin https://github.com/<you>/<repo>.git
git push -u origin main
git rev-parse HEAD        # <- this is the "final commit ID" for the form
```
- [ ] Open the repo in a private/incognito window to confirm it loads. Confirm README, report, notebooks, models/ and reports/figures are visible.
- [ ] If private, add the reviewer as collaborator per the invitation email.

## Part D - The 3-5 minute video (record with OBS, Zoom or Xbox Game Bar)
Suggested script (about 4:30):
1. **0:00-0:30 Problem.** "Forecast passengers per bus trip 7 days before departure so planners can add or cut capacity. Synthetic data, 12 routes, 4 years."
2. **0:30-1:15 EDA** (notebook 01): show data-quality chart, occupancy histogram with the 100% spike (censoring), hour/weekday/month patterns. Mention duplicates, missing values and impossible values were cleaned.
3. **1:15-2:00 Method:** chronological split diagram (train to Jun 2025, validation Q3, test Q4), leakage rule (first lag is 14 days back), baselines vs Ridge / Random forest / Gradient boosting.
4. **2:00-3:00 Results** (`res_overall_metrics.png`, `res_wape_by_route.png`, `res_peak_offpeak.png`): WAPE 20.9% -> 16.5%; better on all 12 routes and peak/off-peak. Be honest that disruption days are not predictable.
5. **3:00-4:15 Live prediction:** in the Streamlit app (or Swagger `/docs`) forecast R01, 2025-12-24 18:00. Point to the forecast, interval, flag, actual-vs-forecast, drivers. Then a second input (e.g. R08 on 2025-12-25) to show a different answer. Then enter an unknown time like 03:17 to show the warning.
6. **4:15-4:45 Limits:** synthetic data, new routes, disruptions, human review required.
- [ ] Upload to YouTube (unlisted) or Google Drive (anyone with link). Test the link in incognito.

## Part E - Submission form answers
| Field | What to enter |
|---|---|
| Project title | Bus Passenger-Demand Forecasting |
| Role | AI & Machine Learning |
| Repo / branch / commit | your GitHub URL / `main` / output of `git rev-parse HEAD` |
| Demo | video link; "live demo: local-only (FastAPI + Streamlit, commands in README)" unless you deploy Streamlit Community Cloud (free) |
| Documentation | link to `reports/TECHNICAL_REPORT.md` in the repo (and .docx); architecture: `reports/figures/architecture.png` |
| Tech stack | Python 3.12, pandas, scikit-learn (HistGradientBoosting), FastAPI, Streamlit, pytest, matplotlib |
| Dataset | Synthetic generator `busdemand/data_generator.py` (seed 42), no external licence |
| Model artifact | `models/demand_model.joblib` in repo (+ regenerate with `python -m busdemand.train`) |
| Metrics | Test MAE 4.94, RMSE 6.18, WAPE 16.5% vs baseline MAE 6.25, RMSE 7.84, WAPE 20.9% |
| Inference | FastAPI `uvicorn api.main:app` (`/docs`), Streamlit `app/streamlit_app.py` (local) |
| Known incomplete / assumptions | synthetic data; approximate holiday calendar; single test window; no booking-curve features; disruptions not forecastable |
| AI tools | Claude (Anthropic) for scaffolding/code/report drafting; I reviewed, tested and can explain everything (only tick this if true) |

## Part F - Final checks (Day 15)
- [ ] Fresh clone in a new folder, `pip install -r requirements.txt`, `python run_all.py --skip-notebooks` works.
- [ ] All links open in incognito. No `.env` or secrets committed (only `.env.example`).

## Part G - Interview prep (likely questions)
1. **Why a 7-day horizon and why is the first lag 14 days?** Forecast issued at 00:00 on D-7, so the same weekday last week (D-7) hasn't departed yet; D-14 is the first observable one. `features.py: history_features`.
2. **How do you prove there's no leakage?** `leakage_audit.py` and `tests/test_features.py`: overwrite everything after the cutoff with 9,999; features must not change.
3. **Why WAPE?** Scale-free, comparable across routes, no division by small counts, interpretable as "% of passengers mis-forecast".
4. **Why not a random split?** Time series: random split lets the model see the future and inflates scores.
5. **Why gradient boosting?** Best validation MAE; captures interactions (route x time x weekday x holiday). Ridge can't.
6. **What is censoring?** Sold-out buses hide true demand; we forecast boarded passengers and flag sell-out risk.
7. **How is the interval built?** Split conformal: quantile of absolute validation residuals, separately for peak/off-peak; test coverage 79.9% for the 80% interval.
8. **Where does the model do badly?** Tourist routes, festivals, disruption days. Report section 11.
9. **Live change ideas:** change the thresholds in `config.py`, add a feature (e.g. `is_friday`) in `features.py` and retrain, add an API field.
