"""Builds a narrated demo video (slides + synthetic voice) from the real project outputs.

    python scripts/build_demo_video.py        (needs ffmpeg and espeak-ng)
Output: reports/demo_video.mp4
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from busdemand import config as C  # noqa: E402
from busdemand.predict import DemandPredictor  # noqa: E402

W, H = 1280, 720
TMP = ROOT / "reports" / "_video_tmp"
TMP.mkdir(exist_ok=True, parents=True)
F = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FM = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
font = lambda p, s: ImageFont.truetype(p, s)  # noqa: E731
BG, INK, ACC = (248, 249, 251), (30, 36, 48), (31, 119, 180)

m = json.loads(C.METRICS_JSON.read_text())
b = m["best_model"]
T = m["test"]
pred = DemandPredictor()


def base(title, sub=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 86], fill=ACC)
    d.text((40, 22), title, font=font(FB, 36), fill="white")
    if sub:
        d.text((40, 100), sub, font=font(F, 22), fill=(90, 98, 112))
    d.text((40, H - 34), "Bus Passenger-Demand Forecasting  |  Algions AI Labs assessment", font=font(F, 16), fill=(120, 126, 138))
    return im, d


def bullets(d, items, x=50, y=140, size=28, width=62, gap=16):
    for it in items:
        lines = textwrap.wrap(it, width)
        d.text((x, y), "•", font=font(FB, size), fill=ACC)
        for ln in lines:
            d.text((x + 28, y), ln, font=font(F, size), fill=INK)
            y += size + 8
        y += gap


def paste(im, path, box):
    img = Image.open(path).convert("RGB")
    bw, bh = box[2] - box[0], box[3] - box[1]
    img.thumbnail((bw, bh))
    im.paste(img, (box[0] + (bw - img.width) // 2, box[1] + (bh - img.height) // 2))


slides = []  # (image, narration)

# 1 title
im, d = base("Bus Passenger-Demand Forecasting")
d.text((60, 200), "Forecast passengers per bus trip", font=font(FB, 44), fill=INK)
d.text((60, 262), "7 days before departure", font=font(FB, 44), fill=ACC)
bullets(d, ["Decision support for capacity planning", "Synthetic data: 12 routes, 4 years, about 100 thousand trips",
            "Models: baselines, Ridge, Random forest, Gradient boosting", "FastAPI service + Streamlit app + 30 tests"], y=370, size=26)
slides.append((im, "Hello. This is my project demonstration for the Bus Passenger Demand Forecasting assignment. The tool forecasts how many passengers will board a scheduled bus trip, seven days before departure, so a planner can add a standby bus or consolidate empty trips. The data is synthetic, with twelve routes over four years."))

# 2 framing
im, d = base("Problem framing", "What exactly is predicted, and when")
bullets(d, ["Decision: how many buses, and which size, for each departure",
            "Unit: one scheduled trip = route + departure date and time",
            "Target: passengers who boarded, never more than the seats",
            "Point in time: 00:00, seven days before departure. Only trips up to eight days before are visible, so the first lag is fourteen days back",
            "Metrics: MAE, RMSE and WAPE, plus a flag for trips that will be nearly full"], size=26)
slides.append((im, "The decision is how many buses to schedule for each departure. The prediction unit is one trip, and the target is passengers boarded. The forecast is made at midnight, seven days before departure. That means the same weekday last week has not happened yet, so the first history feature is fourteen days back. This rule is what prevents leakage. I judge models with mean absolute error, RMSE, and WAPE, plus a flag for nearly full trips."))

# 3 data quality + target
q = m["data"]["quality"]
im, d = base("Data quality and the target", "Raw export was deliberately messy; every fix is a documented rule")
paste(im, C.FIGURES / "eda_data_quality.png", (30, 130, 620, 520))
paste(im, C.FIGURES / "eda_target_distribution.png", (640, 130, 1250, 520))
d.text((40, 545), f"{q['duplicates_dropped']} duplicates dropped | {q['missing_before']['capacity_seats']} missing capacities filled | {q['impossible_passengers_set_nan']} impossible counts removed", font=font(F, 22), fill=INK)
d.text((40, 585), "Spike at 100% occupancy = sold-out trips: demand is censored, so we forecast boarded passengers", font=font(F, 22), fill=INK)
slides.append((im, f"The raw data had duplicates, missing capacities, and impossible passenger counts, such as more passengers than seats. Each problem is fixed by a documented rule, and the counts are in the quality report. Outliers are flagged but kept, because they are real disruptions. Note the spike at one hundred percent occupancy. When a bus sells out we only see the seat count, not the true demand, so the model forecasts boarded passengers and the app shows a sell out risk flag."))

# 4 EDA
im, d = base("Exploratory analysis", "Demand follows time of day, weekday, season and holidays")
paste(im, C.FIGURES / "eda_patterns.png", (30, 130, 1250, 420))
paste(im, C.FIGURES / "eda_daily_total.png", (130, 420, 1150, 690))
slides.append((im, "Exploration shows clear patterns. Demand changes by hour, with commuter peaks in the morning and evening. It changes by weekday, with intercity routes peaking on Friday and Sunday and tourist routes on Saturday. It has a yearly season and a growth trend, and festivals create visible spikes in the daily total."))

# 5 method
sp = m["leakage_audit"]["chronological_split"]
im, d = base("Method: split, leakage control, models")
bullets(d, [f"Chronological split. Train: up to {sp['train'][1]}. Validation: {sp['validation'][0]} to {sp['validation'][1]}. Test: {sp['test'][0]} to {sp['test'][1]}",
            "Tuning and model choice use validation only; test is used once, after refitting",
            "Leakage audit passes: future data is overwritten and features do not change",
            "Baselines: route/time average and seasonal naive",
            "Candidates: Ridge, Random forest, Gradient boosting (tuned on validation MAE)"], size=26)
slides.append((im, "I split the data by time, never randomly. Training runs to June twenty twenty five, validation is the third quarter, and the test quarter, October to December, is used exactly once. An automated audit overwrites all future data and checks that the features do not change. I compare two baselines, including the required route and time average, against Ridge regression, random forest, and gradient boosting, tuned on validation only."))

# 6 results
im, d = base("Results on the held-out test set", f"Best model: gradient boosting | WAPE {T['baseline_route_time']['WAPE']:.1%} -> {T[b]['WAPE']:.1%}")
paste(im, C.FIGURES / "res_overall_metrics.png", (20, 130, 1260, 520))
d.text((40, 560), f"MAE {T['baseline_route_time']['MAE']:.2f} -> {T[b]['MAE']:.2f} passengers  |  relative WAPE reduction {m['improvement_vs_baseline']['relative_wape_reduction_pct']}%", font=font(FB, 26), fill=INK)
d.text((40, 610), f"80% interval covers {m['interval']['test_coverage_80']:.1%} of test trips (target 80%)", font=font(F, 24), fill=INK)
slides.append((im, f"On the untouched test quarter, the route and time baseline has a WAPE of {T['baseline_route_time']['WAPE']*100:.1f} percent. Gradient boosting reduces it to {T[b]['WAPE']*100:.1f} percent, a {m['improvement_vs_baseline']['relative_wape_reduction_pct']} percent relative improvement, and cuts the average miss from {T['baseline_route_time']['MAE']:.1f} to {T[b]['MAE']:.1f} passengers. The eighty percent prediction interval covers {m['interval']['test_coverage_80']*100:.1f} percent of test trips, so the uncertainty is honest."))

# 7 error analysis
n_better = sum(m["by_route"][b][r]["WAPE"] < m["by_route"]["baseline_route_time"][r]["WAPE"] for r in m["by_route"][b])
im, d = base("Error analysis", f"Model beats the baseline on {n_better} of {len(m['by_route'][b])} routes and in peak and off-peak")
paste(im, C.FIGURES / "res_wape_by_route.png", (20, 130, 800, 460))
paste(im, C.FIGURES / "res_peak_offpeak.png", (810, 130, 1260, 460))
dis = m["by_disruption"][b]["disruption"]
bullets(d, [f"Largest gains on holidays; hardest: tourist routes and festivals",
            f"On {dis['n']} disruption trips (strikes) the model is no better than the baseline: nobody can forecast an unannounced strike"], y=490, size=24, width=90, gap=8)
slides.append((im, f"The model beats the baseline on {n_better} of twelve routes, and in both peak and off peak periods. It helps most on holidays. I also report where it fails. On the few test trips hit by a hidden disruption, such as a strike, it is no better than the baseline, because an unannounced strike cannot be forecast seven days ahead. The hardest cases are tourist routes and festivals."))

# 8 live prediction (REAL outputs)
cases = [("R01", "2025-12-24 18:00", None), ("R08", "2025-12-25 09:00", 40), ("R01", "2025-12-23 03:17", None)]
im, d = base("Live prediction: real model output", "Same code path as the API: DemandPredictor.predict()")
y = 130
narr_bits = []
for rid, dep, cap in cases:
    r = pred.predict(rid, dep, cap, explain=False)
    d.text((40, y), f"POST /predict   route={rid}   departure={dep}   capacity={r['capacity_seats']}", font=font(FM, 21), fill=ACC)
    y += 34
    act = r.get("actual_passengers_for_comparison")
    line = (f"-> forecast {r['predicted_passengers']:.0f}  (80% interval {r['interval_low']:.0f}-{r['interval_high']:.0f})   occupancy {r['predicted_occupancy']:.0%}   "
            f"flag {r['capacity_flag']}")
    d.text((40, y), line, font=font(FM, 21), fill=INK)
    y += 30
    d.text((40, y), f"   baseline {r['baseline_route_time_prediction']:.0f}" + (f"   ACTUAL {act:.0f}" if act is not None else ""), font=font(FM, 21), fill=INK)
    y += 30
    for w in r["warnings"][:1]:
        for ln in textwrap.wrap("WARNING: " + w, 95):
            d.text((40, y), "   " + ln, font=font(FM, 18), fill=(190, 80, 30))
            y += 26
    y += 30
    narr_bits.append(f"{rid} at {dep[-5:]} is forecast at {r['predicted_passengers']:.0f} passengers" + (f", and the actual was {act:.0f}" if act is not None else ""))
d.text((40, y + 6), "Different inputs give different answers; unusual inputs trigger warnings and wider intervals.", font=font(F, 22), fill=INK)
slides.append((im, "Now a live prediction, using the real saved model. " + ". ".join(narr_bits) + ". In the third request I entered a departure time that is not a scheduled slot, and the tool warns that the forecast is less reliable and widens the interval. The same predictor powers both the FastAPI service and the Streamlit app."))

# 9 explanation
imp = m["importance"]
im, d = base("Explainability", "Permutation importance on the test set + per-prediction drivers")
paste(im, C.FIGURES / "res_importance.png", (20, 130, 760, 640))
bullets(d, ["Recent same-slot demand and route dominate", "Then weekday, peak hour, holiday proximity",
            "Each forecast lists its top drivers in passengers", "Worst error: Christmas on a tourist route (rare in training, sold out)"], x=790, y=170, size=22, width=30, gap=12)
slides.append((im, "For explainability, permutation importance shows that recent demand on the same route and time, and the route itself, matter most, followed by weekday, peak hour, and holiday proximity. Every forecast also returns its top drivers measured in passengers. My largest non disruption error is Christmas day on a tourist route, which is rare in the training data and sold out, so the model under reacts."))

# 10 limits
im, d = base("Limits and responsible use")
bullets(d, ["Human review required: the tool recommends, a planner decides",
            "Synthetic data: re-validate with real operator data before any use",
            "Cannot forecast new routes, strikes, closures or unusual events; it warns instead",
            "Boarded passengers understate demand on sold-out trips; unmet demand should be monitored",
            "No personal data: only aggregate trip counts",
            "Reproducible: python run_all.py, 30 passing tests, fixed seed"], size=26)
slides.append((im, "Finally, the limits. This is decision support, so a human planner must review every forecast. The data is synthetic and must be re validated with real operator data. The model cannot forecast new routes, strikes, or unusual events, and it warns instead of guessing. Observed demand understates real demand on sold out trips. No personal data is used. Everything is reproducible with one command and thirty passing tests. Thank you."))

# ---- render
parts = []
for i, (im, text) in enumerate(slides, 1):
    png, wav, mp4 = TMP / f"s{i}.png", TMP / f"s{i}.wav", TMP / f"s{i}.mp4"
    im.save(png)
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "150", "-p", "40", "-w", str(wav), text], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-loop", "1", "-i", str(png), "-i", str(wav), "-c:v", "libx264", "-tune", "stillimage",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-shortest", "-r", "10", str(mp4)], check=True)
    parts.append(mp4)
lst = TMP / "list.txt"
lst.write_text("".join(f"file '{p}'\n" for p in parts))
out = ROOT / "reports" / "demo_video.mp4"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out)], check=True)
print("wrote", out)
