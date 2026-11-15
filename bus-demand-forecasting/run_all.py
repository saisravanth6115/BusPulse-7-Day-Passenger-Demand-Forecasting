"""One command to reproduce everything:  python run_all.py

generate data -> clean -> features -> train/evaluate -> figures -> notebooks -> report -> tests
Use --skip-notebooks to save a minute.
"""
import subprocess
import sys

steps = [
    ("1/6 generate synthetic dataset", [sys.executable, "-m", "busdemand.data_generator"]),
    ("2/6 train + evaluate + figures (includes cleaning & leakage audit)", [sys.executable, "-m", "busdemand.train"]),
    ("3/6 build technical report", [sys.executable, "-m", "busdemand.make_report"]),
]
if "--skip-notebooks" not in sys.argv:
    steps.append(("4/6 build + execute notebooks", [sys.executable, "scripts/build_notebooks.py"]))
steps.append(("5/6 run tests", [sys.executable, "-m", "pytest", "-q"]))

for title, cmd in steps:
    print(f"\n=== {title} ===", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"step failed: {title}")
print("\nAll steps finished OK. Next: uvicorn api.main:app   or   streamlit run app/streamlit_app.py")
