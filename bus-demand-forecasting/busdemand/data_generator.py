"""Synthetic trip-level bus ridership generator (no real/personal data).

One row = one scheduled bus trip (route + departure date/time).
Target `passengers` = passengers who actually boarded (capped at seat capacity).

Demand is built from interpretable multiplicative effects:
    route size x time-of-day x weekday x season x trend x holiday x hidden shocks
Hidden shocks (rain, strikes, landslides) are NOT available at forecast time and are
never used as model features; they only create realistic unexplained error.

Raw output deliberately contains data-quality problems (duplicates, missing values,
impossible values) so the cleaning step has real work to do.

Run:  python -m busdemand.data_generator
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from . import config as C
from .holidays_cal import holiday_table

# route_id, origin, destination, km, type, neutral mean demand per trip, default seats, alt seats, slots
ROUTES = [
    ("R01", "Chandigarh", "Delhi", 250, "intercity", 24, 45, 52, ["05:30", "08:00", "11:30", "15:00", "18:00", "21:30"]),
    ("R02", "Ludhiana", "Chandigarh", 100, "commuter", 21, 40, 45, ["06:00", "07:30", "09:00", "12:00", "15:30", "17:00", "18:30"]),
    ("R03", "Jalandhar", "Delhi", 370, "intercity", 23, 45, 52, ["06:00", "09:30", "13:00", "17:30", "21:00", "23:00"]),
    ("R04", "Amritsar", "Delhi", 450, "intercity", 22, 52, 45, ["06:30", "10:00", "16:00", "19:00", "22:00"]),
    ("R05", "Amritsar", "Jalandhar", 80, "commuter", 19, 40, 45, ["06:30", "08:00", "10:30", "13:00", "16:30", "18:00"]),
    ("R06", "Ludhiana", "Delhi", 310, "intercity", 24, 45, 52, ["05:00", "08:30", "12:00", "16:00", "19:30", "22:30"]),
    ("R07", "Patiala", "Chandigarh", 65, "commuter", 17, 40, 45, ["06:30", "08:00", "09:30", "13:30", "17:00", "18:30"]),
    ("R08", "Chandigarh", "Shimla", 115, "tourist", 19, 40, 45, ["06:30", "09:00", "12:00", "15:00", "19:00"]),
    ("R09", "Chandigarh", "Manali", 310, "tourist", 20, 45, 52, ["06:00", "09:00", "17:30", "20:30"]),
    ("R10", "Delhi", "Jaipur", 280, "intercity", 24, 45, 52, ["05:30", "08:30", "12:30", "16:30", "20:00", "23:00"]),
    ("R11", "Jalandhar", "Ludhiana", 60, "commuter", 16, 40, 45, ["06:00", "07:00", "08:30", "11:00", "14:30", "17:30", "19:00"]),
    ("R12", "Delhi", "Haridwar", 220, "tourist", 19, 45, 52, ["05:30", "07:30", "10:30", "14:00", "21:00"]),
]
ROUTE_COLS = ["route_id", "origin", "destination", "distance_km", "route_type", "base_demand",
              "default_capacity", "alt_capacity", "slots"]

# relative demand by weekday (Mon..Sun)
DOW = {
    "commuter": [1.15, 1.05, 1.05, 1.08, 1.20, 0.85, 0.70],
    "intercity": [1.00, 0.88, 0.85, 0.95, 1.30, 1.05, 1.30],
    "tourist": [0.80, 0.75, 0.75, 0.85, 1.25, 1.45, 1.20],
}
# relative demand by month (Jan..Dec)
SEASON = {
    "commuter": [1.00, 1.00, 1.00, 1.00, 0.95, 0.80, 0.95, 1.00, 1.02, 1.00, 1.00, 0.95],
    "intercity": [0.95, 0.95, 1.00, 1.00, 1.08, 1.05, 0.95, 0.95, 1.00, 1.05, 1.08, 1.12],
    "tourist": [0.70, 0.75, 0.95, 1.20, 1.50, 1.45, 0.85, 0.80, 1.00, 1.10, 1.05, 1.45],
}
HOLIDAY_EFFECT = {  # (on holiday, 1 day before, 2 days before, 1 day after, major-festival-week bonus)
    "commuter": (0.60, 1.05, 1.00, 1.00, 1.00),
    "intercity": (1.00, 1.35, 1.15, 1.25, 1.20),
    "tourist": (1.25, 1.40, 1.20, 1.20, 1.10),
}
RAIN_EFFECT = {"commuter": 0.93, "intercity": 0.97, "tourist": 0.85}
TREND_PER_YEAR = 0.05


def hour_factor(hour: float, rtype: str) -> float:
    """Time-of-day effect, depends on route type."""
    if rtype == "commuter":
        if 6 <= hour < 10 or 16 <= hour < 20:
            return 1.45
        if 10 <= hour < 16:
            return 0.80
        return 0.60
    if rtype == "intercity":
        if 5 <= hour < 10:
            return 1.00
        if 10 <= hour < 16:
            return 0.85
        if 16 <= hour < 21:
            return 1.25
        return 1.05 if hour < 23 else 0.85
    # tourist
    if 5 <= hour < 10:
        return 1.30
    if 10 <= hour < 17:
        return 0.80
    return 0.95


def routes_frame() -> pd.DataFrame:
    df = pd.DataFrame(ROUTES, columns=ROUTE_COLS)
    df["slots"] = df["slots"].apply(lambda s: "|".join(s))
    return df


def _holiday_multiplier(rtype: str, tbl: pd.DataFrame) -> np.ndarray:
    on, b1, b2, a1, fest = HOLIDAY_EFFECT[rtype]
    m = np.ones(len(tbl))
    to_next = tbl["days_to_next_holiday"].values
    since = tbl["days_since_prev_holiday"].values
    m = np.where(to_next == 1, m * b1, m)
    m = np.where(to_next == 2, m * b2, m)
    m = np.where(since == 1, m * a1, m)
    m = np.where((to_next <= 3) & (tbl["next_holiday_major"].values == 1), m * fest, m)
    m = np.where(tbl["is_holiday"].values == 1, m * on, m)
    return m


def generate(seed: int = C.SEED, start: str = C.START_DATE, end: str = C.END_DATE):
    """Return (clean_trips, routes, events). Clean = before data-quality problems are injected."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n_days = len(dates)
    cal = holiday_table(dates)
    dow = dates.dayofweek.values
    month = dates.month.values
    years = (dates - dates[0]).days.values / 365.25

    # hidden shock 1: rain (shared by all routes on a day); monsoon months rain more often
    p_rain = np.where(np.isin(month, [7, 8, 9]), 0.40, 0.12)
    rain = rng.random(n_days) < p_rain

    # hidden shock 2: route-level disruptions (strikes, road closures, landslides)
    events = []
    for rid, _, _, _, rtype, *_ in ROUTES:
        n_ev = 7 + (4 if rtype == "tourist" and rid in ("R08", "R09") else 0)
        for _ in range(n_ev):
            s = int(rng.integers(0, n_days - 4))
            if rid in ("R08", "R09") and rng.random() < 0.5:   # hill routes: monsoon landslides
                yr = int(rng.integers(0, max(1, n_days // 365)))
                s = min(n_days - 4, yr * 365 + int(rng.integers(190, 240)))
            dur = int(rng.integers(1, 4))
            events.append((rid, dates[s], dates[min(n_days - 1, s + dur - 1)], round(float(rng.uniform(0.10, 0.50)), 2)))
    events_df = pd.DataFrame(events, columns=["route_id", "start_date", "end_date", "demand_multiplier"])

    frames = []
    for rid, origin, dest, km, rtype, base, cap_def, cap_alt, slots in ROUTES:
        hol = _holiday_multiplier(rtype, cal)
        ev_mult = np.ones(n_days)
        for _, e in events_df[events_df.route_id == rid].iterrows():
            mask = (dates >= e.start_date) & (dates <= e.end_date)
            ev_mult[mask] = np.minimum(ev_mult[mask], e.demand_multiplier)
        day_mult = (np.array(DOW[rtype])[dow] * np.array(SEASON[rtype])[month - 1] * (1 + TREND_PER_YEAR * years)
                    * hol * np.where(rain, RAIN_EFFECT[rtype], 1.0) * ev_mult)
        for slot in slots:
            hh, mm = map(int, slot.split(":"))
            mu = base * hour_factor(hh + mm / 60, rtype) * day_mult
            mu = mu * rng.lognormal(0.0, 0.12, n_days)               # trip-level random effect
            demand = rng.poisson(mu)                                   # latent demand
            cap = np.where(rng.random(n_days) < 0.8, cap_def, cap_alt)  # fleet assignment
            pax = np.minimum(demand, cap)                              # observed = capped by seats
            dep = dates + pd.Timedelta(hours=hh, minutes=mm)
            frames.append(pd.DataFrame({
                "trip_id": [f"{rid}-{d:%Y%m%d}-{hh:02d}{mm:02d}" for d in dates],
                "route_id": rid, "origin": origin, "destination": dest, "distance_km": km,
                "departure_datetime": dep, "capacity_seats": cap, "passengers": pax,
            }))
    trips = pd.concat(frames, ignore_index=True).sort_values(["departure_datetime", "route_id"]).reset_index(drop=True)
    return trips, routes_frame(), events_df


def inject_quality_problems(trips: pd.DataFrame, seed: int = C.SEED) -> pd.DataFrame:
    """Make the raw file messy like a real booking-system export."""
    rng = np.random.default_rng(seed + 1)
    df = trips.copy()
    df["passengers"] = df["passengers"].astype("float64")
    df["capacity_seats"] = df["capacity_seats"].astype("float64")
    df["distance_km"] = df["distance_km"].astype("float64")
    n = len(df)
    pick = lambda frac: rng.choice(n, size=int(n * frac), replace=False)  # noqa: E731

    df.loc[pick(0.008), "passengers"] = np.nan                    # missing target (e.g. logging gap)
    df.loc[pick(0.006), "capacity_seats"] = np.nan                # missing capacity
    df.loc[pick(0.003), "distance_km"] = np.nan                   # missing distance
    bad = pick(0.003)                                             # impossible values
    cap = df.loc[bad, "capacity_seats"].fillna(45).values
    df.loc[bad, "passengers"] = np.where(rng.random(len(bad)) < 0.5, -1.0, np.round(cap * rng.uniform(1.4, 3.0, len(bad))))
    dups = df.iloc[pick(0.004)]                                   # exact duplicate rows
    df = pd.concat([df, dups], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["departure_datetime"] = df["departure_datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    C.DATA_RAW.mkdir(parents=True, exist_ok=True)
    trips, routes, events = generate(seed=args.seed)
    raw = inject_quality_problems(trips, seed=args.seed)
    raw.to_csv(C.RAW_TRIPS, index=False)
    routes.to_csv(C.ROUTES_CSV, index=False)
    events.to_csv(C.EVENTS_CSV, index=False)
    print(f"wrote {len(raw):,} raw rows -> {C.RAW_TRIPS}")
    print(f"wrote {len(routes)} routes, {len(events)} disruption events (ground truth, used only for error analysis)")


if __name__ == "__main__":
    main()
