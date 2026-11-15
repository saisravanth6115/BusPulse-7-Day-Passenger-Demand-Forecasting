"""Small, deterministic Indian (north-India flavoured) holiday calendar.

The dates are APPROXIMATE and only used for this synthetic project. A holiday
calendar is known months in advance, so using it as a feature is NOT leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAL_START = pd.Timestamp("2022-01-01")
CAL_END = pd.Timestamp("2027-12-31")

_MAJOR = {  # movable festivals with a big travel effect
    "Holi": ["2022-03-18", "2023-03-08", "2024-03-25", "2025-03-14", "2026-03-04", "2027-03-22"],
    "Raksha Bandhan": ["2022-08-11", "2023-08-30", "2024-08-19", "2025-08-09", "2026-08-28", "2027-08-17"],
    "Dussehra": ["2022-10-05", "2023-10-24", "2024-10-12", "2025-10-02", "2026-10-20", "2027-10-09"],
    "Diwali": ["2022-10-24", "2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08", "2027-10-29"],
    "Eid al-Fitr": ["2022-05-03", "2023-04-22", "2024-04-11", "2025-03-31", "2026-03-20", "2027-03-10"],
    "Guru Nanak Jayanti": ["2022-11-08", "2023-11-27", "2024-11-15", "2025-11-05", "2026-11-24", "2027-11-14"],
}
# New Year, Republic Day, Baisakhi, Independence Day, Gandhi Jayanti, Christmas
_FIXED = ["01-01", "01-26", "04-14", "08-15", "10-02", "12-25"]


def _build():
    major = {pd.Timestamp(d) for v in _MAJOR.values() for d in v}
    minor = {pd.Timestamp(f"{y}-{md}") for y in range(2022, 2028) for md in _FIXED}
    all_h = sorted(major | minor)
    hols = np.array(all_h, dtype="datetime64[D]")
    major64 = {np.datetime64(m, "D") for m in major}
    is_major = np.array([h in major64 for h in hols])
    return hols, is_major


_HOLS, _IS_MAJOR = _build()


def calendar_covered(date) -> bool:
    """True if the holiday calendar has data for this date."""
    return CAL_START <= pd.Timestamp(date) <= CAL_END


def holiday_table(dates) -> pd.DataFrame:
    """Vectorised calendar features for an array-like of dates (time part ignored).

    days_to_next_holiday   : days until the next holiday strictly after the date (capped at 10)
    days_since_prev_holiday: days since the previous holiday strictly before the date (capped at 10)
    """
    d = pd.DatetimeIndex(pd.to_datetime(dates)).normalize().values.astype("datetime64[D]")
    hol_list = set(_HOLS.tolist())
    major_list = set(_HOLS[_IS_MAJOR].tolist())
    dl = d.tolist()
    is_hol = np.array([x in hol_list for x in dl], dtype=int)
    is_major = np.array([x in major_list for x in dl], dtype=int)

    n = len(_HOLS)
    idx_next = np.searchsorted(_HOLS, d, side="right")      # first holiday > date
    idx_prev = np.searchsorted(_HOLS, d, side="left") - 1   # last holiday < date
    big = np.timedelta64(10, "D")
    nxt = np.where(idx_next < n, _HOLS[np.clip(idx_next, 0, n - 1)], d + big)
    prv = np.where(idx_prev >= 0, _HOLS[np.clip(idx_prev, 0, n - 1)], d - big)
    to_next = np.minimum((nxt - d).astype(int), 10)
    since_prev = np.minimum((d - prv).astype(int), 10)
    next_major = np.where(idx_next < n, _IS_MAJOR[np.clip(idx_next, 0, n - 1)], False).astype(int)
    return pd.DataFrame({
        "is_holiday": is_hol,
        "is_major_festival": is_major,
        "days_to_next_holiday": to_next,
        "days_since_prev_holiday": since_prev,
        "next_holiday_major": next_major,
    })
