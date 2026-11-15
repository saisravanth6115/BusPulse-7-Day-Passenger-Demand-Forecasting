import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def small_history():
    """Tiny deterministic history: route R01, slot 08:00, passengers == day number % 50 (easy to verify)."""
    dates = pd.date_range("2025-01-01", "2025-03-31")
    df = pd.DataFrame({
        "route_id": "R01", "slot": "08:00", "date": dates,
        "passengers": [float(d.toordinal() % 50) for d in dates], "capacity_seats": 45.0,
    })
    return df
