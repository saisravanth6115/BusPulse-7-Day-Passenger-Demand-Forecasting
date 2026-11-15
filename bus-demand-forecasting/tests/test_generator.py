import pandas as pd

from busdemand.data_generator import generate, inject_quality_problems


def test_generator_is_deterministic_and_valid():
    a, _, ea = generate(seed=7, start="2024-01-01", end="2024-01-31")
    b, _, eb = generate(seed=7, start="2024-01-01", end="2024-01-31")
    c, _, _ = generate(seed=8, start="2024-01-01", end="2024-01-31")
    pd.testing.assert_frame_equal(a, b)
    assert not a["passengers"].equals(c["passengers"])
    assert (a["passengers"] <= a["capacity_seats"]).all() and (a["passengers"] >= 0).all()
    assert len(a) > 1000


def test_injected_problems_exist():
    a, _, _ = generate(seed=7, start="2024-01-01", end="2024-03-31")
    raw = inject_quality_problems(a, seed=7)
    assert raw["trip_id"].duplicated().any() and raw["passengers"].isna().any() and raw["capacity_seats"].isna().any()
