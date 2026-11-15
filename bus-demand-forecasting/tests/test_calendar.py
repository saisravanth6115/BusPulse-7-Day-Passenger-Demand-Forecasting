import pandas as pd

from busdemand.holidays_cal import calendar_covered, holiday_table


def test_diwali_flags():
    t = holiday_table(pd.to_datetime(["2025-10-19", "2025-10-20", "2025-10-21"]))
    assert t.loc[1, "is_holiday"] == 1 and t.loc[1, "is_major_festival"] == 1
    assert t.loc[0, "days_to_next_holiday"] == 1 and t.loc[0, "next_holiday_major"] == 1
    assert t.loc[2, "days_since_prev_holiday"] == 1


def test_values_capped_and_coverage():
    t = holiday_table(pd.to_datetime(["2025-06-10"]))
    assert 1 <= t.loc[0, "days_to_next_holiday"] <= 10
    assert calendar_covered("2025-05-05") and not calendar_covered("2035-01-01")
