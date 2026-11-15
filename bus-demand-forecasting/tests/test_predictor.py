import pytest

from busdemand.predict import DemandPredictor


@pytest.fixture(scope="module")
def pred():
    return DemandPredictor()


def test_prediction_is_valid(pred):
    r = pred.predict("R01", "2025-12-24 18:00")
    assert 0 <= r["interval_low"] <= r["predicted_passengers"] <= r["interval_high"] <= r["capacity_seats"]
    assert r["capacity_flag"] in {"HIGH_DEMAND", "ELEVATED", "NORMAL", "LOW_DEMAND"}
    assert r["human_review_required"] and len(r["top_drivers"]) > 0


def test_model_is_really_used_not_hardcoded(pred):
    a = pred.predict("R01", "2025-12-26 18:00")["predicted_passengers"]    # Friday evening, intercity
    b = pred.predict("R01", "2025-12-23 11:30")["predicted_passengers"]    # Tuesday midday
    c = pred.predict("R11", "2025-12-23 11:00")["predicted_passengers"]    # different route
    assert len({a, b, c}) == 3
    assert pred.predict("R01", "2025-12-26 18:00")["predicted_passengers"] == a   # deterministic


def test_capacity_changes_flag_and_clips(pred):
    small = pred.predict("R01", "2025-12-24 18:00", capacity=20)
    assert small["predicted_passengers"] <= 20 and small["capacity_flag"] in {"HIGH_DEMAND", "ELEVATED"}


def test_unknown_route_and_bad_capacity(pred):
    with pytest.raises(ValueError):
        pred.predict("R99", "2025-12-24 18:00")
    with pytest.raises(ValueError):
        pred.predict("R01", "2025-12-24 18:00", capacity=500)


def test_warnings_for_unusual_inputs(pred):
    off = pred.predict("R01", "2025-12-23 03:17")          # time that is not a scheduled slot
    assert any("not a scheduled slot" in w for w in off["warnings"])
    far = pred.predict("R01", "2027-03-10 08:00")          # far beyond the data
    assert any("history ends" in w.lower() for w in far["warnings"])
    assert (far["interval_high"] - far["interval_low"]) >= (pred.predict("R01", "2025-12-23 08:00")["interval_high"] - pred.predict("R01", "2025-12-23 08:00")["interval_low"]) or far["interval_high"] == far["capacity_seats"]


def test_artifact_metrics_beat_baseline(pred):
    assert pred.art["test_metrics"]["WAPE"] < pred.art["baseline_test_metrics"]["WAPE"]
