import pytest
from fastapi.testclient import TestClient

from api.main import app
from busdemand.predict import get_predictor

client = TestClient(app)


def test_health_and_routes():
    assert client.get("/health").json()["status"] == "ok"
    assert len(client.get("/routes").json()) == 12


def test_predict_matches_saved_model():
    body = {"route_id": "R03", "departure": "2025-12-24T17:30:00", "capacity_seats": 45}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    direct = get_predictor().predict("R03", "2025-12-24 17:30", 45)
    assert r.json()["predicted_passengers"] == direct["predicted_passengers"]


def test_different_inputs_give_different_outputs():
    a = client.post("/predict", json={"route_id": "R03", "departure": "2025-12-24T17:30:00"}).json()["predicted_passengers"]
    b = client.post("/predict", json={"route_id": "R05", "departure": "2025-12-24T08:00:00"}).json()["predicted_passengers"]
    assert a != b


def test_validation_errors():
    assert client.post("/predict", json={"route_id": "R99", "departure": "2025-12-24T17:30:00"}).status_code == 422
    assert client.post("/predict", json={"route_id": "R01", "departure": "not-a-date"}).status_code == 422
    assert client.post("/predict", json={"route_id": "R01", "departure": "2025-12-24T17:30:00", "capacity_seats": 5}).status_code == 422


def test_batch_and_model_info():
    r = client.post("/predict/batch", json={"items": [{"route_id": "R01", "departure": "2025-12-24T18:00:00"},
                                                      {"route_id": "R99", "departure": "2025-12-24T18:00:00"}]}).json()
    assert "predicted_passengers" in r[0] and "error" in r[1]
    info = client.get("/model-info").json()
    assert info["test_metrics"]["WAPE"] < info["baseline_test_metrics"]["WAPE"]
