"""Tests for the FastAPI service (Phase 6). They use a tiny synthetic model, never models/."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api import main
from src import config
from src.predict import load_model

VALID = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}


@pytest.fixture
def client(tiny_artifacts: dict, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client whose lifespan loads the tiny model instead of models/model.joblib."""
    monkeypatch.setattr(
        main,
        "load_model",
        lambda: load_model(tiny_artifacts["model_path"], tiny_artifacts["metadata_path"]),
    )
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def client_no_model(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Client for an app whose model failed to load."""

    def _missing() -> None:
        raise FileNotFoundError("no model")

    monkeypatch.setattr(main, "load_model", _missing)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_model_info(client: TestClient, tiny_artifacts: dict) -> None:
    response = client.get("/model-info")
    assert response.status_code == 200
    body = response.json()
    assert body["version"] == tiny_artifacts["meta"]["version"]
    assert body["threshold"] == pytest.approx(tiny_artifacts["meta"]["threshold"])
    assert "torque_nm" in body["features"]
    assert isinstance(body["test_metrics"], dict)


def test_predict_valid(client: TestClient) -> None:
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["prediction"] in (0, 1)
    assert body["label"] == ("failure" if body["prediction"] else "no_failure")
    assert body["prediction"] == int(body["failure_probability"] >= body["threshold"])
    assert body["model_version"]


def test_predict_batch(client: TestClient) -> None:
    high_torque = {**VALID, "torque_nm": 70.0}
    response = client.post("/predict/batch", json={"readings": [VALID, high_torque, VALID]})
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    # Same input -> same output, and order is preserved.
    assert results[0] == results[2]
    single = client.post("/predict", json=high_torque).json()
    assert results[1] == single


def test_batch_over_limit(client: TestClient) -> None:
    readings = [VALID] * (config.API_MAX_BATCH_SIZE + 1)
    assert client.post("/predict/batch", json={"readings": readings}).status_code == 422


def test_batch_at_limit_ok(client: TestClient) -> None:
    readings = [VALID] * config.API_MAX_BATCH_SIZE
    response = client.post("/predict/batch", json={"readings": readings})
    assert response.status_code == 200
    assert len(response.json()["results"]) == config.API_MAX_BATCH_SIZE


def test_batch_empty_is_422(client: TestClient) -> None:
    assert client.post("/predict/batch", json={"readings": []}).status_code == 422


def test_422_bad_type(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID, "torque_nm": "abc"}).status_code == 422
    assert client.post("/predict", json={**VALID, "type": "X"}).status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("air_temperature_k", 500.0),
        ("process_temperature_k", 100.0),
        ("rotational_speed_rpm", 99999),
        ("torque_nm", -5.0),
        ("tool_wear_min", 1000),
    ],
)
def test_422_out_of_range(client: TestClient, field: str, value: float) -> None:
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_422_missing_field(client: TestClient) -> None:
    payload = {k: v for k, v in VALID.items() if k != "torque_nm"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "torque_nm" in response.text


def test_422_unknown_field(client: TestClient) -> None:
    assert client.post("/predict", json={**VALID, "twf": 1}).status_code == 422


def test_503_when_model_missing(client_no_model: TestClient) -> None:
    assert client_no_model.get("/health").json() == {"status": "ok", "model_loaded": False}
    assert client_no_model.get("/model-info").status_code == 503
    assert client_no_model.post("/predict", json=VALID).status_code == 503
    assert client_no_model.post("/predict/batch", json={"readings": [VALID]}).status_code == 503


def test_cors_header_present(client: TestClient) -> None:
    response = client.get("/health", headers={"Origin": "http://localhost:8501"})
    assert response.headers.get("access-control-allow-origin") == "*"


def test_request_log_has_no_payload(client: TestClient, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level("INFO", logger="api"):
        client.post("/predict", json=VALID)
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "POST /predict -> 200" in messages
    assert "42.8" not in messages
