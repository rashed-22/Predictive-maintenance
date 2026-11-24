"""Pydantic v2 request/response models for the API (contract: docs/ARCHITECTURE.md section 4)."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src import config

_AIR = config.VALUE_RANGES["air_temperature_k"]
_PROC = config.VALUE_RANGES["process_temperature_k"]
_RPM = config.VALUE_RANGES["rotational_speed_rpm"]
_TORQUE = config.VALUE_RANGES["torque_nm"]
_WEAR = config.VALUE_RANGES["tool_wear_min"]

_EXAMPLE_READING = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1551,
    "torque_nm": 42.8,
    "tool_wear_min": 0,
}


class SensorReading(BaseModel):
    """One raw sensor reading. Ranges come from ``src/config.py``; unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": _EXAMPLE_READING})

    type: Literal["L", "M", "H"] = Field(description="Product quality variant")
    air_temperature_k: float = Field(ge=_AIR[0], le=_AIR[1], description="Air temperature [K]")
    process_temperature_k: float = Field(
        ge=_PROC[0], le=_PROC[1], description="Process temperature [K]"
    )
    rotational_speed_rpm: float = Field(
        ge=_RPM[0], le=_RPM[1], description="Rotational speed [rpm]"
    )
    torque_nm: float = Field(ge=_TORQUE[0], le=_TORQUE[1], description="Torque [Nm]")
    tool_wear_min: float = Field(ge=_WEAR[0], le=_WEAR[1], description="Tool wear [min]")


class PredictionResponse(BaseModel):
    """Prediction for one reading."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "failure_probability": 0.03,
                "prediction": 0,
                "label": "no_failure",
                "threshold": 0.26,
                "model_version": "1.0.0",
            }
        }
    )

    failure_probability: float = Field(ge=0.0, le=1.0)
    prediction: Literal[0, 1] = Field(description="1 if probability >= threshold")
    label: Literal["failure", "no_failure"]
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    """Several readings in one call (1 to ``API_MAX_BATCH_SIZE``)."""

    model_config = ConfigDict(json_schema_extra={"example": {"readings": [_EXAMPLE_READING]}})

    readings: list[SensorReading] = Field(min_length=1, max_length=config.API_MAX_BATCH_SIZE)


class BatchResponse(BaseModel):
    """Predictions in the same order as the request."""

    results: list[PredictionResponse]


class ModelInfo(BaseModel):
    """Model metadata taken from ``models/metadata.json``."""

    model_name: str
    version: str
    trained_at: str
    threshold: float
    features: list[str]
    test_metrics: dict[str, float]


class Health(BaseModel):
    """Liveness and readiness."""

    status: Literal["ok"]
    model_loaded: bool
