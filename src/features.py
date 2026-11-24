"""Feature engineering: the single source of truth for model input features.

``build_features`` is a pure function: no fitting, no I/O, no mutation of its input.
It is used identically by training, the API and the dashboard.
"""

import math

import pandas as pd

from src import config

# One-hot columns for the product quality variant, in a fixed order.
TYPE_COLUMNS: list[str] = [f"type_{t}" for t in config.VALID_TYPES]

# Final ordered list of columns fed to the model.
FEATURE_COLUMNS: list[str] = (
    TYPE_COLUMNS + config.NUMERIC_SENSOR_COLUMNS + config.ENGINEERED_COLUMNS
)

_RPM_TO_RAD_PER_S: float = 2 * math.pi / 60


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the model-ready feature frame (columns == FEATURE_COLUMNS).

    Accepts one or many rows. Target, leakage flags, IDs and any other extra
    columns are ignored. The input is never modified; the row index is kept.

    Raises:
        ValueError: if a raw feature column is missing or ``type`` is not L/M/H.
    """
    missing = [c for c in config.RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    types = df["type"]
    invalid = sorted({str(v) for v in types[~types.isin(config.VALID_TYPES)]})
    if invalid:
        raise ValueError(f"Invalid type value(s) {invalid}; expected one of {config.VALID_TYPES}")

    out = pd.DataFrame(index=df.index)
    for t, col in zip(config.VALID_TYPES, TYPE_COLUMNS, strict=True):
        out[col] = (types == t).astype("float64")
    for col in config.NUMERIC_SENSOR_COLUMNS:
        out[col] = df[col].astype("float64")

    out["temp_diff_k"] = out["process_temperature_k"] - out["air_temperature_k"]
    out["power_w"] = out["torque_nm"] * out["rotational_speed_rpm"] * _RPM_TO_RAD_PER_S
    out["strain"] = out["tool_wear_min"] * out["torque_nm"]
    return out[FEATURE_COLUMNS]
