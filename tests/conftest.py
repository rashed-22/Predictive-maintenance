"""Shared test fixtures."""

import numpy as np
import pandas as pd
import pytest

from src import config


def make_synthetic_raw() -> pd.DataFrame:
    """Small raw-style frame (300 rows, ~10 % failures) with ID and leakage columns."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n = 300
    df = pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "product_id": [f"P{i}" for i in range(n)],
            "type": rng.choice(list(config.VALID_TYPES), size=n, p=[0.6, 0.3, 0.1]),
            "air_temperature_k": rng.normal(300, 2, n),
            "process_temperature_k": rng.normal(310, 1, n),
            "rotational_speed_rpm": rng.integers(1200, 2800, n),
            "torque_nm": rng.normal(40, 10, n),
            "tool_wear_min": rng.integers(0, 250, n),
        }
    )
    # Failures are more likely at high torque, so the problem is learnable.
    df[config.TARGET_COLUMN] = (df["torque_nm"] > df["torque_nm"].quantile(0.9)).astype(int)
    for col in config.LEAKAGE_COLUMNS:
        df[col] = 0
    return df


@pytest.fixture
def synthetic_raw() -> pd.DataFrame:
    """Fresh copy of the synthetic raw frame for each test."""
    return make_synthetic_raw()


@pytest.fixture(scope="session")
def tiny_artifacts(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A tiny model trained once per test session on synthetic data, saved in a temp dir.

    Never touches ``models/``; returns paths, metadata and the frame used.
    """
    from src.train import train_final

    folder = tmp_path_factory.mktemp("tiny_model")
    df = make_synthetic_raw()
    model_path = folder / "model.joblib"
    metadata_path = folder / "metadata.json"
    meta = train_final(
        df,
        n_iter=2,
        n_splits=2,
        model_path=model_path,
        metadata_path=metadata_path,
        threshold_plot_path=folder / "threshold.png",
    )
    return {
        "dir": folder,
        "model_path": model_path,
        "metadata_path": metadata_path,
        "meta": meta,
        "df": df,
    }
