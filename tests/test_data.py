"""Tests for src/data.py. All data is built in memory; data/raw is never needed."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data import load_raw, split, validate


def make_valid_df(n: int = 1000, failure_rate: float = 0.035) -> pd.DataFrame:
    """Build a small synthetic dataframe that satisfies the schema."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    n_fail = round(n * failure_rate)
    failure = np.zeros(n, dtype=int)
    failure[:n_fail] = 1
    rng.shuffle(failure)
    return pd.DataFrame(
        {
            "udi": np.arange(1, n + 1),
            "product_id": [f"L{i}" for i in range(n)],
            "type": rng.choice(config.VALID_TYPES, size=n),
            "air_temperature_k": rng.uniform(296, 304, n),
            "process_temperature_k": rng.uniform(306, 314, n),
            "rotational_speed_rpm": rng.integers(1200, 2800, n),
            "torque_nm": rng.uniform(10, 70, n),
            "tool_wear_min": rng.integers(0, 250, n),
            "machine_failure": failure,
            **{flag: np.zeros(n, dtype=int) for flag in config.LEAKAGE_COLUMNS},
        }
    )


def test_valid_dataframe_passes() -> None:
    assert validate(make_valid_df()) is None


def test_missing_column_raises() -> None:
    df = make_valid_df().drop(columns=["torque_nm"])
    with pytest.raises(ValueError, match="Missing required columns.*torque_nm"):
        validate(df)


def test_bad_type_value_raises() -> None:
    df = make_valid_df()
    df.loc[0, "type"] = "X"
    with pytest.raises(ValueError, match="'type' has invalid values.*X"):
        validate(df)


def test_out_of_range_value_raises() -> None:
    df = make_valid_df()
    df.loc[0, "rotational_speed_rpm"] = 9999
    with pytest.raises(ValueError, match="rotational_speed_rpm.*outside"):
        validate(df)


def test_missing_values_raise() -> None:
    df = make_valid_df()
    df.loc[3, "air_temperature_k"] = np.nan
    with pytest.raises(ValueError, match="Missing values found.*air_temperature_k"):
        validate(df)


def test_duplicate_udi_raises() -> None:
    df = make_valid_df()
    df.loc[1, "udi"] = df.loc[0, "udi"]
    with pytest.raises(ValueError, match="duplicate 'udi'"):
        validate(df)


def test_non_binary_target_raises() -> None:
    df = make_valid_df()
    df.loc[0, "machine_failure"] = 2
    with pytest.raises(ValueError, match="machine_failure.*0/1"):
        validate(df)


def test_non_numeric_sensor_raises() -> None:
    df = make_valid_df()
    df["torque_nm"] = df["torque_nm"].astype(str)
    with pytest.raises(ValueError, match="'torque_nm' must be numeric"):
        validate(df)


def test_all_failures_are_reported_together() -> None:
    df = make_valid_df()
    df.loc[0, "type"] = "X"
    df.loc[1, "torque_nm"] = 500.0
    with pytest.raises(ValueError) as exc:
        validate(df)
    assert "'type'" in str(exc.value) and "'torque_nm'" in str(exc.value)


def test_split_is_stratified_and_disjoint() -> None:
    df = make_valid_df(n=10_000, failure_rate=0.034)
    train, test = split(df)
    assert len(train) == 8000 and len(test) == 2000
    overall = df["machine_failure"].mean()
    assert abs(train["machine_failure"].mean() - overall) < 0.005
    assert abs(test["machine_failure"].mean() - overall) < 0.005
    assert set(train.index).isdisjoint(test.index)


def test_split_is_reproducible() -> None:
    df = make_valid_df()
    first = split(df)[1].index
    second = split(df)[1].index
    assert first.equals(second)


def test_load_raw_renames_columns(tmp_path) -> None:
    inverse = {v: k for k, v in config.COLUMN_MAPPING.items()}
    csv_path = tmp_path / "raw.csv"
    make_valid_df(n=20).rename(columns=inverse).to_csv(csv_path, index=False)
    df = load_raw(csv_path)
    assert list(df.columns) == list(config.COLUMN_MAPPING.values())


def test_load_raw_missing_file_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_raw(tmp_path / "nope.csv")
