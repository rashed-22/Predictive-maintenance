"""Tests for src/features.py."""

import math

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src import config
from src.features import FEATURE_COLUMNS, build_features

ROW = {
    "type": "M",
    "air_temperature_k": 298.1,
    "process_temperature_k": 308.6,
    "rotational_speed_rpm": 1500,
    "torque_nm": 40.0,
    "tool_wear_min": 100,
}


def _full_frame() -> pd.DataFrame:
    """Raw-style frame including ID, target and leakage columns."""
    row = {"udi": 1, "product_id": "M14860", **ROW, config.TARGET_COLUMN: 0}
    row.update({c: 0 for c in config.LEAKAGE_COLUMNS})
    return pd.DataFrame([row, {**row, "udi": 2, "type": "H"}], index=[10, 20])


def test_temp_diff_k_exact():
    out = build_features(pd.DataFrame([ROW]))
    assert out["temp_diff_k"].iloc[0] == pytest.approx(10.5, abs=1e-9)  # 308.6 - 298.1


def test_power_w_hand_computed():
    # 40 Nm * 1500 rpm = 40 * 157.0796... rad/s = 6283.185... W (= 2000 * pi)
    out = build_features(pd.DataFrame([ROW]))
    assert out["power_w"].iloc[0] == pytest.approx(2000 * math.pi, rel=1e-12)
    assert out["power_w"].iloc[0] == pytest.approx(6283.1853, abs=1e-4)


def test_strain_exact():
    out = build_features(pd.DataFrame([ROW]))
    assert out["strain"].iloc[0] == pytest.approx(4000.0)  # 100 min * 40 Nm


def test_column_order_equals_feature_columns():
    assert list(build_features(_full_frame()).columns) == FEATURE_COLUMNS
    assert list(build_features(pd.DataFrame([ROW])).columns) == FEATURE_COLUMNS


def test_input_not_mutated():
    df = _full_frame()
    before = df.copy(deep=True)
    build_features(df)
    assert_frame_equal(df, before)


def test_leakage_target_and_id_columns_removed():
    out = build_features(_full_frame())
    forbidden = config.LEAKAGE_COLUMNS + config.ID_COLUMNS + [config.TARGET_COLUMN]
    assert not set(forbidden) & set(out.columns)


def test_works_without_target_leakage_or_id_columns():
    out = build_features(pd.DataFrame([ROW]))
    assert out.shape == (1, len(FEATURE_COLUMNS))


def test_single_row_has_all_type_columns():
    out = build_features(pd.DataFrame([ROW]))
    assert out.loc[0, ["type_L", "type_M", "type_H"]].tolist() == [0.0, 1.0, 0.0]


def test_multiple_rows_keep_index_and_encode_types():
    out = build_features(_full_frame())
    assert list(out.index) == [10, 20]
    assert out["type_M"].tolist() == [1.0, 0.0]
    assert out["type_H"].tolist() == [0.0, 1.0]


def test_missing_column_raises():
    with pytest.raises(ValueError, match="torque_nm"):
        build_features(pd.DataFrame([ROW]).drop(columns="torque_nm"))


def test_invalid_type_raises():
    with pytest.raises(ValueError, match="Invalid type"):
        build_features(pd.DataFrame([{**ROW, "type": "X"}]))
