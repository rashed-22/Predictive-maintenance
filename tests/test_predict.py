"""Tests for src/predict.py - use a tiny synthetic model, never models/model.joblib."""

import pandas as pd
import pytest

from src import config
from src.predict import LoadedModel, load_model, predict_df


@pytest.fixture
def loaded(tiny_artifacts) -> LoadedModel:
    return load_model(tiny_artifacts["model_path"], tiny_artifacts["metadata_path"])


def test_load_model_is_cached(tiny_artifacts, loaded):
    again = load_model(tiny_artifacts["model_path"], tiny_artifacts["metadata_path"])
    assert again is loaded
    assert loaded.threshold == tiny_artifacts["meta"]["threshold"]


def test_load_model_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="src.train"):
        load_model(tmp_path / "nope.joblib", tmp_path / "nope.json")


def test_predict_df_output_columns_and_index(tiny_artifacts, loaded):
    df = tiny_artifacts["df"].iloc[10:20]  # includes ID/leakage/target columns -> ignored
    out = predict_df(df, loaded)
    assert list(out.columns) == ["failure_probability", "prediction", "label"]
    assert out.index.equals(df.index)
    assert out["failure_probability"].between(0, 1).all()
    assert set(out["prediction"]) <= {0, 1}
    assert set(out["label"]) <= {"failure", "no_failure"}


def test_predict_df_applies_stored_threshold(tiny_artifacts, loaded):
    out = predict_df(tiny_artifacts["df"], loaded)
    expected = (out["failure_probability"] >= loaded.threshold).astype(int)
    assert (out["prediction"] == expected).all()
    assert (out.loc[out["prediction"] == 1, "label"] == "failure").all()
    assert (out.loc[out["prediction"] == 0, "label"] == "no_failure").all()


def test_predict_df_single_row_and_input_not_mutated(tiny_artifacts, loaded):
    row = tiny_artifacts["df"][config.RAW_FEATURE_COLUMNS].iloc[[0]].copy()
    before = row.copy()
    out = predict_df(row, loaded)
    assert len(out) == 1
    pd.testing.assert_frame_equal(row, before)


def test_predict_df_missing_column_raises(tiny_artifacts, loaded):
    df = tiny_artifacts["df"].drop(columns=["torque_nm"])
    with pytest.raises(ValueError, match="torque_nm"):
        predict_df(df, loaded)
