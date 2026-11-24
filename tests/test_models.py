"""Tests for src/models.py."""

import numpy as np
import pytest

from src import config
from src.features import FEATURE_COLUMNS
from src.models import get_candidates

EXPECTED = {"logreg_weighted", "rf_weighted", "rf_smote", "xgb_weighted", "xgb_smote"}


def test_candidates_names(synthetic_raw):
    assert set(get_candidates(synthetic_raw[config.TARGET_COLUMN])) == EXPECTED


def test_scale_pos_weight_from_train_labels(synthetic_raw):
    y = synthetic_raw[config.TARGET_COLUMN]
    model = get_candidates(y)["xgb_weighted"].named_steps["model"]
    assert model.scale_pos_weight == pytest.approx((y == 0).sum() / (y == 1).sum())


def test_requires_positive_labels(synthetic_raw):
    with pytest.raises(ValueError, match="no positive"):
        get_candidates(synthetic_raw[config.TARGET_COLUMN] * 0)


def test_pipelines_fit_on_raw_frame_and_ignore_leakage(synthetic_raw):
    """Full raw frame (IDs, target, leakage flags) goes in; only the 11 features reach the model."""
    y = synthetic_raw[config.TARGET_COLUMN]
    for name, pipe in get_candidates(y).items():
        pipe.fit(synthetic_raw, y)
        proba = pipe.predict_proba(synthetic_raw)[:, 1]
        assert proba.shape == (len(y),), name
        assert np.all((proba >= 0) & (proba <= 1)), name
        assert pipe.named_steps["model"].n_features_in_ == len(FEATURE_COLUMNS), name


def test_smote_keeps_type_categorical_and_features_consistent(synthetic_raw):
    """SMOTENC must not create fractional type_* values or inconsistent engineered features."""
    y = synthetic_raw[config.TARGET_COLUMN]
    pipe = get_candidates(y)["rf_smote"]
    x_raw = pipe.named_steps["select_raw"].transform(synthetic_raw)
    x_res, y_res = pipe.named_steps["smote"].fit_resample(x_raw, y)
    assert len(x_res) > len(x_raw)  # minority was oversampled
    assert set(x_res["type"]) <= set(config.VALID_TYPES)

    feats = pipe.named_steps["features"].transform(x_res)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert feats[["type_L", "type_M", "type_H"]].isin([0.0, 1.0]).all().all()
    assert (feats[["type_L", "type_M", "type_H"]].sum(axis=1) == 1).all()
    # Engineered features recomputed from synthetic raw values (not interpolated):
    expected_strain = x_res["tool_wear_min"].astype(float) * x_res["torque_nm"].astype(float)
    np.testing.assert_allclose(feats["strain"], expected_strain)
