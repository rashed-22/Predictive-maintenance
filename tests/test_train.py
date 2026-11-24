"""Tests for src/train.py (fast smoke tests on tiny synthetic data)."""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import recall_score

from src import config
from src.features import FEATURE_COLUMNS
from src.models import get_candidates
from src.train import SEARCH_SPACES, choose_threshold, compare_models, train_final

METRICS = ["pr_auc", "roc_auc", "recall", "precision", "f1"]


def test_compare_models_smoke(synthetic_raw, tmp_path):
    out = tmp_path / "model_comparison.csv"
    results = compare_models(synthetic_raw, n_splits=2, output_path=out)

    assert len(results) == 5
    for m in METRICS:
        assert results[f"{m}_mean"].between(0, 1).all()
        assert (results[f"{m}_std"] >= 0).all()
    assert results["pr_auc_mean"].is_monotonic_decreasing  # sorted best first

    saved = pd.read_csv(out, index_col="model")
    assert set(saved.index) == set(results.index)
    assert set(saved.columns) == {f"{m}_{s}" for m in METRICS for s in ("mean", "std")}


# ------------------------------------------------------------------ choose_threshold
def test_choose_threshold_picks_highest_precision_with_recall_target():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1, 0.1, 0.05])
    choice = choose_threshold(y, proba, target_recall=0.75)
    # Threshold 0.7 keeps the three top positives (recall 3/4) and no negative (precision 1);
    # lower thresholds only add the 0.6 negative, so precision drops.
    assert choice["rule"] == "recall_target"
    assert choice["target_reached"] is True
    assert choice["threshold"] == 0.7
    assert choice["recall"] == 0.75
    assert choice["precision"] == 1.0


def test_choose_threshold_falls_back_to_f2_when_target_unreachable():
    y = np.array([0] * 9 + [1])
    proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.35])
    choice = choose_threshold(y, proba, target_recall=1.1)  # recall can never exceed 1
    assert choice["rule"] == "max_f2"
    assert choice["target_reached"] is False
    assert 0 < choice["recall"] <= 1


def test_choose_threshold_prediction_rule_is_greater_or_equal():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.4, 0.9])
    choice = choose_threshold(y, proba, target_recall=1.0)
    pred = (proba >= choice["threshold"]).astype(int)
    assert recall_score(y, pred) == choice["recall"]


# ------------------------------------------------------------------------ train_final
def test_search_spaces_cover_only_tuned_models_and_valid_params():
    assert set(SEARCH_SPACES) == set(config.TUNED_MODELS) == {"rf_weighted", "xgb_weighted"}
    pipelines = get_candidates(pd.Series([0, 1]))
    for name, space in SEARCH_SPACES.items():
        assert set(space) <= set(pipelines[name].get_params())


def test_train_final_writes_model_and_metadata(tiny_artifacts):
    meta = tiny_artifacts["meta"]
    assert tiny_artifacts["model_path"].is_file()
    assert (tiny_artifacts["dir"] / "threshold.png").is_file()

    saved = json.loads(tiny_artifacts["metadata_path"].read_text(encoding="utf-8"))
    assert saved == meta
    assert saved["model_name"] in config.TUNED_MODELS
    assert saved["version"] == "1.0.0"
    assert saved["trained_at"].endswith("Z")
    assert 0 < saved["threshold"] < 1
    assert saved["features"] == FEATURE_COLUMNS
    assert saved["best_params"]
    assert {"cv_pr_auc_mean", "oof_recall", "oof_precision", "oof_f2"} <= set(saved["cv_metrics"])
    assert {"python", "scikit-learn", "xgboost"} <= set(saved["libraries"])


def test_metadata_has_no_absolute_local_paths(tiny_artifacts):
    text = tiny_artifacts["metadata_path"].read_text(encoding="utf-8")
    assert str(config.PROJECT_ROOT) not in text
    assert str(tiny_artifacts["dir"]) not in text


def test_saved_model_accepts_raw_columns_only(tiny_artifacts):
    model = joblib.load(tiny_artifacts["model_path"])
    df = tiny_artifacts["df"][config.RAW_FEATURE_COLUMNS]  # no IDs, no leakage columns
    proba = model.predict_proba(df)[:, 1]
    assert proba.shape == (len(df),)


def test_train_final_rejects_oversized_model(synthetic_raw, tmp_path):
    with pytest.raises(ValueError, match="above"):
        train_final(
            synthetic_raw,
            n_iter=1,
            n_splits=2,
            model_path=tmp_path / "m.joblib",
            metadata_path=tmp_path / "m.json",
            threshold_plot_path=None,
            max_size_mb=0.0001,
        )
