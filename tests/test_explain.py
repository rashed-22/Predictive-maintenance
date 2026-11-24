"""Tests for src/explain.py on tiny synthetic models."""

import pytest

from src import config
from src.explain import compute_shap_values, save_shap_plots
from src.features import FEATURE_COLUMNS
from src.models import get_candidates


@pytest.mark.parametrize("name", ["rf_weighted", "xgb_weighted"])
def test_shap_values_have_feature_shape_for_both_tree_models(synthetic_raw, name):
    y = synthetic_raw[config.TARGET_COLUMN]
    inputs = synthetic_raw[config.RAW_FEATURE_COLUMNS]
    pipeline = get_candidates(y)[name].set_params(model__n_estimators=20).fit(inputs, y)

    values, features = compute_shap_values(pipeline, inputs.iloc[:50])

    assert values.shape == (50, len(FEATURE_COLUMNS))
    assert list(features.columns) == FEATURE_COLUMNS


def test_save_shap_plots_writes_both_figures(tiny_artifacts, tmp_path):
    importance = save_shap_plots(
        tiny_artifacts["model_path"],
        tiny_artifacts["df"],
        summary_path=tmp_path / "summary.png",
        bar_path=tmp_path / "bar.png",
        sample_size=100,
    )
    assert (tmp_path / "summary.png").stat().st_size > 0
    assert (tmp_path / "bar.png").stat().st_size > 0
    assert set(importance.index) == set(FEATURE_COLUMNS)
    assert importance.is_monotonic_decreasing
