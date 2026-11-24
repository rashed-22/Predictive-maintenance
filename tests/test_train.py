"""Tests for src/train.py (fast smoke tests on tiny synthetic data)."""

import pandas as pd

from src.train import compare_models

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
