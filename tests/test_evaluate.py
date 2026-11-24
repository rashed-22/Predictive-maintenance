"""Tests for src/evaluate.py on synthetic data (never the real test split)."""

import json
import shutil

import numpy as np
import pytest

from src.evaluate import check_targets, compute_metrics, evaluate, format_checks


def test_compute_metrics_hand_example():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    proba = np.array([0.9, 0.8, 0.6, 0.2, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1])
    m = compute_metrics(y, proba, threshold=0.5)
    # predicted positive: 0.9, 0.8, 0.6, 0.7 -> TP 3, FP 1, FN 1, TN 5
    assert m["confusion_matrix"] == {"tn": 5, "fp": 1, "fn": 1, "tp": 3}
    assert m["recall"] == pytest.approx(0.75)
    assert m["precision"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(0.75)
    assert m["f2"] == pytest.approx(0.75)
    assert m["n_positive"] == 4


def test_check_targets_pass_and_miss():
    good = {"pr_auc": 0.9, "recall": 0.9, "precision": 0.7}
    checks = check_targets(good, baseline_pr_auc=0.5)
    assert {c["status"] for c in checks.values()} == {"PASS"}

    bad = {"pr_auc": 0.7, "recall": 0.8, "precision": 0.5}
    checks = check_targets(bad, baseline_pr_auc=0.75)
    assert {c["status"] for c in checks.values()} == {"MISS"}
    assert "MISS" in format_checks({"target_checks": checks})


def test_evaluate_writes_reports_and_updates_metadata(tiny_artifacts, tmp_path):
    # Work on copies so the shared session fixture stays pristine.
    model_path = tmp_path / "model.joblib"
    metadata_path = tmp_path / "metadata.json"
    shutil.copy(tiny_artifacts["model_path"], model_path)
    shutil.copy(tiny_artifacts["metadata_path"], metadata_path)
    df = tiny_artifacts["df"]
    kwargs = {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "metrics_path": tmp_path / "metrics.json",
        "confusion_matrix_path": tmp_path / "cm.png",
        "pr_curve_path": tmp_path / "pr.png",
    }

    report = evaluate(df.iloc[:200], df.iloc[200:], **kwargs)

    assert (tmp_path / "cm.png").is_file()
    assert (tmp_path / "pr.png").is_file()
    saved = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert saved["test_metrics"]["n_samples"] == 100
    assert set(report["target_checks"]) == set(saved["target_checks"])
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert {"pr_auc", "recall", "precision", "f1"} <= set(meta["test_metrics"])
    assert meta["threshold"] == tiny_artifacts["meta"]["threshold"]  # threshold untouched

    # A second run is refused: the test set is used once.
    with pytest.raises(RuntimeError, match="already evaluated"):
        evaluate(df.iloc[:200], df.iloc[200:], **kwargs)
    evaluate(df.iloc[:200], df.iloc[200:], force=True, **kwargs)
