"""One-time test-set evaluation (Phase 5).

``python -m src.evaluate`` scores the saved model on the held-out TEST split at the stored
threshold, writes ``reports/metrics.json`` and two plots, adds ``test_metrics`` to
``models/metadata.json`` and prints PASS/MISS against the PRD success metrics.

The test set must be used once: nothing may be tuned from these numbers. As a guard, the
script refuses to run again while ``metadata.json`` already contains ``test_metrics``
(retraining writes fresh metadata, or pass ``--force`` knowingly).
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config, data
from src.models import get_candidates

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray | pd.Series, proba: np.ndarray, threshold: float
) -> dict[str, Any]:
    """Threshold-free (PR-AUC, ROC-AUC) and thresholded metrics plus the confusion matrix."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, pred, beta=2, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true)),
    }


def check_targets(metrics: dict[str, Any], baseline_pr_auc: float) -> dict[str, dict[str, Any]]:
    """Compare test metrics with the PRD section 7 success metrics (PASS / MISS)."""

    def row(value: float, target: float, ok: bool) -> dict[str, Any]:
        return {"value": value, "target": target, "status": "PASS" if ok else "MISS"}

    return {
        "pr_auc >= 0.80": row(
            metrics["pr_auc"], config.TARGET_PR_AUC, metrics["pr_auc"] >= config.TARGET_PR_AUC
        ),
        "recall >= 0.85": row(
            metrics["recall"], config.TARGET_RECALL, metrics["recall"] >= config.TARGET_RECALL
        ),
        "precision >= 0.60": row(
            metrics["precision"],
            config.TARGET_PRECISION,
            metrics["precision"] >= config.TARGET_PRECISION,
        ),
        "beats logreg baseline (pr_auc)": row(
            metrics["pr_auc"], baseline_pr_auc, metrics["pr_auc"] > baseline_pr_auc
        ),
    }


def save_confusion_matrix_plot(
    cm: dict[str, int], path: Path = config.CONFUSION_MATRIX_PATH
) -> None:
    """Save the 2x2 confusion matrix as an annotated heatmap."""
    matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
    fig = Figure(figsize=(5, 4.5))
    ax = fig.subplots()
    ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center", fontsize=16,
                color="white" if value > matrix.max() / 2 else "black")  # fmt: skip
    ax.set_xticks([0, 1], ["No failure", "Failure"])
    ax.set_yticks([0, 1], ["No failure", "Failure"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion matrix (test set)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def save_pr_curve_plot(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    metrics: dict[str, Any],
    path: Path = config.PR_CURVE_PATH,
) -> None:
    """Save the test PR curve with the operating point at the stored threshold."""
    precision, recall, _ = precision_recall_curve(y_true, proba)
    fig = Figure(figsize=(6, 4.5))
    ax = fig.subplots()
    ax.plot(recall, precision, color="#1f77b4", label=f"PR curve (AP = {metrics['pr_auc']:.3f})")
    ax.scatter(
        [metrics["recall"]],
        [metrics["precision"]],
        color="#d62728",
        zorder=3,
        label=f"Threshold {metrics['threshold']:.3f}",
    )
    ax.axhline(float(np.mean(y_true)), color="grey", linestyle=":", label="Chance (prevalence)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curve (test set)")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def evaluate(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
    metrics_path: Path = config.METRICS_PATH,
    confusion_matrix_path: Path | None = config.CONFUSION_MATRIX_PATH,
    pr_curve_path: Path | None = config.PR_CURVE_PATH,
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate the saved model on ``test`` at the stored threshold (call this once).

    ``train`` is only used to fit the untuned logistic-regression baseline that the PRD
    asks us to beat; it plays no part in choosing anything.

    Returns:
        The dict written to ``metrics_path`` (metrics, target checks, baseline).
    """
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if "test_metrics" in metadata and not force:
        raise RuntimeError(
            "The test set was already evaluated for this model. Re-evaluating invites "
            "tuning on the test set; retrain first or pass --force."
        )

    pipeline = joblib.load(model_path)
    y_test = test[config.TARGET_COLUMN]
    proba = pipeline.predict_proba(test[config.RAW_FEATURE_COLUMNS])[:, 1]
    metrics = compute_metrics(y_test, proba, float(metadata["threshold"]))

    baseline = get_candidates(train[config.TARGET_COLUMN])["logreg_weighted"]
    baseline.fit(train[config.RAW_FEATURE_COLUMNS], train[config.TARGET_COLUMN])
    baseline_pr_auc = float(
        average_precision_score(
            y_test, baseline.predict_proba(test[config.RAW_FEATURE_COLUMNS])[:, 1]
        )
    )
    checks = check_targets(metrics, baseline_pr_auc)

    report = {
        "model_name": metadata["model_name"],
        "model_version": metadata["version"],
        "test_metrics": metrics,
        "baseline_logreg_pr_auc": baseline_pr_auc,
        "target_checks": checks,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if confusion_matrix_path is not None:
        save_confusion_matrix_plot(metrics["confusion_matrix"], confusion_matrix_path)
    if pr_curve_path is not None:
        save_pr_curve_plot(y_test, proba, metrics, pr_curve_path)

    scalar_keys = ["pr_auc", "roc_auc", "recall", "precision", "f1", "f2"]
    metadata["test_metrics"] = {k: metrics[k] for k in scalar_keys}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return report


def format_checks(report: dict[str, Any]) -> str:
    """Readable PASS/MISS table for the console."""
    lines = [f"{'Success metric (PRD)':<34}{'value':>8}{'target':>8}  result"]
    for name, c in report["target_checks"].items():
        lines.append(f"{name:<34}{c['value']:>8.3f}{c['target']:>8.2f}  {c['status']}")
    return "\n".join(lines)


def main() -> None:
    """CLI entry point: evaluate once on the test split."""
    parser = argparse.ArgumentParser(description="Evaluate the saved model on the test split.")
    parser.add_argument("--force", action="store_true", help="re-evaluate even if already done")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raw = data.load_raw()
    data.validate(raw)
    train, test = data.split(raw)
    report = evaluate(train, test, force=args.force)

    m = report["test_metrics"]
    cm = m["confusion_matrix"]
    logger.info(
        "Test metrics @ threshold %.3f: PR-AUC %.3f | ROC-AUC %.3f | recall %.3f | "
        "precision %.3f | F1 %.3f | F2 %.3f",
        m["threshold"], m["pr_auc"], m["roc_auc"], m["recall"], m["precision"], m["f1"], m["f2"],
    )  # fmt: skip
    logger.info("Confusion matrix: TN %d FP %d FN %d TP %d", cm["tn"], cm["fp"], cm["fn"], cm["tp"])
    logger.info("Logistic-regression baseline test PR-AUC: %.3f", report["baseline_logreg_pr_auc"])
    logger.info("\n%s", format_checks(report))


if __name__ == "__main__":
    main()
