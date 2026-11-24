"""Training: CV comparison, tuning, threshold selection, saving artefacts (Phases 4-5).

Pipeline of ``python -m src.train``:

1. compare the five candidates with CV (Phase 4, ``reports/model_comparison.csv``);
2. tune only ``rf_weighted`` and ``xgb_weighted`` with RandomizedSearchCV on the TRAIN split
   (the SMOTE variants and logistic regression lost the Phase 4 comparison);
3. keep the one with the best CV PR-AUC;
4. choose the decision threshold from out-of-fold probabilities (docs/DATA.md "Threshold rule");
5. refit the winner on the full TRAIN split and save ``model.joblib`` + ``metadata.json``;
6. save SHAP plots (train data only).

The test split is never touched here; ``src/evaluate.py`` uses it exactly once.
"""

import argparse
import json
import logging
import platform
from datetime import UTC, datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import loguniform, uniform
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from src import config, data, explain
from src.features import FEATURE_COLUMNS
from src.models import get_candidates

logger = logging.getLogger(__name__)

# Metric key -> scikit-learn scorer. PR-AUC / ROC-AUC use probabilities; the rest use
# the default 0.5 threshold (the real threshold is chosen in train_final).
_SCORING: dict[str, object] = {
    "pr_auc": "average_precision",
    "roc_auc": "roc_auc",
    "recall": make_scorer(recall_score, zero_division=0),
    "precision": make_scorer(precision_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def format_comparison(results: pd.DataFrame) -> str:
    """Render the comparison as a readable ``mean ± std`` text table."""
    metrics = list(_SCORING)
    shown = pd.DataFrame(
        {
            m: results[f"{m}_mean"].map("{:.3f}".format)
            + " ± "
            + results[f"{m}_std"].map("{:.3f}".format)
            for m in metrics
        }
    )
    return shown.to_string()


def compare_models(
    train: pd.DataFrame | None = None,
    *,
    n_splits: int = config.CV_FOLDS,
    output_path: Path | None = config.MODEL_COMPARISON_PATH,
) -> pd.DataFrame:
    """Compare the candidate pipelines with stratified CV on the TRAIN split only.

    Args:
        train: raw training frame (with target). If None, the raw data is loaded,
            validated and split, and only the train part is used (test is never touched).
        n_splits: number of CV folds.
        output_path: where to write the CSV; None skips writing.

    Returns:
        DataFrame indexed by model name with ``<metric>_mean`` / ``<metric>_std``
        columns, sorted by mean PR-AUC (best first).
    """
    if train is None:
        raw = data.load_raw()
        data.validate(raw)
        train, _ = data.split(raw)  # the test part is deliberately discarded here

    y = train[config.TARGET_COLUMN]
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)

    rows: dict[str, dict[str, float]] = {}
    for name, pipeline in get_candidates(y).items():
        logger.info("Cross-validating %s ...", name)
        scores = cross_validate(pipeline, train, y, cv=cv, scoring=_SCORING, n_jobs=1)
        row: dict[str, float] = {}
        for metric in _SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_std"] = float(values.std())
        rows[name] = row

    results = pd.DataFrame.from_dict(rows, orient="index")
    results.index.name = "model"
    results = results.sort_values("pr_auc_mean", ascending=False)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.round(4).to_csv(output_path)
        logger.info("Saved comparison to %s", output_path)
    logger.info(
        "Model comparison (mean ± std, %d-fold CV, threshold 0.5):\n%s",
        n_splits,
        format_comparison(results),
    )
    return results


# ---------------------------------------------------------------------------- tuning
# Search spaces are deliberately small: the train split has only ~270 positive rows, so a
# big search mostly overfits the CV folds, and shallow trees keep model.joblib small.
# Keys use the pipeline step name ``model__`` as prefix.
SEARCH_SPACES: dict[str, dict[str, Any]] = {
    "rf_weighted": {
        # More trees = smoother probabilities (better PR curve) but a bigger file.
        "model__n_estimators": [100, 150, 200],
        # Max depth limits how finely the forest can memorise noise; also caps file size.
        "model__max_depth": [6, 8, 10, 12],
        # Minimum rows per leaf: larger values smooth probabilities (few positives).
        "model__min_samples_leaf": [1, 2, 3, 5],
        # Features tried per split: sqrt(11) ~ 3 (more diverse trees) vs 50 % (stronger trees).
        "model__max_features": ["sqrt", 0.5],
    },
    "xgb_weighted": {
        # Number of boosting rounds; trades off against learning_rate.
        "model__n_estimators": [100, 200, 300],
        # Tree depth = how many sensors can interact; shallow trees resist overfitting.
        "model__max_depth": [3, 4, 5, 6],
        # Step size per round; smaller values need more rounds but generalise better.
        "model__learning_rate": loguniform(0.03, 0.3),
        # Fraction of rows per tree (0.7-1.0): row sampling acts as regularisation.
        "model__subsample": uniform(0.7, 0.3),
        # Fraction of features per tree (0.6-1.0): decorrelates trees, which helps with the
        # strongly correlated sensor pairs found in EDA.
        "model__colsample_bytree": uniform(0.6, 0.4),
        # Minimum hessian weight per leaf: higher values stop tiny leaves from fitting a
        # handful of failure rows.
        "model__min_child_weight": [1, 3, 5],
    },
}


def _to_builtin(value: Any) -> Any:
    """Convert numpy scalars to plain Python so the value is JSON-serialisable."""
    return value.item() if isinstance(value, np.generic) else value


def tune_models(
    inputs: pd.DataFrame,
    y: pd.Series,
    *,
    n_iter: int = config.TUNING_ITERATIONS,
    n_splits: int = config.CV_FOLDS,
) -> dict[str, RandomizedSearchCV]:
    """Run RandomizedSearchCV (scoring = PR-AUC) for each tuned model on TRAIN data only.

    The searches use ``refit=False``: the winner is refitted explicitly afterwards.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    candidates = get_candidates(y)
    searches: dict[str, RandomizedSearchCV] = {}
    for name in config.TUNED_MODELS:
        logger.info("Tuning %s (%d candidates x %d folds) ...", name, n_iter, n_splits)
        search = RandomizedSearchCV(
            candidates[name],
            SEARCH_SPACES[name],
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            refit=False,
            random_state=config.RANDOM_STATE,
            n_jobs=1,  # the models themselves already use all cores
        )
        search.fit(inputs, y)
        logger.info(
            "%s: best CV PR-AUC %.3f with %s", name, search.best_score_, search.best_params_
        )
        searches[name] = search
    return searches


# ----------------------------------------------------------------------- threshold
def choose_threshold(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    target_recall: float = config.TARGET_RECALL,
) -> dict[str, Any]:
    """Pick the decision threshold from out-of-fold probabilities (docs/DATA.md rule).

    Among thresholds with recall >= ``target_recall`` take the one with the highest
    precision (ties: higher recall). If no threshold reaches the target, maximise F2.
    A sample is predicted positive when ``proba >= threshold``.

    Returns:
        dict with ``threshold``, ``rule`` ("recall_target" or "max_f2"), ``target_reached``
        and the OOF ``precision`` / ``recall`` at that threshold.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    # The last PR point (precision 1, recall 0) has no threshold; drop it.
    precision, recall = precision[:-1], recall[:-1]

    reachable = recall >= target_recall
    if reachable.any():
        idx = np.flatnonzero(reachable)
        order = np.lexsort((recall[idx], precision[idx]))  # last = best precision, then recall
        best = int(idx[order[-1]])
        rule = "recall_target"
    else:
        denom = 4 * precision + recall
        f2 = np.divide(5 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
        best = int(np.argmax(f2))
        rule = "max_f2"
    return {
        "threshold": float(thresholds[best]),
        "rule": rule,
        "target_reached": bool(reachable.any()),
        "precision": float(precision[best]),
        "recall": float(recall[best]),
    }


def save_threshold_plot(
    y_true: np.ndarray | pd.Series,
    proba: np.ndarray,
    choice: dict[str, Any],
    path: Path = config.THRESHOLD_PLOT_PATH,
) -> None:
    """Save precision and recall as functions of the threshold, marking the chosen one."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    fig = Figure(figsize=(7, 4.5))
    ax = fig.subplots()
    ax.plot(thresholds, precision[:-1], label="Precision", color="#1f77b4")
    ax.plot(thresholds, recall[:-1], label="Recall", color="#d62728")
    ax.axhline(config.TARGET_RECALL, color="grey", linestyle=":", label="Recall target 0.85")
    ax.axvline(
        choice["threshold"],
        color="black",
        linestyle="--",
        label=f"Chosen threshold {choice['threshold']:.3f}",
    )
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (out-of-fold, train split)")
    ax.set_title("Precision and recall vs threshold")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info("Saved %s", path.name)


# --------------------------------------------------------------------- final training
def _library_versions() -> dict[str, str]:
    """Versions of the libraries that affect the pickled pipeline."""
    names = ["numpy", "pandas", "scikit-learn", "imbalanced-learn", "xgboost", "joblib", "shap"]
    versions = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def train_final(
    train: pd.DataFrame,
    *,
    n_iter: int = config.TUNING_ITERATIONS,
    n_splits: int = config.CV_FOLDS,
    model_path: Path = config.MODEL_PATH,
    metadata_path: Path = config.METADATA_PATH,
    threshold_plot_path: Path | None = config.THRESHOLD_PLOT_PATH,
    max_size_mb: float = config.MAX_MODEL_SIZE_MB,
) -> dict[str, Any]:
    """Tune, pick the best model, choose the threshold, refit on TRAIN and save artefacts.

    Note: hyperparameter tuning and threshold selection use the SAME training folds
    (same ``StratifiedKFold`` seed), so the CV / out-of-fold numbers are slightly
    optimistic. The untouched test split gives the unbiased estimate.

    Returns:
        The metadata dict that was written to ``metadata_path``.
    """
    y = train[config.TARGET_COLUMN]
    inputs = train[config.RAW_FEATURE_COLUMNS]  # leakage / ID columns never reach the pipeline

    # Tune both models; pick the winner by CV PR-AUC.
    searches = tune_models(inputs, y, n_iter=n_iter, n_splits=n_splits)
    best_name = max(searches, key=lambda n: searches[n].best_score_)
    search = searches[best_name]
    logger.info("Best model by CV PR-AUC: %s (%.3f)", best_name, search.best_score_)

    # Out-of-fold probabilities for the winner -> decision threshold.
    best_pipeline = clone(get_candidates(y)[best_name]).set_params(**search.best_params_)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.RANDOM_STATE)
    oof = cross_val_predict(best_pipeline, inputs, y, cv=cv, method="predict_proba", n_jobs=1)[:, 1]
    choice = choose_threshold(y, oof)
    threshold = choice["threshold"]
    oof_pred = (oof >= threshold).astype(int)
    if choice["target_reached"]:
        logger.info(
            "Recall >= %.2f reachable: threshold %.3f -> OOF recall %.3f, precision %.3f",
            config.TARGET_RECALL,
            threshold,
            choice["recall"],
            choice["precision"],
        )
    else:
        logger.warning(
            "Recall >= %.2f NOT reachable; maximised F2 instead: threshold %.3f -> "
            "OOF recall %.3f, precision %.3f",
            config.TARGET_RECALL,
            threshold,
            choice["recall"],
            choice["precision"],
        )
    if threshold_plot_path is not None:
        save_threshold_plot(y, oof, choice, threshold_plot_path)

    cv_metrics = {
        "cv_pr_auc_mean": float(search.best_score_),
        "cv_pr_auc_std": float(search.cv_results_["std_test_score"][search.best_index_]),
        "oof_pr_auc": float(average_precision_score(y, oof)),
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_recall": float(recall_score(y, oof_pred, zero_division=0)),
        "oof_precision": float(precision_score(y, oof_pred, zero_division=0)),
        "oof_f1": float(f1_score(y, oof_pred, zero_division=0)),
        "oof_f2": float(fbeta_score(y, oof_pred, beta=2, zero_division=0)),
    }

    # Refit the winner on the full TRAIN split and save.
    final_model = clone(best_pipeline).fit(inputs, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_path, compress=3)
    size_mb = model_path.stat().st_size / 1024**2
    logger.info("Saved %s (%.2f MB)", model_path.name, size_mb)
    if size_mb > max_size_mb:
        raise ValueError(f"{model_path.name} is {size_mb:.1f} MB, above the {max_size_mb} MB limit")

    meta: dict[str, Any] = {
        "model_name": best_name,
        "version": config.MODEL_VERSION,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "threshold": threshold,
        "threshold_rule": {
            "rule": choice["rule"],
            "target_recall": config.TARGET_RECALL,
            "target_reached": choice["target_reached"],
        },
        "raw_features": config.RAW_FEATURE_COLUMNS,
        "features": FEATURE_COLUMNS,
        "cv_metrics": cv_metrics,
        "candidates_cv_pr_auc": {n: float(s.best_score_) for n, s in searches.items()},
        "best_params": {k: _to_builtin(v) for k, v in search.best_params_.items()},
        "model_size_mb": round(size_mb, 3),
        "train_rows": int(len(train)),
        "train_positive_rate": float(y.mean()),
        "random_state": config.RANDOM_STATE,
        "libraries": _library_versions(),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("Saved %s", metadata_path.name)
    return meta


def main() -> None:
    """CLI entry point: comparison -> tuning -> threshold -> refit -> save -> SHAP."""
    parser = argparse.ArgumentParser(description="Train the failure-prediction model.")
    parser.add_argument(
        "--skip-comparison", action="store_true", help="skip re-running the Phase 4 comparison"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    raw = data.load_raw()
    data.validate(raw)
    train, _ = data.split(raw)  # the test part is deliberately discarded here

    if not args.skip_comparison:
        compare_models(train)
    meta = train_final(train)
    explain.save_shap_plots(config.MODEL_PATH, train)

    cv = meta["cv_metrics"]
    logger.info(
        "DONE %s | CV PR-AUC %.3f | threshold %.3f (%s) | OOF recall %.3f, precision %.3f",
        meta["model_name"],
        cv["cv_pr_auc_mean"],
        meta["threshold"],
        meta["threshold_rule"]["rule"],
        cv["oof_recall"],
        cv["oof_precision"],
    )


if __name__ == "__main__":
    main()
