"""Training: CV comparison, tuning, threshold selection, saving artefacts (Phases 4-5)."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config, data
from src.models import get_candidates

logger = logging.getLogger(__name__)

# Metric key -> scikit-learn scorer. PR-AUC / ROC-AUC use probabilities; the rest use
# the default 0.5 threshold (threshold tuning is Phase 5).
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


def main() -> None:
    """CLI entry point: run the model comparison."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    compare_models()


if __name__ == "__main__":
    main()
