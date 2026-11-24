"""SHAP explainability for the saved tree model (Phase 5).

Uses TRAIN data only, so it can run without touching the test split.
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
from imblearn.pipeline import Pipeline

from src import config

logger = logging.getLogger(__name__)


def compute_shap_values(
    pipeline: Pipeline, raw_inputs: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    """Return SHAP values for the positive (failure) class and the feature frame used.

    The pipeline is split in two: every step but the last turns raw columns into the
    model features (``build_features``); ``TreeExplainer`` then explains the last step.
    Units: random forest -> probability points; XGBoost -> log-odds.

    Returns:
        ``(shap_values, features)`` with ``shap_values.shape == features.shape``.
    """
    features = pipeline[:-1].transform(raw_inputs)
    explainer = shap.TreeExplainer(pipeline[-1])
    values = explainer.shap_values(features)
    # Older/newer shap versions return a list per class or an (n, f, classes) array
    # for random forests; XGBoost returns a plain (n, f) array.
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]
    return np.asarray(values), features


def save_shap_plots(
    model_path: Path,
    train: pd.DataFrame,
    summary_path: Path = config.SHAP_SUMMARY_PATH,
    bar_path: Path = config.SHAP_BAR_PATH,
    sample_size: int = config.SHAP_SAMPLE_SIZE,
) -> pd.Series:
    """Save a beeswarm summary plot and a mean-|SHAP| bar plot; return the importance ranking.

    Args:
        model_path: saved pipeline (``models/model.joblib``).
        train: raw TRAIN frame; a random sample of ``sample_size`` rows is explained.
    """
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")  # no display needed
    pipeline = joblib.load(model_path)
    inputs = train[config.RAW_FEATURE_COLUMNS]
    if len(inputs) > sample_size:
        inputs = inputs.sample(sample_size, random_state=config.RANDOM_STATE)
    values, features = compute_shap_values(pipeline, inputs)

    for plot_type, path, title in (
        ("dot", summary_path, "SHAP summary (failure class, train sample)"),
        ("bar", bar_path, "Mean |SHAP value| per feature (failure class)"),
    ):
        plt.figure(figsize=(8, 5.5))
        shap.summary_plot(values, features, plot_type=plot_type, show=False)
        plt.title(title)
        path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved %s", path.name)

    importance = pd.Series(np.abs(values).mean(axis=0), index=features.columns)
    importance = importance.sort_values(ascending=False)
    logger.info("Mean |SHAP| ranking:\n%s", importance.round(4).to_string())
    return importance
