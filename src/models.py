"""Candidate model pipelines: logreg, random forest, XGBoost (Phase 4).

Every pipeline takes the RAW columns (type + 5 sensors; extra columns are ignored) and
returns probabilities, so preprocessing can never drift from the model.

SMOTE variants use ``SMOTENC`` on the raw columns, *before* ``build_features``:
``type`` stays a real category (no fractional one-hot values) and the engineered
features are recomputed from the synthetic raw values, so they stay physically
consistent (e.g. power = torque x speed). See docs/ARCHITECTURE.md D8.
"""

import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.features import build_features


def _select_raw(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the raw model inputs (drops IDs, target and leakage columns)."""
    return df[config.RAW_FEATURE_COLUMNS]


def _features() -> FunctionTransformer:
    """Pipeline step wrapping the single source of truth for features."""
    return FunctionTransformer(build_features)


def _smote_steps() -> list[tuple[str, object]]:
    """Raw-column selection + SMOTENC (type marked categorical), then feature building."""
    return [
        ("select_raw", FunctionTransformer(_select_raw)),
        (
            "smote",
            SMOTENC(
                categorical_features=config.CATEGORICAL_COLUMNS, random_state=config.RANDOM_STATE
            ),
        ),
        ("features", _features()),
    ]


def _random_forest(**kwargs: object) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=config.RANDOM_STATE, **kwargs
    )


def _xgboost(**kwargs: object) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=200,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=config.RANDOM_STATE,
        **kwargs,
    )


def get_candidates(y_train: pd.Series) -> dict[str, Pipeline]:
    """Return the untuned candidate pipelines, keyed by name.

    ``y_train`` (TRAIN labels only) is used for XGBoost's ``scale_pos_weight``
    (= n_negative / n_positive). Inside CV it is computed on the whole train split
    rather than per fold; with stratified folds the ratio is practically identical.
    """
    n_pos = int((y_train == 1).sum())
    if n_pos == 0:
        raise ValueError("y_train contains no positive samples")
    scale_pos_weight = (len(y_train) - n_pos) / n_pos

    return {
        "logreg_weighted": Pipeline(
            [
                ("features", _features()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE
                    ),
                ),
            ]
        ),
        "rf_weighted": Pipeline(
            [("features", _features()), ("model", _random_forest(class_weight="balanced"))]
        ),
        "rf_smote": Pipeline([*_smote_steps(), ("model", _random_forest())]),
        "xgb_weighted": Pipeline(
            [
                ("features", _features()),
                ("model", _xgboost(scale_pos_weight=scale_pos_weight)),
            ]
        ),
        "xgb_smote": Pipeline([*_smote_steps(), ("model", _xgboost())]),
    }
