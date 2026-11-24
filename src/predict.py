"""Model loading and inference helpers used by the API and dashboard (Phase 5)."""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

LABEL_FAILURE = "failure"
LABEL_NO_FAILURE = "no_failure"


@dataclass(frozen=True)
class LoadedModel:
    """A fitted pipeline together with its metadata (threshold, version, ...)."""

    pipeline: Any
    metadata: dict[str, Any]

    @property
    def threshold(self) -> float:
        """Decision threshold chosen in training."""
        return float(self.metadata["threshold"])


@lru_cache(maxsize=4)
def load_model(
    model_path: Path = config.MODEL_PATH, metadata_path: Path = config.METADATA_PATH
) -> LoadedModel:
    """Load the pipeline and metadata once; later calls with the same paths reuse them.

    Raises:
        FileNotFoundError: if either file is missing (run ``python -m src.train``).
    """
    for path in (model_path, metadata_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{path} not found - run `python -m src.train` first")
    pipeline = joblib.load(model_path)
    metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    logger.info("Loaded model %s v%s", metadata.get("model_name"), metadata.get("version"))
    return LoadedModel(pipeline=pipeline, metadata=metadata)


def predict_df(df: pd.DataFrame, model: LoadedModel | None = None) -> pd.DataFrame:
    """Predict failure probability, class and label for raw sensor rows.

    Args:
        df: raw columns (``type`` + 5 sensors); extra columns are ignored.
        model: loaded model; defaults to the cached ``load_model()``.

    Returns:
        DataFrame (same index as ``df``) with ``failure_probability``, ``prediction``
        (1 if probability >= stored threshold) and ``label``.
    """
    model = model or load_model()
    missing = [c for c in config.RAW_FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    proba = model.pipeline.predict_proba(df[config.RAW_FEATURE_COLUMNS])[:, 1]
    prediction = (proba >= model.threshold).astype(int)
    return pd.DataFrame(
        {
            "failure_probability": proba,
            "prediction": prediction,
            "label": pd.Series(prediction, index=df.index).map(
                {1: LABEL_FAILURE, 0: LABEL_NO_FAILURE}
            ),
        },
        index=df.index,
    )
