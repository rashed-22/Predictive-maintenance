"""Data loading, schema validation and train/test split."""

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config

logger = logging.getLogger(__name__)

# Columns that must be numeric (the sensors, id, target and the failure-mode flags).
_NUMERIC_COLUMNS: list[str] = (
    ["udi"] + config.NUMERIC_SENSOR_COLUMNS + [config.TARGET_COLUMN] + config.LEAKAGE_COLUMNS
)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Read the raw AI4I CSV and rename its columns to snake_case."""
    csv_path = Path(path) if path is not None else config.RAW_DATA_PATH
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Raw data file not found: {csv_path}. "
            "Download ai4i2020.csv from UCI (dataset 601) into data/raw/."
        )
    df = pd.read_csv(csv_path).rename(columns=config.COLUMN_MAPPING)
    logger.info("Loaded %d rows x %d columns from %s", *df.shape, csv_path)
    return df


def validate(df: pd.DataFrame) -> None:
    """Check the dataframe against the schema in docs/DATA.md.

    Runs every check and raises ONE ValueError listing all failures, so a bad file
    can be fixed in a single pass. Returns None when the data is valid.
    """
    missing_cols = [c for c in config.REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        # Later checks need these columns, so stop here.
        raise ValueError(f"Missing required columns: {missing_cols}")

    problems: list[str] = []

    # Dtypes
    for col in _NUMERIC_COLUMNS:
        if not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"Column '{col}' must be numeric, got dtype {df[col].dtype}")

    # Missing values
    null_counts = df[config.REQUIRED_COLUMNS].isna().sum()
    null_counts = null_counts[null_counts > 0]
    if not null_counts.empty:
        problems.append(f"Missing values found: {null_counts.to_dict()}")

    # Duplicate ids
    n_dup = int(df["udi"].duplicated().sum())
    if n_dup:
        problems.append(f"Found {n_dup} duplicate 'udi' values")

    # Categorical values
    bad_types = sorted(set(df["type"].dropna().unique()) - set(config.VALID_TYPES))
    if bad_types:
        problems.append(f"Column 'type' has invalid values {bad_types}; allowed: L, M, H")

    # Target and flags must be binary
    for col in [config.TARGET_COLUMN, *config.LEAKAGE_COLUMNS]:
        if pd.api.types.is_numeric_dtype(df[col]):
            bad = sorted(set(df[col].dropna().unique()) - {0, 1})
            if bad:
                problems.append(f"Column '{col}' must be 0/1, found values {bad}")

    # Physical ranges
    for col, (low, high) in config.VALUE_RANGES.items():
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue  # already reported above
        n_out = int(((df[col] < low) | (df[col] > high)).sum())
        if n_out:
            problems.append(
                f"Column '{col}': {n_out} values outside [{low}, {high}] "
                f"(observed min={df[col].min()}, max={df[col].max()})"
            )

    if problems:
        raise ValueError("Data validation failed:\n- " + "\n- ".join(problems))
    logger.info("Validation passed (%d rows)", len(df))


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the target (80/20, fixed seed).

    Original row indices are kept so the test set can be identified by index.
    """
    train, test = train_test_split(
        df,
        test_size=config.TEST_SIZE,
        stratify=df[config.TARGET_COLUMN],
        random_state=config.RANDOM_STATE,
    )
    return train, test


def _log_summary(df: pd.DataFrame) -> None:
    """Log shape, class balance and basic sensor statistics."""
    logger.info("Shape: %d rows x %d columns", *df.shape)
    counts = df[config.TARGET_COLUMN].value_counts().sort_index()
    for label, n in counts.items():
        logger.info("Class %s: %d rows (%.2f%%)", label, n, 100 * n / len(df))
    logger.info("Type distribution: %s", df["type"].value_counts(normalize=True).round(3).to_dict())
    stats = df[config.NUMERIC_SENSOR_COLUMNS].describe().round(2)
    logger.info("Sensor statistics:\n%s", stats.to_string())


def main() -> None:
    """CLI entry point: load, validate, split and summarise the raw data."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    df = load_raw()
    validate(df)
    _log_summary(df)
    train, test = split(df)
    for name, part in (("train", train), ("test", test)):
        rate = 100 * part[config.TARGET_COLUMN].mean()
        logger.info("%s: %d rows, failure rate %.2f%%", name, len(part), rate)


if __name__ == "__main__":
    main()
