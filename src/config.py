"""Central configuration: paths, seed, column names and physical value ranges."""

from pathlib import Path

# --------------------------------------------------------------------------- paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
RAW_DATA_PATH: Path = RAW_DATA_DIR / "ai4i2020.csv"
SAMPLE_BATCH_PATH: Path = SAMPLE_DATA_DIR / "sample_batch.csv"

MODELS_DIR: Path = PROJECT_ROOT / "models"
MODEL_PATH: Path = MODELS_DIR / "model.joblib"
METADATA_PATH: Path = MODELS_DIR / "metadata.json"

REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"
METRICS_PATH: Path = REPORTS_DIR / "metrics.json"
MODEL_COMPARISON_PATH: Path = REPORTS_DIR / "model_comparison.csv"

# --------------------------------------------------------------------- reproducibility
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5

# ------------------------------------------------------------------------ raw columns
# Raw CSV header -> snake_case name used everywhere in the code.
COLUMN_MAPPING: dict[str, str] = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

# ------------------------------------------------------------------ column roles (snake_case)
TARGET_COLUMN: str = "machine_failure"

# Identifier columns: carry no physical signal (product_id also encodes type).
ID_COLUMNS: list[str] = ["udi", "product_id"]

# Failure-mode flags are outcomes of the same event as the target -> LEAKAGE.
# Allowed in EDA only, never as model input.
LEAKAGE_COLUMNS: list[str] = ["twf", "hdf", "pwf", "osf", "rnf"]

CATEGORICAL_COLUMNS: list[str] = ["type"]
NUMERIC_SENSOR_COLUMNS: list[str] = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]

# Raw inputs the model/API accepts (before feature engineering).
RAW_FEATURE_COLUMNS: list[str] = CATEGORICAL_COLUMNS + NUMERIC_SENSOR_COLUMNS

# Features created in src/features.py (formulas live ONLY there).
ENGINEERED_COLUMNS: list[str] = ["temp_diff_k", "power_w", "strain"]

# Columns that must exist in the validated raw dataframe.
REQUIRED_COLUMNS: list[str] = ID_COLUMNS + RAW_FEATURE_COLUMNS + [TARGET_COLUMN] + LEAKAGE_COLUMNS

# Every column a model may use = raw features + engineered. The final ordered
# FEATURE_COLUMNS list (after encoding) is defined in src/features.py in Phase 3.
MODEL_INPUT_COLUMNS: list[str] = RAW_FEATURE_COLUMNS + ENGINEERED_COLUMNS

VALID_TYPES: tuple[str, ...] = ("L", "M", "H")

# -------------------------------------------------------------------- physical ranges
# TO CONFIRM IN EDA (Phase 2). Inclusive (min, max). Initial values come from
# docs/ARCHITECTURE.md section 4 with a safety margin; observed values in the raw data
# (checked at scaffold time): air 295.3-304.5, process 305.7-313.8, rpm 1168-2886,
# torque 3.8-76.6, tool wear 0-253.
VALUE_RANGES: dict[str, tuple[float, float]] = {
    "air_temperature_k": (290.0, 310.0),
    "process_temperature_k": (300.0, 320.0),
    "rotational_speed_rpm": (1000.0, 3000.0),
    "torque_nm": (0.0, 100.0),
    "tool_wear_min": (0.0, 300.0),
}
