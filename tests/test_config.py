"""Tests for src/config.py: paths resolve and leakage columns stay out of the features."""

from src import config


def test_project_root_contains_core_folders() -> None:
    assert config.PROJECT_ROOT.is_dir()
    for folder in (config.DATA_DIR, config.MODELS_DIR, config.REPORTS_DIR):
        assert folder.is_dir(), f"{folder} should exist"
    assert (config.PROJECT_ROOT / "pyproject.toml").is_file()


def test_paths_are_inside_project_root() -> None:
    for path in (config.RAW_DATA_PATH, config.MODEL_PATH, config.METADATA_PATH):
        assert config.PROJECT_ROOT in path.parents


def test_leakage_columns_not_in_features() -> None:
    forbidden = set(config.LEAKAGE_COLUMNS) | set(config.ID_COLUMNS) | {config.TARGET_COLUMN}
    assert forbidden.isdisjoint(config.RAW_FEATURE_COLUMNS)
    assert forbidden.isdisjoint(config.MODEL_INPUT_COLUMNS)


def test_leakage_list_is_complete() -> None:
    assert set(config.LEAKAGE_COLUMNS) == {"twf", "hdf", "pwf", "osf", "rnf"}


def test_column_mapping_covers_required_columns() -> None:
    assert set(config.REQUIRED_COLUMNS) == set(config.COLUMN_MAPPING.values())


def test_ranges_defined_for_every_sensor_and_valid() -> None:
    assert set(config.VALUE_RANGES) == set(config.NUMERIC_SENSOR_COLUMNS)
    for low, high in config.VALUE_RANGES.values():
        assert low < high


def test_random_state() -> None:
    assert config.RANDOM_STATE == 42
