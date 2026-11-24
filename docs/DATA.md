# Data Card — AI4I 2020 Predictive Maintenance Dataset

## Source
- UCI Machine Learning Repository, dataset id **601** ("AI4I 2020 Predictive Maintenance Dataset").
- Synthetic dataset that reflects real industrial milling-machine data. 10,000 rows, 14 columns. License: CC BY 4.0.
- Get it by downloading `ai4i2020.csv` from the UCI page into `data/raw/`
  (or `pip install ucimlrepo` → `fetch_ucirepo(id=601)`; Kaggle also mirrors it).

## Columns
| Raw name | snake_case name | Type | Role |
|---|---|---|---|
| UDI | `udi` | int | ID → **drop** |
| Product ID | `product_id` | str | ID (encodes Type) → **drop** |
| Type | `type` | cat L/M/H | feature (L ≈ 60 %, M ≈ 30 %, H ≈ 10 %) |
| Air temperature [K] | `air_temperature_k` | float | feature |
| Process temperature [K] | `process_temperature_k` | float | feature |
| Rotational speed [rpm] | `rotational_speed_rpm` | int | feature |
| Torque [Nm] | `torque_nm` | float | feature |
| Tool wear [min] | `tool_wear_min` | int | feature |
| Machine failure | `machine_failure` | 0/1 | **target** (~3.4 % positive) |
| TWF | `twf` | 0/1 | tool wear failure → **LEAKAGE, drop** |
| HDF | `hdf` | 0/1 | heat dissipation failure → **LEAKAGE, drop** |
| PWF | `pwf` | 0/1 | power failure → **LEAKAGE, drop** |
| OSF | `osf` | 0/1 | overstrain failure → **LEAKAGE, drop** |
| RNF | `rnf` | 0/1 | random failure → **LEAKAGE, drop** |

The failure-mode flags can be used **only in EDA** to understand the failure types, never as model input.
Note: a few rows have `machine_failure = 1` with no mode flag set, or a mode flag (often RNF) with
`machine_failure = 0`. Document how many rows this affects in EDA; keep the target as given.

## Engineered features (implemented only in `src/features.py`)
| Feature | Formula | Physical motivation |
|---|---|---|
| `temp_diff_k` | process_temperature_k − air_temperature_k | Heat dissipation failure happens when this is small and speed is low |
| `power_w` | torque_nm × rotational_speed_rpm × 2π / 60 | Power failure happens when power is very low or very high |
| `strain` | tool_wear_min × torque_nm | Overstrain failure happens when this exceeds a limit that depends on Type |
| `type_L`, `type_M`, `type_H` | one-hot of `type` (fixed categories, done in `build_features`; see ARCHITECTURE D8) | Product quality variant |

Scaling (StandardScaler) is done **inside the pipeline**, only for models that need it (logistic regression).

## Split strategy
1. Stratified on `machine_failure`: **80 % train / 20 % test**, `random_state = 42`.
2. Test set saved by index and used **once** in `src/evaluate.py`.
3. Model selection: `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on train.
4. Imbalance handling options compared in CV:
   - `class_weight="balanced"` (LogReg, RF) / `scale_pos_weight = n_neg / n_pos` (XGBoost)
   - SMOTE inside `imblearn.pipeline.Pipeline` (applied only to training folds)

## Threshold rule
Using out-of-fold predicted probabilities on the training set:
among all thresholds with **recall ≥ 0.85**, pick the one with the **highest precision**.
If no threshold reaches recall 0.85, maximise **F2** (recall weighted twice as much as precision).
Store the chosen value in `models/metadata.json`.

## Data validation checks (`src/data.py`)
- Expected columns present, correct dtypes
- No missing values, no duplicate `udi`
- `type ∈ {L, M, H}`, target ∈ {0, 1}
- Sensor values inside physical ranges defined in `src/config.py`
- Fail loudly (raise a clear `ValueError`) if any check fails
