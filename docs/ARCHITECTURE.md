# Architecture — Predictive Maintenance

## 1. System overview
```
 ┌──────────────┐   CSV    ┌──────────────────────────── Training (offline) ───────────┐
 │ AI4I 2020    │ ───────▶ │ data.py → features.py → train.py (CV, tuning, threshold)  │
 │ raw dataset  │          │                         → evaluate.py (test set, SHAP)    │
 └──────────────┘          └───────────────┬───────────────────────────────────────────┘
                                           │ models/model.joblib + metadata.json
                                           ▼
 ┌─────────────────┐  HTTP/JSON  ┌──────────────────────────┐
 │ Streamlit       │ ──────────▶ │ FastAPI service          │
 │ dashboard       │ ◀────────── │  /predict  /predict/batch│
 └─────────────────┘             │  /health   /model-info   │
          ▲                      │  uses features.py + model│
          │ browser              └──────────────────────────┘
       user                               ▲
                                          │ HTTP/JSON
                              ┌───────────────────────┐
                              │ sensor_simulator.py   │  (bonus: edge device / ESP32)
                              └───────────────────────┘
```
Training and serving share `src/features.py`, so a feature is defined exactly once.

## 2. Folder layout
```
predictive-maintenance/

├── README.md
├── docs/                 PRD.md, ARCHITECTURE.md, DATA.md,
├── data/
│   ├── raw/              ai4i2020.csv            (git-ignored)
│   └── sample/           sample_batch.csv        (small, committed, for demo/tests)
├── notebooks/            01_eda.ipynb, 02_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py         paths, RANDOM_STATE, column names, physical ranges
│   ├── data.py           load_raw(), validate(), split()
│   ├── features.py       build_features(df), FEATURE_COLUMNS
│   ├── models.py         candidate pipelines (logreg, rf, xgb, with/without SMOTE)
│   ├── train.py          CV comparison, tuning, threshold, save artefacts  (python -m src.train)
│   ├── evaluate.py       one-time test metrics + plots → reports/
│   ├── explain.py        SHAP plots (train data only) → reports/figures/
│   └── predict.py        load_model(), predict_df()  (used by API)
├── api/
│   ├── __init__.py
│   ├── main.py           FastAPI app, routes
│   └── schemas.py        pydantic request/response models
├── app/
│   └── dashboard.py      Streamlit UI (calls API)
├── scripts/
│   └── sensor_simulator.py
├── models/               model.joblib, metadata.json  (committed; small)
├── reports/              metrics.json, model_comparison.csv, figures/*.png
├── tests/                test_data.py, test_features.py, test_predict.py, test_api.py
├── docker/
│   ├── api.Dockerfile
│   └── dashboard.Dockerfile
├── docker-compose.yml
├── .github/workflows/ci.yml
├── requirements.txt      runtime deps (pinned)
├── requirements-dev.txt  pytest, ruff, jupyter, ...
├── pyproject.toml        ruff + pytest config
└── .gitignore
```

## 3. Components
| Component | Responsibility | Depends on |
|---|---|---|
| `src/config.py` | Constants, paths, column names, valid sensor ranges | — |
| `src/data.py` | Read CSV, rename columns to snake_case, schema validation, stratified split | config |
| `src/features.py` | Pure function `build_features(df) -> df`; no fitting, no I/O | config |
| `src/models.py` | Factory returning named candidate pipelines | features, sklearn, imblearn, xgboost |
| `src/train.py` | CV comparison → pick best → tune → threshold → fit on full train → save | data, features, models |
| `src/evaluate.py` | One-time test-set evaluation (refuses a second run), plots, PRD pass/miss | train artefacts |
| `src/explain.py` | SHAP summary + bar plot on a TRAIN sample (no test data needed) | train artefacts |
| `src/predict.py` | Load pipeline + metadata once; `predict_df()` returns proba + label | features |
| `api/` | HTTP layer only: validation, routing, error handling; no ML logic | src/predict |
| `app/dashboard.py` | UI only; talks to API through `API_URL` | API |

## 4. API contract
Base URL: `http://localhost:8000` (local) / `https://<user>-pdm-api.hf.space` (deployed)

### `GET /health` → `200`
```json
{"status": "ok", "model_loaded": true}
```

### `GET /model-info` → `200`
```json
{"model_name": "xgboost_weighted", "version": "1.0.0", "trained_at": "2026-12-01T12:00:00Z",
 "threshold": 0.31, "features": ["..."], "test_metrics": {"pr_auc": 0.0, "recall": 0.0, "precision": 0.0, "f1": 0.0}}
```

### `POST /predict`
Request:
```json
{"type": "M", "air_temperature_k": 298.1, "process_temperature_k": 308.6,
 "rotational_speed_rpm": 1551, "torque_nm": 42.8, "tool_wear_min": 0}
```
Response `200`:
```json
{"failure_probability": 0.03, "prediction": 0, "label": "no_failure",
 "threshold": 0.31, "model_version": "1.0.0"}
```
Validation (pydantic, `422` on error): `type ∈ {L, M, H}`; numeric ranges from `src/config.py`
(e.g. air temp 290–310 K, process temp 300–320 K, rpm 1000–3000, torque 0–100 Nm,
tool wear 0–300 min — confirm against EDA and keep a margin).

### `POST /predict/batch`
Request: `{"readings": [<reading>, ...]}` (max 1000) → Response: `{"results": [<prediction>, ...]}`

## 5. Deployment
| Environment | How |
|---|---|
| Local dev | `uvicorn` + `streamlit run` |
| Local full stack | `docker compose up --build` → API :8000, dashboard :8501 |
| Production (demo) | Two Hugging Face Spaces (Docker SDK): `pdm-api` and `pdm-dashboard`. Spaces expose port **7860** by default (or set `app_port` in the Space README YAML). Dashboard gets `API_URL` as a Space variable. |
| CI | GitHub Actions: install → ruff → pytest on push / PR |

## 6. Key decisions (ADR-lite)
| # | Decision | Why | Alternatives considered |
|---|---|---|---|
| D1 | AI4I 2020 first, CMAPSS later | Tabular, small, fits a 2-week scope | CMAPSS (time-series RUL, more complex) |
| D2 | PR-AUC as primary metric | ~3.4 % positives; accuracy and ROC-AUC look misleadingly good | Accuracy, ROC-AUC |
| D3 | Save whole pipeline in one joblib | Preprocessing can't drift from the model | Separate scaler + model files |
| D4 | Threshold stored in metadata, applied in `predict.py` | Business rule separate from model; easy to change | Hard-coded 0.5 |
| D5 | Dashboard calls API instead of loading the model | Realistic service architecture; one place for inference | Dashboard loads joblib directly |
| D6 | Two Spaces instead of one container | One process per container; independent scaling and logs | supervisord in one container |
| D7 | No MLflow in v1 | Scope; `reports/model_comparison.csv` is enough | MLflow, Weights & Biases |
| D8 | `type` one-hot encoded inside `build_features` (`type_L/M/H`, fixed categories from `config.VALID_TYPES`), not in a `ColumnTransformer` | Categories are fixed and known, so nothing is fitted (no leakage risk); API/dashboard/training share one numeric frame; `FEATURE_COLUMNS` is the real final list and SHAP names stay clean; single-row input always yields all 3 columns; unknown types fail loudly | `ColumnTransformer` + `OneHotEncoder` in the pipeline (standard, but final feature names only exist after fitting); ordinal L=0/M=1/H=2 (imposes an order/spacing the data doesn't justify) |
| D9 | SMOTE variants use `SMOTENC` on the raw columns (`type` categorical) *before* `build_features`: `select_raw -> SMOTENC -> build_features -> model` | Plain SMOTE interpolates the one-hot `type_*` columns (fractional values) and the engineered features (`power_w`, `strain` are products, so interpolating them makes synthetic rows physically inconsistent). Sampling raw columns fixes both and keeps `build_features` the single source of truth | SMOTENC on the one-hot columns (can produce all-zero / multi-hot types, does not fix engineered features); plain SMOTE + rounding; no SMOTE |
