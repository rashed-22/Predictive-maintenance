# Product Requirements Document — Predictive Maintenance

| Field | Value |
|---|---|
| Project | Predictive Maintenance — Machine Failure Prediction |
| Version | 1.0 |
| Owner | (your name) |
| Status | Draft → In progress |
| Timeline | ~2 weeks (Level 1 portfolio project) |

## 1. Problem
Unplanned machine downtime is expensive in manufacturing. If a model can flag a likely
failure from routine sensor readings, maintenance can be scheduled *before* the breakdown.
Missing a real failure (false negative) costs far more than an unnecessary inspection
(false positive).

## 2. Goal
Build an end-to-end, deployed ML system that takes one or many sensor readings and returns
a failure probability and a fail / no-fail decision, with an explanation of which
sensors drove the prediction.

## 3. Target users
- **Maintenance engineer** — checks a single machine reading in the dashboard.
- **Line supervisor** — uploads a CSV of readings and sees which machines are at risk.
- **Other software / IoT device** — calls the REST API directly (e.g. an edge gateway).
- **Recruiter / interviewer** — opens the live demo and README.

## 4. Scope
### In scope (v1)
- AI4I 2020 dataset, binary target `Machine failure`
- EDA, physics-based feature engineering, imbalance handling
- Model comparison: Logistic Regression, Random Forest, XGBoost
- Threshold tuning, SHAP explainability
- FastAPI service, Streamlit dashboard, Docker, CI, Hugging Face Spaces deploy

### Out of scope (v1) → future work
- Failure *type* classification (multi-label TWF/HDF/PWF/OSF/RNF)
- NASA CMAPSS Remaining Useful Life (RUL) regression
- Authentication, database, real-time streaming infrastructure
- Model retraining pipeline / MLOps platform

## 5. Functional requirements
| ID | Requirement |
|---|---|
| FR-1 | Load the raw CSV and validate schema (columns, dtypes, value ranges, no missing values). |
| FR-2 | EDA notebook with target imbalance, distributions, failure vs normal comparisons, correlations, written insights. |
| FR-3 | Feature engineering in one module: `temp_diff`, `power_w`, `strain`, encoded `Type`. |
| FR-4 | Train and compare at least 3 models using stratified CV with PR-AUC as the primary metric. |
| FR-5 | Handle class imbalance (class weights and/or SMOTE inside the pipeline) and compare approaches. |
| FR-6 | Choose a decision threshold from CV predictions (rule in `docs/DATA.md`), store it in metadata. |
| FR-7 | Final evaluation on the untouched test set; save metrics, confusion matrix, PR curve, SHAP plots to `reports/`. |
| FR-8 | Persist the full pipeline (`models/model.joblib`) and `models/metadata.json`. |
| FR-9 | API: `GET /health`, `GET /model-info`, `POST /predict` (single), `POST /predict/batch` (list). |
| FR-10 | API validates input (types, physical ranges, `Type ∈ {L,M,H}`) and returns clear 422 errors. |
| FR-11 | Dashboard: single prediction form with sliders, CSV batch upload with downloadable results, model info / explainability tab. |
| FR-12 | Dashboard talks to the API over HTTP (URL from env var `API_URL`). |
| FR-13 | Docker images for API and dashboard; `docker compose up` runs the full system locally. |
| FR-14 | Public deployment of API and dashboard on Hugging Face Spaces. |
| FR-15 | README with problem, results table, architecture diagram, screenshots, live links, how to run. |
| FR-16 (bonus) | Sensor simulator script that streams readings to the API (edge-device scenario). |

## 6. Non-functional requirements
| ID | Requirement |
|---|---|
| NFR-1 | Reproducible: fixed random seed, pinned dependency versions, `python -m src.train` rebuilds the model. |
| NFR-2 | `/predict` p95 latency < 100 ms locally for a single reading. |
| NFR-3 | Test coverage of `src/` and `api/` ≥ 70 %; CI runs ruff + pytest on every push. |
| NFR-4 | Docker image for the API < 1 GB. |
| NFR-5 | No secrets or raw data in git. |
| NFR-6 | Code readable by a junior: type hints, docstrings, no clever one-liners. |

## 7. Success metrics (on the held-out test set)
| Metric | Target |
|---|---|
| PR-AUC | ≥ 0.80 |
| Recall (failure class) at chosen threshold | ≥ 0.85 |
| Precision at chosen threshold | ≥ 0.60 |
| Beats Logistic Regression baseline on PR-AUC | yes |

Targets are guidelines; if not reached, document why in `reports/` and the README.

## 8. Milestones
| Phase | Deliverable | Est. days |
|---|---|---|
| 0 | Repo scaffold, environment, docs in place | 0.5 |
| 1 | Data loading + validation (FR-1) | 0.5 |
| 2 | EDA notebook (FR-2) | 1.5 |
| 3 | Feature engineering + tests (FR-3) | 1 |
| 4 | Baseline + model comparison (FR-4, FR-5) | 2 |
| 5 | Tuning, threshold, SHAP, final eval, export (FR-6–8) | 1.5 |
| 6 | FastAPI + tests (FR-9, FR-10) | 1.5 |
| 7 | Streamlit dashboard (FR-11, FR-12) | 1.5 |
| 8 | Docker + compose (FR-13) | 1 |
| 9 | CI + Hugging Face deploy (FR-14) | 1 |
| 10 | README + polish (FR-15) | 1 |
| Bonus | Sensor simulator (FR-16) | 0.5 |
