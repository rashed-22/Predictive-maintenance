"""FastAPI application and routes (HTTP layer only; all ML logic lives in ``src/predict.py``)."""

import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    BatchRequest,
    BatchResponse,
    Health,
    ModelInfo,
    PredictionResponse,
    SensorReading,
)
from src import config
from src.predict import LoadedModel, load_model, predict_df

logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load the model once at startup; keep serving /health (503 elsewhere) if it is missing."""
    try:
        app.state.model = load_model()
    except Exception:  # missing or corrupt artefacts must not crash the process
        logger.exception("Model could not be loaded - prediction routes will return 503")
        app.state.model = None
    yield


app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts machine failure from sensor readings (AI4I 2020 dataset).",
    version=config.MODEL_VERSION,
    lifespan=lifespan,
)

_origins = os.getenv("CORS_ORIGINS", config.API_DEFAULT_CORS_ORIGINS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins.split(",") if o.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Log method, path, status and duration. Bodies and client addresses are never logged."""
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("%s %s -> %d (%.1f ms)", request.method, request.url.path, status, duration_ms)


def get_model(request: Request) -> LoadedModel:
    """Dependency: the loaded model, or HTTP 503 if loading failed at startup."""
    model: LoadedModel | None = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


ModelDep = Annotated[LoadedModel, Depends(get_model)]


def _predict(readings: list[SensorReading], model: LoadedModel) -> list[PredictionResponse]:
    """Convert validated readings to a DataFrame, predict, and build response objects."""
    df = pd.DataFrame([r.model_dump() for r in readings])
    out = predict_df(df, model)
    version = str(model.metadata["version"])
    return [
        PredictionResponse(
            failure_probability=float(row.failure_probability),
            prediction=int(row.prediction),
            label=row.label,
            threshold=model.threshold,
            model_version=version,
        )
        for row in out.itertuples(index=False)
    ]


@app.get("/health", response_model=Health)
def health(request: Request) -> Health:
    """Liveness check; also reports whether the model is loaded."""
    return Health(status="ok", model_loaded=getattr(request.app.state, "model", None) is not None)


@app.get("/model-info", response_model=ModelInfo)
def model_info(model: ModelDep) -> ModelInfo:
    """Return model name, version, threshold, feature list and held-out test metrics."""
    meta = model.metadata
    return ModelInfo(
        model_name=meta["model_name"],
        version=meta["version"],
        trained_at=meta["trained_at"],
        threshold=model.threshold,
        features=meta["features"],
        test_metrics=meta.get("test_metrics", {}),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(reading: SensorReading, model: ModelDep) -> PredictionResponse:
    """Predict failure for a single sensor reading."""
    return _predict([reading], model)[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, model: ModelDep) -> BatchResponse:
    """Predict failure for up to 1000 readings; results keep the request order."""
    return BatchResponse(results=_predict(body.readings, model))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", config.API_DEFAULT_PORT)))
