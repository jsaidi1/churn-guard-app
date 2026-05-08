from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException

from api.model_loader import (
    MODEL_NAME,
    get_model,
    get_model_version,
    load_model,
)
from api.schemas import CustomerFeatures, PredictionResponse


# ---------------------------------------------------------
# Chargement du modèle UNE SEULE FOIS au démarrage
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Charge le modèle au démarrage de l'API."""

    load_model()
    yield


# ---------------------------------------------------------
# Création API FastAPI
# ---------------------------------------------------------
app = FastAPI(
    title="ChurnGuard API",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# GET /health
# ---------------------------------------------------------
@app.get("/health")
def health() -> dict[str, str]:
    """Retourne l’état de santé de l’API."""

    if get_model() is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {
        "status": "ok",
        "model": MODEL_NAME,
        "version": get_model_version(),
    }


# ---------------------------------------------------------
# Fonction interne prédiction unique
# ---------------------------------------------------------
def predict_one(
    payload: CustomerFeatures,
) -> PredictionResponse:
    """Prédit le churn pour un client."""

    model: Any = get_model()

    # modèle indisponible
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    # payload -> DataFrame pandas
    df = pd.DataFrame([payload.model_dump()])

    # prédiction binaire
    prediction = model.predict(df)[0]

    # probabilité churn
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(df)[0][1])
    else:
        probability = float(prediction)

    return PredictionResponse(
        churn=bool(prediction),
        probability=probability,
    )


# ---------------------------------------------------------
# POST /predict
# ---------------------------------------------------------
@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    payload: CustomerFeatures,
) -> PredictionResponse:
    """Prédit le churn pour un client."""

    return predict_one(payload)


# ---------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------
@app.post(
    "/predict/batch",
    response_model=list[PredictionResponse],
)
def predict_batch(
    payloads: list[CustomerFeatures],
) -> list[PredictionResponse]:
    """Prédit le churn pour un batch de clients."""

    # batch vide
    if not payloads:
        raise HTTPException(
            status_code=400,
            detail="Batch cannot be empty",
        )

    # batch > 100
    if len(payloads) > 100:
        raise HTTPException(
            status_code=400,
            detail="Batch size cannot exceed 100",
        )

    return [predict_one(payload) for payload in payloads]
