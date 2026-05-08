# import os
# from typing import Any

# import mlflow
# import mlflow.pyfunc
# from mlflow.tracking import MlflowClient

# MODEL_NAME = "churnguard"
# # MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
# MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
# STAGE_TO_LOAD_FROM = "Production"

# model: Any | None = None
# model_version: str = "not_loaded"


# def load_model() -> None:
#     """Charge la dernière version du modèle en stage Production."""

#     global model, model_version

#     mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

#     client = MlflowClient()

#     # Toutes les versions du modèle
#     versions = client.search_model_versions(
#         filter_string=f"name = '{MODEL_NAME}'"
#     )

#     # Filtre uniquement le stage <STAGE_TO_LOAD_FROM>
#     stage_versions = [
#         version
#         for version in versions
#         if version.current_stage == STAGE_TO_LOAD_FROM
#     ]

#     if not stage_versions:
#         raise ValueError(
#             f"Aucune version '{STAGE_TO_LOAD_FROM}' trouvée pour {MODEL_NAME}"
#         )

#     # Version numérique maximale
#     latest_stage_version = max(
#         stage_versions,
#         key=lambda version: int(version.version),
#     )

#     model_version = str(
#         latest_stage_version.version
#     )

#     model_uri = (
#         f"models:/{MODEL_NAME}/{model_version}"
#     )

#     model = mlflow.pyfunc.load_model(model_uri)

#     print(
#         f"Modèle chargé : {MODEL_NAME} "
#         f"({STAGE_TO_LOAD_FROM} v{model_version})"
#     )


# def get_model() -> Any:
#     """Retourne le modèle chargé."""
#     return model


# def get_model_version() -> str:
#     """Retourne la version du modèle chargé."""
#     return model_version

# ------------------------------------------
import os
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

MODEL_NAME = "churnguard"
STAGE_TO_LOAD_FROM = "Production"
MODEL_URI = f"models:/{MODEL_NAME}/{STAGE_TO_LOAD_FROM}"

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)

model: Any | None = None
model_version: str = "not_loaded"


def load_model() -> None:
    """Charge le modèle depuis le stage Production MLflow."""

    global model, model_version

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Chargement fiable du modèle par stage
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # Récupération du numéro de version pour /health
    client = MlflowClient()
    registered_model = client.get_registered_model(MODEL_NAME)

    stage_versions = [
        version
        for version in registered_model.latest_versions
        if version.current_stage == STAGE_TO_LOAD_FROM
    ]

    if stage_versions:
        latest_stage_version = max(
            stage_versions,
            key=lambda version: int(version.version),
        )
        model_version = str(latest_stage_version.version)
    else:
        model_version = STAGE_TO_LOAD_FROM

    print(f"Modèle chargé : {MODEL_NAME} ({STAGE_TO_LOAD_FROM} v{model_version})")


def get_model() -> Any:
    """Retourne le modèle chargé."""
    return model


def get_model_version() -> str:
    """Retourne la version du modèle chargé."""
    return model_version
