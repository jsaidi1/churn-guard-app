"""
Pour exécuter les tests de ce module (avec un rapport de coverage), utilisez la commande suivante dans votre terminal à la racine du projet :
> uv run pytest --cov=churnguard --cov-report=term-missing
"""

import pickle
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

import mlflow
from churnguard.evaluate import compute_metrics
from churnguard.train import (
    promote_best_model_to_staging,
    save_best_model_in_registry,
    save_model,
    train_and_log_model,
    train_and_log_models,
    train_model,
)


@pytest.fixture
def sample_data() -> tuple[pd.DataFrame, pd.Series]:
    """Retourne un petit dataset de test pour le churn."""
    X = pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female", "Male"],
            "SeniorCitizen": [0, 1, 0, 1],
            "Partner": ["Yes", "No", "Yes", "No"],
            "Dependents": ["No", "Yes", "No", "Yes"],
            "tenure": [1, 12, 24, 36],
            "PhoneService": ["No", "Yes", "Yes", "Yes"],
            "MultipleLines": ["No phone service", "No", "Yes", "No"],
            "InternetService": ["DSL", "Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No", "No internet service"],
            "OnlineBackup": ["Yes", "No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "Yes", "No", "No internet service"],
            "TechSupport": ["No", "No", "Yes", "No internet service"],
            "StreamingTV": ["No", "Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "Yes", "No internet service"],
            "Contract": [
                "Month-to-month",
                "One year",
                "Two year",
                "Month-to-month",
            ],
            "PaperlessBilling": ["Yes", "No", "Yes", "No"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            "MonthlyCharges": [29.85, 56.95, 89.10, 45.30],
            "TotalCharges": [29.85, 683.40, 2138.40, 1630.80],
        }
    )

    y = pd.Series([0, 1, 1, 0])

    return X, y


def test_train_model_returns_fitted_pipeline(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """train_model doit retourner un Pipeline entraîné capable de prédire."""
    X, y = sample_data

    model = train_model(
        X=X,
        y=y,
        model_name="random_forest",
        params={
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
    )

    assert isinstance(model, Pipeline)

    predictions = model.predict(X)

    assert len(predictions) == len(y)
    assert set(predictions).issubset({0, 1})


def test_compute_metrics_returns_expected_keys(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """compute_metrics doit retourner les métriques attendues."""
    X, y = sample_data

    model = train_model(
        X=X,
        y=y,
        model_name="random_forest",
        params={
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42,
            "n_jobs": -1,
        },
    )

    metrics = compute_metrics(model, X, y)

    expected_keys = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == expected_keys


def test_save_model_creates_file_and_is_loadable(
    sample_data: tuple[pd.DataFrame, pd.Series], tmp_path
) -> None:
    """save_model doit créer un fichier .pkl et permettre de recharger le modèle."""

    X, y = sample_data

    model = train_model(
        X=X,
        y=y,
        model_name="random_forest",
        params={"n_estimators": 10, "random_state": 42},
    )

    # Dossier temporaire (pytest)
    save_dir = tmp_path / "models"

    save_model(model, str(save_dir))

    model_path = save_dir / "best_model.pkl"

    # 1. Vérifie que le fichier existe
    assert model_path.exists()

    # 2. Vérifie que le modèle est chargeable
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)

    assert isinstance(loaded_model, Pipeline)

    # 3. Vérifie que le modèle fonctionne
    preds = loaded_model.predict(X)

    assert len(preds) == len(y)


def test_train_model_raises_error_for_unknown_model(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """train_model doit lever une erreur pour un modèle inconnu."""

    X, y = sample_data

    with pytest.raises(ValueError, match="Modèle non supporté"):
        train_model(
            X=X,
            y=y,
            model_name="invalid_model",
            params={},
        )


def test_train_model_supports_logistic_regression(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """train_model doit supporter LogisticRegression."""
    X, y = sample_data

    model = train_model(
        X=X,
        y=y,
        model_name="logistic_regression",
        params={
            "max_iter": 1000,
            "random_state": 42,
        },
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)


def test_train_model_supports_gradient_boosting(
    sample_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """train_model doit supporter GradientBoostingClassifier."""
    X, y = sample_data

    model = train_model(
        X=X,
        y=y,
        model_name="gradient_boosting",
        params={
            "n_estimators": 10,
            "random_state": 42,
        },
    )

    predictions = model.predict(X)

    assert len(predictions) == len(y)


def test_train_and_log_model_logs_metrics_params_and_model(
    tmp_path, sample_data: tuple[pd.DataFrame, pd.Series]
) -> None:
    """train_and_log_model doit logger paramètres, métriques et artefact modèle."""

    # mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    # mlflow.set_tracking_uri(str(tmp_path / "mlruns"))
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("test-churnguard")

    X, y = sample_data

    model, metrics = train_and_log_model(
        X_train=X,
        X_test=X,
        y_train=y,
        y_test=y,
        model_name="random_forest",
        params={"n_estimators": 10, "random_state": 42},
        register=False,
    )

    assert isinstance(model, Pipeline)

    expected_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert set(metrics.keys()) == expected_metrics

    run = mlflow.last_active_run()
    assert run is not None

    client = mlflow.tracking.MlflowClient()
    run_data = client.get_run(run.info.run_id)

    assert run_data.data.params["model_name"] == "random_forest"
    assert run_data.data.params["n_estimators"] == "10"
    assert run_data.data.params["random_state"] == "42"

    assert expected_metrics.issubset(set(run_data.data.metrics.keys()))

    model_uri = f"runs:/{run.info.run_id}/model"
    loaded_model = mlflow.sklearn.load_model(model_uri)

    assert loaded_model is not None


@patch("churnguard.train.train_and_log_model")
@patch("churnguard.train.split_data")
@patch("churnguard.train.preprocess")
@patch("churnguard.train.load_data")
@patch("churnguard.train.mlflow.set_experiment")
@patch("churnguard.train.mlflow.set_tracking_uri")
def test_train_and_log_models_trains_three_models(
    mock_set_tracking_uri: MagicMock,
    mock_set_experiment: MagicMock,
    mock_load_data: MagicMock,
    mock_preprocess: MagicMock,
    mock_split_data: MagicMock,
    mock_train_and_log_model: MagicMock,
) -> None:
    """train_and_log_models doit entraîner et logger les 3 modèles."""

    # Fake dataset
    df = pd.DataFrame({"a": [1, 2]})

    X = pd.DataFrame(
        {
            "tenure": [1, 2],
            "MonthlyCharges": [10.0, 20.0],
            "TotalCharges": [100.0, 200.0],
            "SeniorCitizen": [0, 1],
        }
    )

    y = pd.Series([0, 1])

    # Mocks
    mock_load_data.return_value = df
    mock_preprocess.return_value = (X, y)

    mock_split_data.return_value = (
        X,
        X,
        y,
        y,
    )

    mock_train_and_log_model.return_value = (
        MagicMock(),
        {
            "accuracy": 0.8,
            "precision": 0.7,
            "recall": 0.6,
            "f1": 0.65,
            "roc_auc": 0.75,
        },
    )

    # Run
    train_and_log_models()

    # Vérifie configuration MLflow
    mock_set_tracking_uri.assert_called_once()
    mock_set_experiment.assert_called_once()

    # Vérifie pipeline data
    mock_load_data.assert_called_once()
    mock_preprocess.assert_called_once()
    mock_split_data.assert_called_once()

    # Vérifie les 3 entraînements
    assert mock_train_and_log_model.call_count == 3

    called_models = [call.kwargs["model_name"] for call in mock_train_and_log_model.call_args_list]

    assert set(called_models) == {
        "logistic_regression",
        "random_forest",
        "gradient_boosting",
    }


@patch("churnguard.train.mlflow.register_model")
@patch("churnguard.train.mlflow.search_runs")
@patch("churnguard.train.mlflow.MlflowClient")
def test_save_best_model_in_registry_registers_best_run(
    mock_mlflow_client: MagicMock,
    mock_search_runs: MagicMock,
    mock_register_model: MagicMock,
) -> None:
    """save_best_model_in_registry doit enregistrer le meilleur run MLflow."""

    # Mock client MLflow
    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance

    # Mock expérience
    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    mock_client_instance.get_experiment_by_name.return_value = mock_experiment

    # Mock meilleur run
    mock_runs = pd.DataFrame(
        [
            {
                "run_id": "best-run-id",
                "metrics.f1": 0.82,
            }
        ]
    )

    mock_search_runs.return_value = mock_runs

    # Mock model version
    mock_model_version = MagicMock()
    mock_model_version.version = 1

    mock_register_model.return_value = mock_model_version

    # Run
    save_best_model_in_registry()

    # Vérifie récupération expérience
    mock_client_instance.get_experiment_by_name.assert_called_once()

    # Vérifie search_runs
    mock_search_runs.assert_called_once()

    # Vérifie enregistrement modèle
    mock_register_model.assert_called_once_with(
        model_uri="runs:/best-run-id/model",
        name="churnguard",
    )


@patch("churnguard.train.mlflow.search_runs")
@patch("churnguard.train.mlflow.MlflowClient")
def test_save_best_model_in_registry_handles_empty_runs(
    mock_mlflow_client: MagicMock,
    mock_search_runs: MagicMock,
) -> None:
    """save_best_model_in_registry doit gérer le cas sans run."""

    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance

    mock_experiment = MagicMock()
    mock_experiment.experiment_id = "123"

    mock_client_instance.get_experiment_by_name.return_value = mock_experiment

    # DataFrame vide
    mock_search_runs.return_value = pd.DataFrame()

    save_best_model_in_registry()

    mock_search_runs.assert_called_once()


@patch("churnguard.train.mlflow.MlflowClient")
def test_promote_best_model_to_staging_transitions_model(
    mock_mlflow_client: MagicMock,
) -> None:
    """promote_best_model_to_staging doit promouvoir le modèle en Staging."""

    # Mock client MLflow
    mock_client_instance = MagicMock()
    mock_mlflow_client.return_value = mock_client_instance

    # Mock version modèle
    mock_model_version = MagicMock()
    mock_model_version.version = 1

    mock_client_instance.get_latest_versions.return_value = [mock_model_version]

    # Run
    promote_best_model_to_staging()

    # Vérifie récupération dernière version
    mock_client_instance.get_latest_versions.assert_called_once_with(
        "churnguard",
        stages=["None"],
    )

    # Vérifie promotion Staging
    mock_client_instance.transition_model_version_stage.assert_called_once_with(
        name="churnguard",
        version=1,
        stage="Staging",
        archive_existing_versions=True,
    )
