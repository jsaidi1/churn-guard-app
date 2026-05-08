"""
- Nom du script : train.py
- But : entraînement + log + registry + promotion + check loading promoted model
- Execution directe : python -m churnguard.train
"""

import os
import pickle
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import mlflow
import mlflow.sklearn
from churnguard.data import load_data, preprocess, split_data
from churnguard.evaluate import compute_metrics
from mlflow.models import infer_signature

IN_CSV = "data/telco_churn.csv"
# MLFLOW_LOCAL_URI = "http://127.0.0.1:5000"
MLFLOW_LOCAL_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000",
)
EXPERIMENT_NAME = "churnguard"
REGISTERED_MODEL_NAME = "churnguard"


def console_msg_intro() -> None:
    """Clear & Affichage d'un message d'intro dans la console avec un formatage spécifique."""

    os.system("cls" if os.name == "nt" else "clear")
    print("""===========================================================================================
    train.py : entraînement + log + registry + promotion + check loading promoted model
===========================================================================================""")


def train_model(X: pd.DataFrame, y: pd.Series, model_name: str, params: dict[str, Any]) -> Pipeline:
    """Entraîne un modèle de machine learning avec les données d'entraînement et retourne le pipeline entraîné."""

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    preprocess = ColumnTransformer(
        [
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if model_name == "random_forest":
        classifier = RandomForestClassifier(**params)
    elif model_name == "logistic_regression":
        classifier = LogisticRegression(**params)
    elif model_name == "gradient_boosting":
        classifier = GradientBoostingClassifier(**params)
    else:
        raise ValueError(f"Modèle non supporté : {model_name}")

    model = Pipeline(
        [
            ("prep", preprocess),
            ("clf", classifier),
        ]
    )

    model.fit(X, y)

    return model


def save_model(model: Pipeline, path_dir: str) -> None:
    """Sauvegarde le modèle entraîné sur le disque."""

    os.makedirs(path_dir, exist_ok=True)

    with open(os.path.join(path_dir, "best_model.pkl"), "wb") as f:
        pickle.dump(model, f)


def train_and_log_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_name: str,
    params: dict[str, Any],
    register: bool = False,
) -> tuple[Pipeline, dict[str, float]]:
    """Entraîne un modèle puis log automatiquement le run dans MLflow (paramètres, métriques (les 5), artefacts (modèle), signature, exemple d'input)."""

    with mlflow.start_run(run_name=f"churnguard-{model_name}"):
        model = train_model(
            X=X_train,
            y=y_train,
            model_name=model_name,
            params=params,
        )

        metrics = compute_metrics(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        # 1) Paramètres
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(params)

        # 2) Métriques : accuracy, precision, recall, f1, roc_auc
        mlflow.log_metrics(metrics)

        # 3) Exemple d’input
        input_example = X_train.head(5)

        # 4) Signature MLflow
        signature = infer_signature(
            input_example,
            model.predict(input_example),
        )

        # 5) Artefact modèle
        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            signature=signature,
            input_example=input_example,
            registered_model_name="churnguard" if register else None,
        )

    return model, metrics


def train_and_log_models() -> None:
    """Entraîne les 3 modèles et log dans MLflow."""
    try:
        print("Etape 1/5 : Entraînement et logging des 3 modèles dans MLflow...")

        # Connexion MLflow
        mlflow.set_tracking_uri(MLFLOW_LOCAL_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        # Chargement données
        df = load_data(IN_CSV)
        X, y = preprocess(df)

        # Split données
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Configuration des modèles
        models = {
            "logistic_regression": {
                "max_iter": 1000,
                "random_state": 42,
            },
            "random_forest": {
                "n_estimators": 100,
                "random_state": 42,
            },
            "gradient_boosting": {
                "n_estimators": 100,
                "random_state": 42,
            },
        }

        # Entraînement + logging
        for model_name, params in models.items():
            print(f"Training {model_name}...")

            model, metrics = train_and_log_model(
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                model_name=model_name,
                params=params,
            )

            print(f"{model_name} metrics: {metrics}")

        print(
            "\n[ok] : Entraînement et logging des 3 modèles dans MLflow est terminé avec succès (Etape 1/5)."
        )
        print("-" * 150)
    except Exception as e:
        print(f"Erreur lors de l'entraînement et du logging des 3 modèles (Etape 1/5): {e}")
        print("-" * 150)


def save_best_model_in_registry() -> None:
    """Enregistre dans le registry MLflow le modèle ayant le meilleur F1-score."""

    try:
        print(
            "Etape 2/5 : Enregistrement dans le registry MLflow le modèle ayant le meilleur F1-score..."
        )

        client = mlflow.MlflowClient()

        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            raise ValueError(f"Expérience MLflow introuvable : {EXPERIMENT_NAME}")

        # runs = mlflow.search_runs(
        #     experiment_ids=[experiment.experiment_id],
        #     order_by=["metrics.f1 DESC"],
        #     max_results=1,
        # )

        runs = pd.DataFrame(
            mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.f1 DESC"],
                max_results=1,
            )
        )

        if runs.empty:
            raise ValueError("Aucun run MLflow trouvé pour cette expérience.")

        best_run = runs.iloc[0]
        best_run_id = best_run["run_id"]
        best_f1 = best_run["metrics.f1"]

        model_uri = f"runs:/{best_run_id}/model"

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME,
        )

        print(
            f"- Meilleur modèle enregistré : run_id={best_run_id}, "
            f"f1={best_f1:.4f}, version={model_version.version}"
        )
        print(
            "\n[ok] : Enregistrement dans le registry MLflow le modèle ayant le meilleur F1-score avec succès (Etape 2/5)."
        )
        print("-" * 150)
    except Exception as e:
        print(
            f"Erreur lors de l'enregistrement du meilleur modèle dans le registry (Etape 2/5) : {e}"
        )
        print("-" * 150)


def promote_best_model_to_staging() -> None:
    """Promouvoit le modèle enregistré ayant le meilleur F1-score en stage."""

    try:
        print("Etape 3/5 : Promotion du modèle ayant le meilleur F1-score en stage (best model)...")
        client = mlflow.MlflowClient()

        # Récupérer la dernière version du modèle enregistré
        latest_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["None"])[0]

        # Promouvoir la version en stage
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=latest_version.version,
            stage="Staging",
            archive_existing_versions=True,
        )

        print(f"- Modèle version {latest_version.version} promu en stage.")
        print(
            "\n[ok] : Promotion du modèle ayant le meilleur F1-score en stage avec succès (Etape 3/5)."
        )
        print("-" * 150)

    except Exception as e:
        print(f"Erreur lors de la promotion du meilleur modèle en stage (Etape 3/5) : {e}")
        print("-" * 150)


def promote_best_model_to_production() -> None:
    """Promouvoit le modèle en stage en production."""

    try:
        print("Etape 4/5 : Promotion du modèle en stage en production...")
        client = mlflow.MlflowClient()

        # Récupérer la version du modèle en stage
        staging_version = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=["Staging"])[0]

        # Promouvoir la version en production
        client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=staging_version.version,
            stage="Production",
            archive_existing_versions=True,
        )

        print(f"- Modèle version {staging_version.version} promu en production.")
        print("\n[ok] : Promotion du modèle en stage en production avec succès (Etape 4/5).")
        print("-" * 150)

    except Exception as e:
        print(f"Erreur lors de la promotion du modèle en stage en production (Etape 4/5) : {e}")
        print("-" * 150)


def check_loading_model_from_registry() -> None:
    """Vérifie que le modèle enregistré en production peut être chargé et utilisé pour faire des prédictions."""

    try:
        print(
            "Etape 5/5 : Vérification du chargement du modèle depuis le registry de production et prédictions sur le test set..."
        )
        model = mlflow.sklearn.load_model(f"models:/{REGISTERED_MODEL_NAME}/Production")

        # Chargement données
        df = load_data(IN_CSV)
        X, y = preprocess(df)

        # Split données
        _, X_test, _, y_test = split_data(X, y)

        # Prédictions
        preds = model.predict(X_test)

        print(f"- Prédictions sur le test set : {preds[:5]}")
        print(
            "\n[ok] : Vérification du chargement du modèle depuis le registry de production et prédictions sur le test set est terminé avec succès (Etape 5/5)."
        )
        print("-" * 150)

    except Exception as e:
        print(f"Erreur lors de la vérification du chargement du modèle (Etape 5/5) : {e}")
        print("-" * 150)


def main() -> None:
    console_msg_intro()
    train_and_log_models()
    save_best_model_in_registry()
    promote_best_model_to_staging()
    promote_best_model_to_production()
    check_loading_model_from_registry()


if __name__ == "__main__":
    main()
