"""
- Nom du script : evaluate.py
- But : métriques, rapport classification
"""

from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def compute_metrics(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Calcule les métriques d'évaluation du modèle sur le dataset de test et retourne un dictionnaire avec les résultats."""

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics
