"""
- Nom du script : data.py
- But : chargement, préprocessing, split
- Pour executer ce script directement : python -m churnguard.data
"""

from typing import cast

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Charge les données depuis le fichier CSV d'entrée."""

    return pd.read_csv(path)


def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Effectue le préprocessing des données et retourne X et y."""

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna()
    df = df.drop(columns=["customerID"])

    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    return X, y


def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split les données en train et test. Retourne X_train, X_test, y_train, y_test."""

    result = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return cast(
        tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],
        result,
    )
