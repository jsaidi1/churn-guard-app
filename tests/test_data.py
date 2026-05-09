"""
Pour exécuter les tests de ce module (avec un rapport de coverage), utilisez la commande suivante dans votre terminal à la racine du projet :
> uv run pytest --cov=churnguard --cov-report=term-missing
"""

import pandas as pd

from churnguard.data import load_data, preprocess, split_data


def test_load_data_returns_dataframe(tmp_path):
    """Test que la fonction load_data retourne un DataFrame pandas valide : vérifie la forme du DataFrame chargé"""

    # Création d’un CSV minimal simulé
    csv_content = "col1,col2\n1,2\n3,4"
    file_path = tmp_path / "test.csv"
    file_path.write_text(csv_content)

    df = load_data(str(file_path))

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_data_has_expected_columns():
    """Vérifie la présence des 21 colonnes attendues dans le DataFrame chargé"""

    data_path = "data/telco_churn.csv"

    df = load_data(data_path)

    expected_columns = {
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    }

    # Vérifie présence des colonnes (ordre non important)
    assert set(df.columns) == expected_columns


def test_preprocess_returns_features_and_target() -> None:
    """Preprocess doit séparer correctement les features X et la cible y."""

    df = pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 12],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": ["500.20", "683.40"],
            "Churn": ["No", "Yes"],
        }
    )

    X, y = preprocess(df)

    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)

    assert "Churn" not in X.columns
    assert "customerID" not in X.columns

    assert X.shape[0] == 2
    assert y.shape[0] == 2

    assert list(y) == [0, 1]  # Encodage correct de la target


def test_preprocess_handles_missing_total_charges() -> None:
    """preprocess doit convertir TotalCharges et gérer les valeurs vides."""

    df = pd.DataFrame(
        {
            "customerID": ["0001", "0002"],
            "gender": ["Female", "Male"],
            "SeniorCitizen": [0, 1],
            "Partner": ["Yes", "No"],
            "Dependents": ["No", "Yes"],
            "tenure": [1, 12],
            "PhoneService": ["No", "Yes"],
            "MultipleLines": ["No phone service", "No"],
            "InternetService": ["DSL", "Fiber optic"],
            "OnlineSecurity": ["No", "Yes"],
            "OnlineBackup": ["Yes", "No"],
            "DeviceProtection": ["No", "Yes"],
            "TechSupport": ["No", "No"],
            "StreamingTV": ["No", "Yes"],
            "StreamingMovies": ["No", "Yes"],
            "Contract": ["Month-to-month", "One year"],
            "PaperlessBilling": ["Yes", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check"],
            "MonthlyCharges": [29.85, 56.95],
            "TotalCharges": [
                " ",
                "683.40",
            ],  # cas problématique => doit supprimer la ligne 1 (util pour le test test_preprocess_handles_missing_total_charges)
            "Churn": ["No", "Yes"],
        }
    )

    X, y = preprocess(df)

    # Vérifie que la colonne est bien numérique
    assert X["TotalCharges"].dtype in ["float64", "float32"]

    # Vérifie qu'il n'y a plus de valeur vide
    assert not X["TotalCharges"].isna().all()

    # Vérifie que le nombre de lignes a diminué de 1 (la ligne avec TotalCharges vide a été supprimée)
    assert X.shape[0] == 1
    assert y.shape[0] == 1


def test_split_data_returns_correct_shapes_and_stratification() -> None:
    """split_data doit respecter le ratio, conserver l'alignement et stratifier y."""

    # Dataset simulé avec déséquilibre
    X = pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": range(100, 200),
        }
    )

    # 80% de 0, 20% de 1
    y = pd.Series([0] * 80 + [1] * 20)

    X_train, X_test, y_train, y_test = split_data(X, y)

    # 1) Vérification du ratio (80/20)
    assert X_train.shape[0] == 80
    assert X_test.shape[0] == 20

    # 2) Alignement X / y
    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]

    # 3) Stratification (proportion conservée)
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()

    assert abs(train_ratio - test_ratio) < 0.05

    # 4) Reproductibilité
    X_train2, _, _, y_test2 = split_data(X, y)

    assert X_train.equals(X_train2)
    assert y_test.equals(y_test2)


def test_fail():
    assert False