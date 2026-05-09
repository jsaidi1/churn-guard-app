# ChurnGuard MLOps

[![CI](https://github.com/jsaidi1/churn-guard-app/actions/workflows/ci.yml/badge.svg)](https://github.com/jsaidi1/churn-guard-app/actions/workflows/ci.yml)
[![Release](https://github.com/jsaidi1/churn-guard-app/actions/workflows/release.yml/badge.svg)](https://github.com/jsaidi1/churn-guard-app/actions/workflows/release.yml)

Industrialisation MLOps d’un modèle de prédiction de churn client avec MLflow, FastAPI, Docker et GitHub Actions.

## Objectif

Le projet transforme un notebook de data science en application MLOps industrialisée :

- code Python modulaire
- tests automatisés avec coverage
- tracking et registry MLflow
- API FastAPI de prédiction
- stack Docker Compose
- CI avec lint, typage, tests, build Docker et scan Trivy
- CD sur tag avec publication GHCR

## Données

**Telco Customer Churn** (IBM Sample Data, ~960 Ko, 7 043 lignes, 21 colonnes,
licence libre à des fins éducatives).

Le fichier n'est pas commité dans le repo. Pour le télécharger :

```bash
python scripts/download_data.py
```

Le script télécharge le CSV depuis un mirror stable et vérifie son intégrité par
SHA-256.

Sources :
- [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- [IBM Sample Data Sets](https://www.ibm.com/community/blogs/datasets/)


## Architecture

```text
                  ┌─────────────────────┐
                  │   Dataset Telco     │
                  └─────────┬───────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │ churnguard.train    │
                  │ training + metrics  │
                  └─────────┬───────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │       MLflow        │
                  │ tracking + registry │
                  └─────────┬───────────┘
                            │
                            ▼
                  ┌─────────────────────┐
                  │   Model Production  │
                  └─────────┬───────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │     FastAPI API      │
                  │ /health /predict ... │
                  └─────────┬────────────┘
                            │
                            ▼
                  ┌───────────────────────┐
                  │ Docker / GHCR / CI-CD │
                  └───────────────────────┘
```

## Structure du projet

    .
    ├── api/
    │   ├── main.py
    │   ├── model_loader.py
    │   └── schemas.py
    ├── churnguard/
    │   ├── data.py
    │   ├── evaluate.py
    │   └── train.py
    ├── scripts/
    │   └── download_data.py
    ├── tests/
    │   └── test_data.py
    │   └── test_train.py
    ├── .github/workflows/
    │   ├── ci.yml
    │   └── release.yml
    ├── Dockerfile
    ├── docker-compose.yml
    ├── pyproject.toml
    ├── README.md
    └── uv.lock

## Stack technique

- Python 3.11
- FastAPI
- MLflow 3
- scikit-learn
- pandas
- Docker
- Docker Compose
- GitHub Actions
- Ruff
- mypy
- pytest
- Trivy
- uv

## Résultats de l'entrainement des midèles sur l'UI MLflow (en local)

| Modèle              | accuracy  | f1        | precision | recall    | roc_auc   |
| ------------------- | --------- | --------- | --------- | --------- | --------- |
| Gradient Boosting   | 0.796     | 0.58      | 0.641     | 0.529     | **0.839** |
| Random Forest       | 0.778     | 0.53      | 0.607     | 0.471     | 0.812     |
| Logistic Regression | **0.804** | **0.608** | **0.648** | **0.572** | 0.836     |

Trois modèles ont été entraînés et loggés dans MLflow :

- LogisticRegression
- RandomForestClassifier
- GradientBoostingClassifier

Les runs ont été comparés dans l’interface MLflow sur les métriques suivantes :
accuracy, precision, recall, f1 et roc_auc.

Le modèle retenu est celui présentant le meilleur compromis entre `roc_auc`, `recall` et `f1` :

**Logistic Regression**

- meilleur accuracy
- meilleur f1
- meilleur precision
- meilleur recall
- roc_auc très proche du meilleur

=> C’est le modèle le plus équilibré et robuste

## Quickstart

### 1. Cloner le dépôt

    git clone https://github.com/jsaidi1/churn-guard-app.git

### 2. Lancer la stack complète
    docker compose up -d

Cette commande démarre :

- MLflow sur http://localhost:5000
- le trainer qui entraîne, log et promeut le modèle
- l’API FastAPI sur http://localhost:8000

### 3. Tester l’API
    curl http://localhost:8000/health

Exemple de réponse attendu :

    {
        "status": "ok",
        "model": "churnguard",
        "version": "1"
    }

Exemple d’appel /predict :

    curl -X POST "http://localhost:8000/predict" \
    -H "Content-Type: application/json" \
    -d '{
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "Yes",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 65.5,
        "TotalCharges": 780.0
    }'

Exemple de réponse attendu :

    {
        "churn": false,
        "probability": 0.23
    }

## Endpoints API
| Méthode | Endpoint         | Description                                        |
| ------- | ---------------- | -------------------------------------------------- |
| GET     | `/health`        | Vérifie que l’API et le modèle sont disponibles    |
| POST    | `/predict`       | Prédit le churn pour un client                     |
| POST    | `/predict/batch` | Prédit le churn pour une liste de clients, max 100 |

#### Swagger 
    http://localhost:8000/docs

## MLflow
- Interface MLflow :
    http://localhost:5000

- Le pipeline entraîne et compare trois modèles :

    - LogisticRegression
    - RandomForestClassifier
    - GradientBoostingClassifier

Le meilleur modèle est enregistré sous :

    churnguard

et promu en stage :

    Production

## Image Docker GHCR
Image publiée :

    ghcr.io/jsaidi1/churnguard-api:v2.0.0

Pull :

    docker pull ghcr.io/jsaidi1/churnguard:v2.0.0

## Tests
Pour lancer les tests :

    uv run pytest --cov=churnguard --cov-report=term-missing --cov-fail-under=70

## Qualité code

    uv run ruff check .
    uv run ruff format --check .
    uv run mypy churnguard api

## CI

La CI est déclenchée sur :

    push
    pull_request

Jobs :

    lint
    typecheck
    test
    build + Trivy scan

Le build Docker est lancé uniquement si lint, typecheck et tests sont verts.

## CD / Release

Le workflow CD se déclenche sur tag. Exemple :

    git tag v1.0.0
    git push origin v1.0.0

Il effectue :

- build de l’image Docker
- push vers GitHub Container Registry
- génération automatique de release notes
- création d’une GitHub Release

## Auteur

J.SAIDI (07/05/2026)