# ChurnGuard — projet de départ

> **Mission** : industrialiser ce projet en 2 jours selon le cahier des charges fourni
> (`Sujet_ChurnGuard_MLOps.docx`). Vous ne touchez pas à la data science.

## Contexte

Vous reprenez le projet d'une data scientist de TelcoFr. Elle a entraîné un
modèle de prédiction de churn dans un notebook qui marche. Personne d'autre que
elle ne sait le faire tourner.

Votre rôle : transformer ce repo en projet MLOps de production.

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

## Ce qu'il y a dans le repo

```
.
├── README.md                       # ce fichier
├── requirements.txt                # dépendances minimales (pandas, sklearn, jupyter)
├── notebook/
│   └── exploration.ipynb           # notebook qui marche, mais qui pue
├── scripts/
│   └── download_data.py            # téléchargement + vérification SHA-256
├── data/
│   └── .gitkeep                    # le CSV est téléchargé ici
└── .gitignore
```

## Démarrage rapide (état initial)

```bash
# 1. installer les dépendances
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. télécharger les données
python scripts/download_data.py

# 3. ouvrir le notebook
jupyter notebook notebook/exploration.ipynb
```

À ce stade le notebook s'exécute de bout en bout et produit un fichier
`models/best_model.pkl`. C'est tout. Pas de tests, pas de Docker, pas d'API,
pas de CI.

## Ce que vous devez faire

Lire le sujet (`Sujet_ChurnGuard_MLOps.docx`), puis livrer :

- un package Python `churnguard/` modulaire avec tests pytest,
- un setup MLflow (tracking + registry) avec un modèle promu en `Production`,
- une API FastAPI qui charge le modèle depuis le registry,
- un Dockerfile multi-stage et un docker-compose qui démarre toute la stack,
- un workflow GitHub Actions complet (lint, typecheck, tests, build, scan).


## Licence

Code : MIT.
Données : IBM Sample Data, voir conditions sur le site IBM.



### Comparaison MLflow (ajoutéééééééééé)
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

1) Meilleur global (équilibre)

➡️ Logistic Regression

meilleur accuracy
meilleur f1
meilleur precision
meilleur recall
roc_auc très proche du meilleur

👉 C’est le modèle le plus équilibré et robuste


Badge README
![CI](https://github.com/<USER>/<REPO>/actions/workflows/ci.yml/badge.svg)