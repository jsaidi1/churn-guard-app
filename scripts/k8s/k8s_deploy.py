"""
Script pour le déploiement k8s

Avant de lancer :

    - cluster k3d démarré
    - Docker Desktop lancé
    - manifests présents dans k8s/ :
        namespace.yaml
        pvc.yaml
        mlflow.yaml
        trainer-job.yaml
        api.yaml

> uv run python .\scripts\k8s\k8s_deploy.py
"""

import subprocess
import sys


NAMESPACE = "churn-guard"


def run(command: str):
    print(f"\n>>> {command}")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print(f"\nCommand failed: {command}")
        sys.exit(result.returncode)


def main():
    # Build Docker image
    run("docker build -t churn-guard:latest .")

    # Import image into k3d
    run("k3d image import churn-guard:latest -c churnguard")

    # Apply namespace and storage
    run("kubectl apply -f k8s/namespace.yaml")
    run("kubectl apply -f k8s/pvc.yaml")

    # Deploy MLflow
    run("kubectl apply -f k8s/mlflow.yaml")

    # Wait for MLflow deployment
    run(f"kubectl rollout status deployment/mlflow -n {NAMESPACE} --timeout=300s")

    # Delete old trainer job if exists
    run(f"kubectl delete job trainer -n {NAMESPACE} --ignore-not-found")

    # Launch trainer job
    run("kubectl apply -f k8s/trainer-job.yaml")

    # Wait for trainer completion
    run(f"kubectl wait --for=condition=complete job/trainer -n {NAMESPACE} --timeout=600s")

    # Deploy API
    run("kubectl apply -f k8s/api.yaml")

    # Wait for API deployment
    run(f"kubectl rollout status deployment/churnguard-api -n {NAMESPACE} --timeout=300s")

    # Display resources
    run(f"kubectl get all -n {NAMESPACE}")

    print("\nDeployment completed successfully.")
    print("API:    http://localhost:8000")
    print("MLflow: http://localhost:5050")


if __name__ == "__main__":
    main()
