"""
> uv run python scripts/k8s/k8s_reset_k3d_cluster.py
"""

import subprocess
import sys


CLUSTER_NAME = "churnguard"
K8S_CONTEXT = "k3d-churnguard"
API_SERVER = "https://127.0.0.1:6550"


def run(command: str, allow_fail: bool = False) -> None:
    print(f"\n>>> {command}")
    result = subprocess.run(command, shell=True)

    if result.returncode != 0 and not allow_fail:
        print(f"\nCommand failed: {command}")
        sys.exit(result.returncode)


def main() -> None:
    print("Reset k3d cluster...")

    # Delete existing cluster if present
    run(f"k3d cluster delete {CLUSTER_NAME}", allow_fail=True)

    # Create fresh cluster
    run(
        f"k3d cluster create {CLUSTER_NAME} "
        f"--agents 1 "
        f'--api-port "127.0.0.1:6550" '
        f'-p "8000:30080@agent:0" '
        f'-p "5050:30050@agent:0"'
    )

    # Merge kubeconfig and switch context
    run(f"k3d kubeconfig merge {CLUSTER_NAME} --kubeconfig-switch-context")

    # Force kubectl to use localhost instead of host.docker.internal
    run(f"kubectl config set-cluster {K8S_CONTEXT} --server={API_SERVER}")

    # Verify nodes
    run("kubectl get nodes")

    print("\nCluster reset completed successfully.")


if __name__ == "__main__":
    main()
