import os
import sys
from pathlib import Path

import flwr
import torch

from config import CLIENT_NUM_CPUS, CLIENT_NUM_GPUS_IF_AVAILABLE, NUM_PARTITIONS, validate_config
from server import server_app
from client import client_app


def _ensure_project_import_path() -> None:
    """Ensure local modules can be imported even when launched outside this folder."""
    project_dir = Path(__file__).resolve().parent
    project_dir_str = str(project_dir)

    os.chdir(project_dir)

    if project_dir_str not in sys.path:
        sys.path.insert(0, project_dir_str)

    current_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [p for p in current_pythonpath.split(os.pathsep) if p]
    if project_dir_str not in pythonpath_parts:
        os.environ["PYTHONPATH"] = (
            project_dir_str
            if not current_pythonpath
            else f"{project_dir_str}{os.pathsep}{current_pythonpath}"
        )


def run_simulation() -> None:
    _ensure_project_import_path()
    validate_config()
    backend_config = {
        "client_resources": {
            "num_cpus": CLIENT_NUM_CPUS,
            "num_gpus": CLIENT_NUM_GPUS_IF_AVAILABLE if torch.cuda.is_available() else 0.0,
        }
    }

    flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    run_simulation()