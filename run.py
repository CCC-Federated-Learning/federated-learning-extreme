import os
import sys
from pathlib import Path

import flwr
import torch

from config import (
    CLIENT_NUM_CPUS,
    CLIENT_NUM_GPUS_IF_AVAILABLE,
    DATA_SEED,
    NUM_PARTITIONS,
    STRATEGY_NAME,
    StrategyName,
    validate_config,
)
from draw_distribution import plot_client_distribution
from app.server import recorder, server_app
from strategies.factory import build_strategy


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


def _finalize_experiment() -> None:
    """Generate and save post-simulation artifacts."""
    if recorder.experiment_dir is not None:
        plot_client_distribution(
            save_dir=str(recorder.experiment_dir),
            save_name="distribution_chart.png",
            show_plot=False,
            seed=DATA_SEED,
            add_timestamp=False,
        )
        print(f"Saved distribution chart to: {recorder.experiment_dir.resolve()}")


def run_simulation() -> None:
    """Main simulation orchestration: validates config → builds strategy → runs simulation."""
    # Silence Ray warning about accelerator env var behavior when num_gpus is zero.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

    # === INITIALIZATION ===
    validate_config()
    # Build once in main process to surface compatibility issues early
    # (instead of failing later inside the ServerApp background thread).
    build_strategy()

    # === EXECUTION ===
    client_app = _get_client_app()
    backend_config = _build_backend_config()

    flwr.simulation.run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    )

    # === FINALIZATION ===
    _finalize_experiment()


if __name__ == "__main__":
    run_simulation()
