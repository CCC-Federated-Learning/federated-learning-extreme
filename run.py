import os

import flwr
import torch

from config import (
    CLIENT_NUM_CPUS,
    CLIENT_NUM_GPUS_IF_AVAILABLE,
    DATA_SEED,
    NUM_PARTITIONS,
    validate_config,
)
from draw_distribution import plot_client_distribution
from server import recorder, server_app
from client import client_app


def run_simulation() -> None:
    # Silence Ray warning about accelerator env var behavior when num_gpus is zero.
    os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")

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

    if recorder.experiment_dir is not None:
        plot_client_distribution(
            save_dir=str(recorder.experiment_dir),
            save_name="distribution_chart.png",
            show_plot=False,
            seed=DATA_SEED,
            add_timestamp=False,
        )
        print(f"Saved distribution chart to: {recorder.experiment_dir.resolve()}")


if __name__ == "__main__":
    run_simulation()