import flwr
import torch

from config import CLIENT_NUM_CPUS, CLIENT_NUM_GPUS_IF_AVAILABLE, NUM_PARTITIONS, validate_config
from server import server_app
from client import client_app


def run_simulation() -> None:
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