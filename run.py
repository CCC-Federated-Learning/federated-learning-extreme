import flwr
import torch

from server import server_app
from client import client_app


backend_config = {
    "client_resources": {
        "num_cpus": 2,
        "num_gpus": 1 if torch.cuda.is_available() else 0.0
    }
}
flwr.simulation.run_simulation(
    server_app=server_app,
    client_app=client_app,
    num_supernodes=10,
    backend_config=backend_config
)