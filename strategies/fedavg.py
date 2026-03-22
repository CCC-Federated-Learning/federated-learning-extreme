from flwr.serverapp.strategy import FedAvg

from config import FRACTION_EVALUATE


def build_fedavg():
    """Build Flower FedAvg strategy from project config."""
    return FedAvg(fraction_evaluate=FRACTION_EVALUATE)
