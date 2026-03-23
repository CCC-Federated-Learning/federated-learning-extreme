from flwr.serverapp.strategy import FedAvg

from config import FRACTION_EVALUATE, FRACTION_TRAIN


def build_fedavg():
    """Build Flower FedAvg strategy from project config."""
    return FedAvg(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
    )
