from flwr.serverapp.strategy import FedXgbCyclic

from config import FRACTION_EVALUATE


def build_fedxgbcyclic():
    """Build Flower FedXgbCyclic strategy from project config."""
    return FedXgbCyclic(
        fraction_evaluate=FRACTION_EVALUATE,
    )
