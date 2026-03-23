from flwr.serverapp.strategy import FedXgbCyclic

from config import FRACTION_EVALUATE, FRACTION_TRAIN


def build_fedxgbcyclic():
    """Build Flower FedXgbCyclic strategy from project config."""
    return FedXgbCyclic(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
    )
