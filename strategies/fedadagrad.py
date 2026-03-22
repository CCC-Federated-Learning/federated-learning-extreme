from flwr.serverapp.strategy import FedAdagrad

from config import FRACTION_EVALUATE, SERVER_ETA, SERVER_ETA_L, SERVER_TAU


def build_fedadagrad():
    """Build Flower FedAdagrad strategy from project config."""
    return FedAdagrad(
        fraction_evaluate=FRACTION_EVALUATE,
        eta=SERVER_ETA,
        eta_l=SERVER_ETA_L,
        tau=SERVER_TAU,
    )
