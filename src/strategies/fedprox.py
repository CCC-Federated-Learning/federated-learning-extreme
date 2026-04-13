from flwr.serverapp.strategy import FedProx

from config import FRACTION_EVALUATE, FRACTION_TRAIN, PROXIMAL_MU


def build_fedprox():
    """Build Flower FedProx strategy from project config."""
    return FedProx(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
        proximal_mu=PROXIMAL_MU,
    )
