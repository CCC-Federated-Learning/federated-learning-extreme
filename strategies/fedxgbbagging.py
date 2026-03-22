from flwr.serverapp.strategy import FedXgbBagging

from config import FRACTION_EVALUATE


def build_fedxgbbagging():
    """Build Flower FedXgbBagging strategy from project config."""
    return FedXgbBagging(
        fraction_evaluate=FRACTION_EVALUATE,
    )
