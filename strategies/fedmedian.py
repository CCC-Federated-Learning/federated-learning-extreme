from flwr.serverapp.strategy import FedMedian

from config import FRACTION_EVALUATE


def build_fedmedian():
    """Build Flower FedMedian strategy from project config."""
    return FedMedian(
        fraction_evaluate=FRACTION_EVALUATE,
    )
