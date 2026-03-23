from flwr.serverapp.strategy import FedMedian

from config import FRACTION_EVALUATE, FRACTION_TRAIN


def build_fedmedian():
    """Build Flower FedMedian strategy from project config."""
    return FedMedian(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
    )
