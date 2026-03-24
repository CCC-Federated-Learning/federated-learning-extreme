from flwr.serverapp.strategy import FedTrimmedAvg

from config import FEDTRIMMEDAVG_BETA, FRACTION_EVALUATE, FRACTION_TRAIN


def build_fedtrimmedavg():
    """Build Flower FedTrimmedAvg strategy from project config."""
    return FedTrimmedAvg(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
        beta=FEDTRIMMEDAVG_BETA,
    )
