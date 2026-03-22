from flwr.serverapp.strategy import FedTrimmedAvg

from config import FEDTRIMMEDAVG_BETA, FRACTION_EVALUATE


def build_fedtrimmedavg():
    """Build Flower FedTrimmedAvg strategy from project config."""
    return FedTrimmedAvg(
        fraction_evaluate=FRACTION_EVALUATE,
        beta=FEDTRIMMEDAVG_BETA,
    )
