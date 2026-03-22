from flwr.serverapp.strategy import QFedAvg

from config import FRACTION_EVALUATE, QFEDAVG_CLIENT_LEARNING_RATE, QFEDAVG_Q


def build_qfedavg():
    """Build Flower QFedAvg strategy from project config."""
    return QFedAvg(
        client_learning_rate=QFEDAVG_CLIENT_LEARNING_RATE,
        q=QFEDAVG_Q,
        fraction_evaluate=FRACTION_EVALUATE,
    )
