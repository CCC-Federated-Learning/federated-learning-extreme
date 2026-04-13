from flwr.serverapp.strategy import QFedAvg

from config import FRACTION_EVALUATE, FRACTION_TRAIN, QFEDAVG_CLIENT_LEARNING_RATE, QFEDAVG_Q


def build_qfedavg():
    """Build Flower QFedAvg strategy from project config."""
    return QFedAvg(
        client_learning_rate=QFEDAVG_CLIENT_LEARNING_RATE,
        q=QFEDAVG_Q,
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
    )
