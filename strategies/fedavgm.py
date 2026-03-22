from flwr.serverapp.strategy import FedAvgM

from config import FRACTION_EVALUATE, SERVER_LEARNING_RATE, SERVER_MOMENTUM


def build_fedavgm():
    """Build Flower FedAvgM strategy from project config."""
    return FedAvgM(
        fraction_evaluate=FRACTION_EVALUATE,
        server_learning_rate=SERVER_LEARNING_RATE,
        server_momentum=SERVER_MOMENTUM,
    )
