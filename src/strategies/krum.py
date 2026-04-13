from flwr.serverapp.strategy import Krum

from config import FRACTION_EVALUATE, FRACTION_TRAIN, KRUM_NUM_MALICIOUS_NODES


def build_krum():
    """Build Flower Krum strategy from project config."""
    return Krum(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
        num_malicious_nodes=KRUM_NUM_MALICIOUS_NODES,
    )
