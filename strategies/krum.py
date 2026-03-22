from flwr.serverapp.strategy import Krum

from config import FRACTION_EVALUATE, KRUM_NUM_MALICIOUS_NODES


def build_krum():
    """Build Flower Krum strategy from project config."""
    return Krum(
        fraction_evaluate=FRACTION_EVALUATE,
        num_malicious_nodes=KRUM_NUM_MALICIOUS_NODES,
    )
