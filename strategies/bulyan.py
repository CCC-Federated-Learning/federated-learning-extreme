from flwr.serverapp.strategy import Bulyan

from config import BULYAN_NUM_MALICIOUS_NODES, FRACTION_EVALUATE


def build_bulyan():
    """Build Flower Bulyan strategy from project config."""
    return Bulyan(
        fraction_evaluate=FRACTION_EVALUATE,
        num_malicious_nodes=BULYAN_NUM_MALICIOUS_NODES,
    )
