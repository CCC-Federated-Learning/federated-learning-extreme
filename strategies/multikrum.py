from flwr.serverapp.strategy import MultiKrum

from config import (
    FRACTION_EVALUATE,
    MULTIKRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_NODES_TO_SELECT,
)


def build_multikrum():
    """Build Flower MultiKrum strategy from project config."""
    return MultiKrum(
        fraction_evaluate=FRACTION_EVALUATE,
        num_malicious_nodes=MULTIKRUM_NUM_MALICIOUS_NODES,
        num_nodes_to_select=MULTIKRUM_NUM_NODES_TO_SELECT,
    )
