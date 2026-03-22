from flwr.serverapp.strategy import DifferentialPrivacyServerSideFixedClipping, FedAvg

from config import (
    DP_CLIPPING_NORM,
    DP_NOISE_MULTIPLIER,
    DP_NUM_SAMPLED_CLIENTS,
    FRACTION_EVALUATE,
)


def build_dp_server_fixed():
    """Build Flower server-side fixed clipping DP strategy from project config."""
    base_strategy = FedAvg(fraction_evaluate=FRACTION_EVALUATE)
    return DifferentialPrivacyServerSideFixedClipping(
        strategy=base_strategy,
        noise_multiplier=DP_NOISE_MULTIPLIER,
        clipping_norm=DP_CLIPPING_NORM,
        num_sampled_clients=DP_NUM_SAMPLED_CLIENTS,
    )
