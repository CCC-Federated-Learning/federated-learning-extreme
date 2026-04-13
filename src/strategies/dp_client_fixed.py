from flwr.serverapp.strategy import DifferentialPrivacyClientSideFixedClipping, FedAvg

from config import (
    DP_CLIPPING_NORM,
    DP_NOISE_MULTIPLIER,
    DP_NUM_SAMPLED_CLIENTS,
    FRACTION_EVALUATE,
    FRACTION_TRAIN,
)


def build_dp_client_fixed():
    """Build Flower client-side fixed clipping DP strategy from project config."""
    base_strategy = FedAvg(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
    )
    return DifferentialPrivacyClientSideFixedClipping(
        strategy=base_strategy,
        noise_multiplier=DP_NOISE_MULTIPLIER,
        clipping_norm=DP_CLIPPING_NORM,
        num_sampled_clients=DP_NUM_SAMPLED_CLIENTS,
    )
