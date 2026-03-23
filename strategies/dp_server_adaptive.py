from flwr.serverapp.strategy import (
    DifferentialPrivacyServerSideAdaptiveClipping,
    FedAvg,
)

from config import (
    DP_CLIP_NORM_LR,
    DP_CLIPPED_COUNT_STDDEV,
    DP_INITIAL_CLIPPING_NORM,
    DP_NOISE_MULTIPLIER,
    DP_NUM_SAMPLED_CLIENTS,
    DP_TARGET_CLIPPED_QUANTILE,
    FRACTION_EVALUATE,
    FRACTION_TRAIN,
)


def build_dp_server_adaptive():
    """Build Flower server-side adaptive clipping DP strategy from project config."""
    base_strategy = FedAvg(
        fraction_train=FRACTION_TRAIN,
        fraction_evaluate=FRACTION_EVALUATE,
    )
    return DifferentialPrivacyServerSideAdaptiveClipping(
        strategy=base_strategy,
        noise_multiplier=DP_NOISE_MULTIPLIER,
        num_sampled_clients=DP_NUM_SAMPLED_CLIENTS,
        initial_clipping_norm=DP_INITIAL_CLIPPING_NORM,
        target_clipped_quantile=DP_TARGET_CLIPPED_QUANTILE,
        clip_norm_lr=DP_CLIP_NORM_LR,
        clipped_count_stddev=DP_CLIPPED_COUNT_STDDEV,
    )
