from pathlib import Path
import sys
from .types import DataDistribution, DatasetName, StrategyName

from .strategy import (
    BETA_1,
    BETA_2,
    BULYAN_NUM_MALICIOUS_NODES,
    DP_CLIPPED_COUNT_STDDEV,
    DP_CLIP_NORM_LR,
    DP_CLIPPING_NORM,
    DP_INITIAL_CLIPPING_NORM,
    DP_NOISE_MULTIPLIER,
    DP_NUM_SAMPLED_CLIENTS as STRATEGY_DP_NUM_SAMPLED_CLIENTS,
    DP_TARGET_CLIPPED_QUANTILE,
    FEDTRIMMEDAVG_BETA,
    KRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_NODES_TO_SELECT,
    PROXIMAL_MU,
    QFEDAVG_CLIENT_LEARNING_RATE as STRATEGY_QFEDAVG_CLIENT_LEARNING_RATE,
    QFEDAVG_Q,
    SERVER_ETA,
    SERVER_ETA_L,
    SERVER_LEARNING_RATE,
    SERVER_MOMENTUM,
    SERVER_TAU,
    XGB_COLSAMPLE_BYTREE,
    XGB_ETA,
    XGB_MAX_DEPTH,
    XGB_MIN_CHILD_WEIGHT,
    XGB_NUM_LOCAL_ROUND,
    XGB_REG_LAMBDA,
    XGB_SUBSAMPLE,
)

# Commonly adjusted experiment settings
# Quick guide:
# 1) Switch algorithm: change STRATEGY_NAME.
# 2) Compare IID vs Non-IID: change DATA_DISTRIBUTION (and DIRICHLET_ALPHA if needed).
# 3) Speed up debug runs: lower NUM_ROUNDS / LOCAL_EPOCHS / NUM_PARTITIONS.
# 4) Improve stability: increase BATCH_SIZE or reduce LR.
# 5) Full participation baseline: keep FRACTION_TRAIN/FRACTION_EVALUATE at 1.0.
# 6) Client sampling studies: lower FRACTION_TRAIN from 1.0.
# 7) Reproducibility: keep DATA_SEED fixed; change it only for new random splits.
# 在這裡改策略
STRATEGY_NAME = StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING
DATA_DISTRIBUTION = DataDistribution.LABEL
DATASET_NAME = DatasetName.MNIST

NUM_ROUNDS = 200
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LR = 0.001
NUM_PARTITIONS = 10

FRACTION_TRAIN = 1.0  # train on all clients every round
FRACTION_EVALUATE = 1  # test all clients every round

# Data split controls
DIRICHLET_ALPHA = 0.5  # Dirichlet parameter (small = Non-IID)
DATA_SEED = 499

def _resolve_strategy_defaults() -> tuple[int, float]:
    """Resolve strategy defaults from global settings when strategy config uses None."""
    dp_num_sampled_clients = STRATEGY_DP_NUM_SAMPLED_CLIENTS
    qfedavg_client_lr = STRATEGY_QFEDAVG_CLIENT_LEARNING_RATE

    if dp_num_sampled_clients is None:
        dp_num_sampled_clients = NUM_PARTITIONS
    if qfedavg_client_lr is None:
        qfedavg_client_lr = LR

    return int(dp_num_sampled_clients), float(qfedavg_client_lr)


DP_NUM_SAMPLED_CLIENTS, QFEDAVG_CLIENT_LEARNING_RATE = _resolve_strategy_defaults()

# Runtime settings
CLIENT_NUM_CPUS = 1
CLIENT_NUM_GPUS_IF_AVAILABLE = 0.1

# XGBoost settings
XGB_OBJECTIVE = "multi:softprob"
XGB_NUM_CLASS = 10

# Draw settings
DRAW_SAVE_DIR = "distribution_chart"
DRAW_SAVE_NAME = "client_distribution.png"
DRAW_SHOW_PLOT = True
DRAW_FIGSIZE = (8.28, 4.49)

# Run record settings
RES_DIR = "res"

# File naming settings
TIMESTAMP_FORMAT = "%Y%m%d-%H%M-%S"


def get_draw_output_path() -> Path:
    return Path(DRAW_SAVE_DIR) / DRAW_SAVE_NAME


def validate_config() -> None:
    from .validation import validate_config_module

    validate_config_module(sys.modules[__name__])
