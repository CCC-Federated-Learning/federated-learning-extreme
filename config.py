from pathlib import Path
import sys
from config_types import DataDistribution, DatasetName, StrategyName

from config_strategy import (
    BETA_1,
    BETA_2,
    BULYAN_NUM_MALICIOUS_NODES,
    DP_CLIPPED_COUNT_STDDEV,
    DP_CLIP_NORM_LR,
    DP_CLIPPING_NORM,
    DP_INITIAL_CLIPPING_NORM,
    DP_NOISE_MULTIPLIER,
    DP_NUM_SAMPLED_CLIENTS,
    DP_TARGET_CLIPPED_QUANTILE,
    FEDTRIMMEDAVG_BETA,
    KRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_NODES_TO_SELECT,
    PROXIMAL_MU,
    QFEDAVG_CLIENT_LEARNING_RATE,
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

# Global experiment settings
NUM_PARTITIONS = 10
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
NUM_ROUNDS = 200
LR = 0.001
DATASET_NAME = DatasetName.MNIST
FRACTION_TRAIN = 1.0  # train on all clients every round
FRACTION_EVALUATE = 1 # test all clients every round

#在這裡改策略
STRATEGY_NAME = StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING

def _resolve_strategy_defaults() -> tuple[int, float]:
    """Resolve strategy defaults from global settings when strategy config uses None."""
    dp_num_sampled_clients = DP_NUM_SAMPLED_CLIENTS
    qfedavg_client_lr = QFEDAVG_CLIENT_LEARNING_RATE

    if dp_num_sampled_clients is None:
        dp_num_sampled_clients = NUM_PARTITIONS
    if qfedavg_client_lr is None:
        qfedavg_client_lr = LR

    return int(dp_num_sampled_clients), float(qfedavg_client_lr)


DP_NUM_SAMPLED_CLIENTS, QFEDAVG_CLIENT_LEARNING_RATE = _resolve_strategy_defaults()

# Data partition settings
DATA_DISTRIBUTION = DataDistribution.LABEL 
# Data distribution options:
DIRICHLET_ALPHA = 0.5     # Dirichlet parameter (small = Non-IID)
DATA_SEED = 499 

# Runtime settings
CLIENT_NUM_CPUS = 2
CLIENT_NUM_GPUS_IF_AVAILABLE = 1.0

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
    from config_validation import validate_config_module

    validate_config_module(sys.modules[__name__])
