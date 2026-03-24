from enum import Enum
from pathlib import Path

from strategy_config import (
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


class _ConfigEnum(str, Enum):
    def __str__(self) -> str:
        return self.value

class StrategyName(_ConfigEnum):
    # Basic & Classic Strategies
    FEDAVG = "FedAvg"
    FEDAVGM = "FedAvgM"
    FEDPROX = "FedProx"

    # Adaptive Strategies
    FEDADAGRAD = "FedAdagrad"
    FEDADAM = "FedAdam"
    FEDYOGI = "FedYogi"
    
    # Robust Aggregation Strategies
    BULYAN = "Bulyan"
    KRUM = "Krum"
    MULTIKRUM = "MultiKrum"
    FEDMEDIAN = "FedMedian"
    FEDTRIMMEDAVG = "FedTrimmedAvg"
    
    # Differential Privacy Strategies
    DIFFERENTIALPRIVACYCLIENTSIDEADAPTIVECLIPPING = "DifferentialPrivacyClientSideAdaptiveClipping"
    DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING = "DifferentialPrivacyClientSideFixedClipping"
    DIFFERENTIALPRIVACYSERVERSIDEADAPTIVECLIPPING = "DifferentialPrivacyServerSideAdaptiveClipping"
    DIFFERENTIALPRIVACYSERVERSIDEFIXEDCLIPPING = "DifferentialPrivacyServerSideFixedClipping"

    # XGBoost specific strategies
    FEDXGBBAGGING = "FedXgbBagging"
    FEDXGBCYCLIC = "FedXgbCyclic"

    # Fairness and other strategies
    QFEDAVG = "QFedAvg"

class DataDistribution(_ConfigEnum):
    IID = "IID"
    DIRICHLET = "Dirichlet"
    LABEL = "label"


# Global experiment settings
NUM_PARTITIONS = 10
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
NUM_ROUNDS = 200
LR = 0.001
DATASET_NAME = "MNIST"
FRACTION_TRAIN = 1.0  # train on all clients every round
FRACTION_EVALUATE = 1 # test all clients every round

#在這裡改策略
STRATEGY_NAME = StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING

# Strategy defaults from strategy_config.py
# Allow None in strategy file to fallback to global defaults here.
if DP_NUM_SAMPLED_CLIENTS is None:
    DP_NUM_SAMPLED_CLIENTS = NUM_PARTITIONS
if QFEDAVG_CLIENT_LEARNING_RATE is None:
    QFEDAVG_CLIENT_LEARNING_RATE = LR

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
    strategy_values = {item.value for item in StrategyName}
    if STRATEGY_NAME not in strategy_values:
        supported = ", ".join(item.value for item in StrategyName)
        raise ValueError(f"STRATEGY_NAME must be one of: {supported}")
    if SERVER_LEARNING_RATE <= 0:
        raise ValueError("SERVER_LEARNING_RATE must be > 0")
    if SERVER_MOMENTUM < 0:
        raise ValueError("SERVER_MOMENTUM must be >= 0")
    if PROXIMAL_MU < 0:
        raise ValueError("PROXIMAL_MU must be >= 0")
    if SERVER_ETA <= 0:
        raise ValueError("SERVER_ETA must be > 0")
    if SERVER_ETA_L <= 0:
        raise ValueError("SERVER_ETA_L must be > 0")
    if SERVER_TAU <= 0:
        raise ValueError("SERVER_TAU must be > 0")
    if not (0 < BETA_1 < 1):
        raise ValueError("BETA_1 must be in (0, 1)")
    if not (0 < BETA_2 < 1):
        raise ValueError("BETA_2 must be in (0, 1)")
    if not (0 <= FEDTRIMMEDAVG_BETA < 0.5):
        raise ValueError("FEDTRIMMEDAVG_BETA must be in [0, 0.5)")
    if BULYAN_NUM_MALICIOUS_NODES < 0:
        raise ValueError("BULYAN_NUM_MALICIOUS_NODES must be >= 0")
    if KRUM_NUM_MALICIOUS_NODES < 0:
        raise ValueError("KRUM_NUM_MALICIOUS_NODES must be >= 0")
    if MULTIKRUM_NUM_MALICIOUS_NODES < 0:
        raise ValueError("MULTIKRUM_NUM_MALICIOUS_NODES must be >= 0")
    if MULTIKRUM_NUM_NODES_TO_SELECT <= 0:
        raise ValueError("MULTIKRUM_NUM_NODES_TO_SELECT must be > 0")
    if DP_NOISE_MULTIPLIER < 0:
        raise ValueError("DP_NOISE_MULTIPLIER must be >= 0")
    if DP_CLIPPING_NORM <= 0:
        raise ValueError("DP_CLIPPING_NORM must be > 0")
    if DP_NUM_SAMPLED_CLIENTS <= 0:
        raise ValueError("DP_NUM_SAMPLED_CLIENTS must be > 0")
    if DP_INITIAL_CLIPPING_NORM <= 0:
        raise ValueError("DP_INITIAL_CLIPPING_NORM must be > 0")
    if not (0 <= DP_TARGET_CLIPPED_QUANTILE <= 1):
        raise ValueError("DP_TARGET_CLIPPED_QUANTILE must be in [0, 1]")
    if DP_CLIP_NORM_LR <= 0:
        raise ValueError("DP_CLIP_NORM_LR must be > 0")
    if DP_CLIPPED_COUNT_STDDEV is not None and DP_CLIPPED_COUNT_STDDEV < 0:
        raise ValueError("DP_CLIPPED_COUNT_STDDEV must be >= 0 when provided")
    if QFEDAVG_CLIENT_LEARNING_RATE <= 0:
        raise ValueError("QFEDAVG_CLIENT_LEARNING_RATE must be > 0")
    if QFEDAVG_Q < 0:
        raise ValueError("QFEDAVG_Q must be >= 0")
    if XGB_NUM_LOCAL_ROUND <= 0:
        raise ValueError("XGB_NUM_LOCAL_ROUND must be > 0")
    if XGB_MAX_DEPTH <= 0:
        raise ValueError("XGB_MAX_DEPTH must be > 0")
    if XGB_ETA <= 0:
        raise ValueError("XGB_ETA must be > 0")
    if not (0 < XGB_SUBSAMPLE <= 1):
        raise ValueError("XGB_SUBSAMPLE must be in (0, 1]")
    if not (0 < XGB_COLSAMPLE_BYTREE <= 1):
        raise ValueError("XGB_COLSAMPLE_BYTREE must be in (0, 1]")
    if XGB_MIN_CHILD_WEIGHT < 0:
        raise ValueError("XGB_MIN_CHILD_WEIGHT must be >= 0")
    if XGB_REG_LAMBDA < 0:
        raise ValueError("XGB_REG_LAMBDA must be >= 0")
    if XGB_NUM_CLASS <= 1:
        raise ValueError("XGB_NUM_CLASS must be > 1")
    # Validate DATA_DISTRIBUTION is a valid enum value
    if not isinstance(DATA_DISTRIBUTION, DataDistribution):
        supported = ", ".join(item.value for item in DataDistribution)
        raise ValueError(f"DATA_DISTRIBUTION must be one of: {supported}")
    if DIRICHLET_ALPHA <= 0:
        raise ValueError("DIRICHLET_ALPHA must be > 0")
    if NUM_PARTITIONS <= 0:
        raise ValueError("NUM_PARTITIONS must be > 0")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0")
    if not (0 < FRACTION_TRAIN <= 1):
        raise ValueError("FRACTION_TRAIN must be in (0, 1]")
    if not (0 <= FRACTION_EVALUATE <= 1):
        raise ValueError("FRACTION_EVALUATE must be in [0, 1]")
