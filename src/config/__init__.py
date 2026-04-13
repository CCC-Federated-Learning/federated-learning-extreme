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
    DP_NUM_SAMPLED_CLIENTS as _STRATEGY_DP_NUM_SAMPLED_CLIENTS,
    DP_TARGET_CLIPPED_QUANTILE,
    FEDTRIMMEDAVG_BETA,
    KRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_MALICIOUS_NODES,
    MULTIKRUM_NUM_NODES_TO_SELECT,
    PROXIMAL_MU,
    QFEDAVG_CLIENT_LEARNING_RATE as _STRATEGY_QFEDAVG_CLIENT_LR,
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

# ─── 在這裡改策略 ────────────────────────────────────────────────
STRATEGY_NAME = StrategyName.DIFFERENTIALPRIVACYCLIENTSIDEFIXEDCLIPPING
DATA_DISTRIBUTION = DataDistribution.IID
DATASET_NAME = DatasetName.MNIST

# ─── 訓練設定 ────────────────────────────────────────────────────
NUM_ROUNDS = 200
LOCAL_EPOCHS = 3
BATCH_SIZE = 64
LR = 0.001
NUM_CLIENTS = 10

FRACTION_TRAIN = 1.0
FRACTION_EVALUATE = 1.0

# ─── 資料分布設定 ─────────────────────────────────────────────────
DIRICHLET_ALPHA = 0.5
DATA_SEED = 499

# ─── 執行環境設定 ─────────────────────────────────────────────────
CLIENT_NUM_CPUS = 2.0
CLIENT_NUM_GPUS_IF_AVAILABLE = 0.1

# ─── XGBoost 設定 ─────────────────────────────────────────────────
XGB_OBJECTIVE = "multi:softprob"
XGB_NUM_CLASS = 10

# ─── 結果輸出設定 ─────────────────────────────────────────────────
RESULTS_DIR = "results/runs"
TIMESTAMP_FORMAT = "%Y%m%d-%H%M-%S"

# ─── 資料分布圖設定 ───────────────────────────────────────────────
CHART_DIR = "results/charts"
CHART_NAME = "distribution.png"
CHART_SHOW = True
CHART_FIGSIZE = (8.28, 4.49)


def _resolve_defaults() -> tuple[int, float]:
    """Resolve strategy-level None placeholders against global config values."""
    dp_num_clients = _STRATEGY_DP_NUM_SAMPLED_CLIENTS
    qfedavg_lr = _STRATEGY_QFEDAVG_CLIENT_LR

    if dp_num_clients is None:
        dp_num_clients = NUM_CLIENTS
    if qfedavg_lr is None:
        qfedavg_lr = LR

    return int(dp_num_clients), float(qfedavg_lr)


DP_NUM_SAMPLED_CLIENTS, QFEDAVG_CLIENT_LEARNING_RATE = _resolve_defaults()


def get_chart_path() -> Path:
    return Path(CHART_DIR) / CHART_NAME


def validate_config() -> None:
    from .validation import validate_config_module
    validate_config_module(sys.modules[__name__])
