from enum import Enum
from pathlib import Path


class _ConfigEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class StrategyName(_ConfigEnum):
    FEDAVG = "FedAvg"
    FEDAVGM = "FedAvgM"
    FEDPROX = "FedProx"


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
FRACTION_EVALUATE = 1 # merge all clients every round

STRATEGY_NAME = StrategyName.FEDAVGM
# Strategy-specific parameters
SERVER_LEARNING_RATE = 1.0  # FedAvgM
SERVER_MOMENTUM = 0.9       # FedAvgM
PROXIMAL_MU = 0.1           # FedProx

# Data partition settings
DATA_DISTRIBUTION = DataDistribution.LABEL 
# Data distribution options:
DIRICHLET_ALPHA = 0.5     # Dirichlet parameter (small = Non-IID)
DATA_SEED = 499 

# Runtime settings
CLIENT_NUM_CPUS = 2
CLIENT_NUM_GPUS_IF_AVAILABLE = 1.0

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
    distribution_values = {item.value.lower() for item in DataDistribution}
    if DATA_DISTRIBUTION.lower() not in distribution_values:
        supported = ", ".join(item.value for item in DataDistribution)
        raise ValueError(f"DATA_DISTRIBUTION must be one of: {supported}")
    if DIRICHLET_ALPHA <= 0:
        raise ValueError("DIRICHLET_ALPHA must be > 0")
    if NUM_PARTITIONS <= 0:
        raise ValueError("NUM_PARTITIONS must be > 0")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0")
