from pathlib import Path

# Global experiment settings
NUM_PARTITIONS = 10
BATCH_SIZE = 64
LOCAL_EPOCHS = 3
LR = 0.001
NUM_ROUNDS = 200
FRACTION_EVALUATE = 0.5
STRATEGY_NAME = "FedAvg"  # options: "FedAvg"

# Data partition settings
# three options: "IID", "Dirichlet", "label"
DATASET_NAME = "MNIST"

# DATA_DISTRIBUTION = "IID"
# DATA_DISTRIBUTION = "Dirichlet"
DATA_DISTRIBUTION = "label"
DIRICHLET_ALPHA = 0.005
DATA_SEED = 49

# Runtime settings
CLIENT_NUM_CPUS = 2
CLIENT_NUM_GPUS_IF_AVAILABLE = 1.0

# Draw settings
DRAW_SAVE_DIR = "distribution_chart"
DRAW_SAVE_NAME = "client_distribution.png"
DRAW_SHOW_PLOT = True
DRAW_FIGSIZE = (8.28, 4.49)

# Run record settings
ACC_CHART_DIR = "accchart"
RES_DIR = "res"

# File naming settings
TIMESTAMP_FORMAT = "%Y%m%d-%H%M-%S"


def get_draw_output_path() -> Path:
    return Path(DRAW_SAVE_DIR) / DRAW_SAVE_NAME


def validate_config() -> None:
    if STRATEGY_NAME.lower() not in {"fedavg"}:
        raise ValueError("STRATEGY_NAME must be one of: fedavg")
    if DATA_DISTRIBUTION.lower() not in {"iid", "dirichlet", "label"}:
        raise ValueError("DATA_DISTRIBUTION must be one of: iid, dirichlet, label")
    if DIRICHLET_ALPHA <= 0:
        raise ValueError("DIRICHLET_ALPHA must be > 0")
    if NUM_PARTITIONS <= 0:
        raise ValueError("NUM_PARTITIONS must be > 0")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0")
