from pathlib import Path

# Global experiment settings
NUM_PARTITIONS = 10
BATCH_SIZE = 64
LOCAL_EPOCHS = 1
LR = 0.001
NUM_ROUNDS = 3
FRACTION_EVALUATE = 0.5

# Data partition settings

# DATA_DISTRIBUTION = "iid"
# DATA_DISTRIBUTION = "dirichlet"
DATA_DISTRIBUTION = "label"
DIRICHLET_ALPHA = 0.005
DATA_SEED = 42

# Runtime settings
CLIENT_NUM_CPUS = 2
CLIENT_NUM_GPUS_IF_AVAILABLE = 1.0

# Draw settings
DRAW_SAVE_DIR = "draw_data_init_img"
DRAW_SAVE_NAME = "client_distribution.png"
DRAW_SHOW_PLOT = True
DRAW_FIGSIZE = (8.28, 4.49)


def get_draw_output_path() -> Path:
    return Path(DRAW_SAVE_DIR) / DRAW_SAVE_NAME


def validate_config() -> None:
    if DATA_DISTRIBUTION not in {"iid", "dirichlet", "label"}:
        raise ValueError("DATA_DISTRIBUTION must be one of: iid, dirichlet, label")
    if DIRICHLET_ALPHA <= 0:
        raise ValueError("DIRICHLET_ALPHA must be > 0")
    if NUM_PARTITIONS <= 0:
        raise ValueError("NUM_PARTITIONS must be > 0")
    if BATCH_SIZE <= 0:
        raise ValueError("BATCH_SIZE must be > 0")
