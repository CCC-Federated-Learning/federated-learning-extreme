"""XGBoost task implementation for federated learning on MNIST.

Provides data loading and model evaluation for XGBoost-based strategies
using the shared data cache for consistency with PyTorch clients.
"""

import numpy as np

from config import DataDistribution
from app.task import _build_partition_indices, _get_datasets
from app.data_cache import get_partition_cache


def _flatten_images(images) -> np.ndarray:
    """Convert [N, 28, 28] torch tensor images into float32 tabular features."""
    return images.view(len(images), -1).numpy().astype(np.float32) / 255.0


def load_partition_data(
    partition_id: int,
    num_partitions: int,
    distribution: DataDistribution,
    dirichlet_alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load partitioned MNIST data as tabular arrays for XGBoost clients.
    
    Uses the shared partition cache to ensure consistency with PyTorch clients.
    """
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id {partition_id} is out of range for num_partitions={num_partitions}"
        )

    train_dataset, test_dataset = _get_datasets()
    partition_cache = get_partition_cache()
    cache_key = (distribution.name, num_partitions, dirichlet_alpha, seed)

    # Use shared cache to get or build partition indices
    train_indices = partition_cache.get_or_build_train(
        cache_key,
        lambda: _build_partition_indices(
            train_dataset.targets,
            num_partitions,
            distribution,
            dirichlet_alpha,
            seed,
        ),
    )[partition_id]
    
    test_indices = partition_cache.get_or_build_test(
        cache_key,
        lambda: _build_partition_indices(
            test_dataset.targets,
            num_partitions,
            distribution,
            dirichlet_alpha,
            seed,
        ),
    )[partition_id]

    x_train = _flatten_images(train_dataset.data[train_indices])
    y_train = train_dataset.targets[train_indices].numpy().astype(np.int32)

    x_test = _flatten_images(test_dataset.data[test_indices])
    y_test = test_dataset.targets[test_indices].numpy().astype(np.int32)

    return x_train, y_train, x_test, y_test


def load_centralized_test_data() -> tuple[np.ndarray, np.ndarray]:
    """Load centralized MNIST test data as tabular arrays for XGBoost eval."""
    _, test_dataset = _get_datasets()
    x_test = _flatten_images(test_dataset.data)
    y_test = test_dataset.targets.numpy().astype(np.int32)
    return x_test, y_test
