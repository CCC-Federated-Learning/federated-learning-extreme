import numpy as np

from config import DataDistribution
from task import _build_partition_indices, _get_datasets


_partition_cache = {"train": {}, "test": {}}


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
    """Load partitioned MNIST data as tabular arrays for XGBoost clients."""
    if partition_id < 0 or partition_id >= num_partitions:
        raise ValueError(
            f"partition_id {partition_id} is out of range for num_partitions={num_partitions}"
        )

    train_dataset, test_dataset = _get_datasets()
    cache_key = (distribution.name, num_partitions, dirichlet_alpha, seed)

    if cache_key not in _partition_cache["train"]:
        _partition_cache["train"][cache_key] = _build_partition_indices(
            train_dataset.targets,
            num_partitions,
            distribution,
            dirichlet_alpha,
            seed,
        )
    if cache_key not in _partition_cache["test"]:
        _partition_cache["test"][cache_key] = _build_partition_indices(
            test_dataset.targets,
            num_partitions,
            distribution,
            dirichlet_alpha,
            seed,
        )

    train_indices = _partition_cache["train"][cache_key][partition_id]
    test_indices = _partition_cache["test"][cache_key][partition_id]

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
