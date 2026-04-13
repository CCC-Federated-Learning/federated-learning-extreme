import numpy as np

from config import DataDistribution
from app.model import _build_partitions
from app.data_cache import get_datasets, get_partition_cache


def _flatten_images(images: "torch.Tensor") -> np.ndarray:
    """Flatten [N, 28, 28] image tensor to float32 array with pixel values in [0, 1]."""
    return images.view(len(images), -1).numpy().astype(np.float32) / 255.0


def load_client_data_xgb(
    client_id: int,
    num_clients: int,
    distribution: DataDistribution,
    dirichlet_alpha: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_train, y_train, x_test, y_test) for the given client as tabular arrays."""
    if not (0 <= client_id < num_clients):
        raise ValueError(f"client_id {client_id} out of range [0, {num_clients})")

    train_ds, test_ds = get_datasets()
    cache = get_partition_cache()
    key = (distribution.name, num_clients, dirichlet_alpha, seed)

    train_idx = cache.get_or_build_train(
        key,
        lambda: _build_partitions(train_ds.targets, num_clients, distribution, dirichlet_alpha, seed),
    )[client_id]

    test_idx = cache.get_or_build_test(
        key,
        lambda: _build_partitions(test_ds.targets, num_clients, distribution, dirichlet_alpha, seed),
    )[client_id]

    x_train = _flatten_images(train_ds.data[train_idx])
    y_train = train_ds.targets[train_idx].numpy().astype(np.int32)
    x_test = _flatten_images(test_ds.data[test_idx])
    y_test = test_ds.targets[test_idx].numpy().astype(np.int32)

    return x_train, y_train, x_test, y_test


def load_test_data_xgb() -> tuple[np.ndarray, np.ndarray]:
    """Return the full MNIST test set as tabular arrays for centralised evaluation."""
    _, test_ds = get_datasets()
    x_test = _flatten_images(test_ds.data)
    y_test = test_ds.targets.numpy().astype(np.int32)
    return x_test, y_test
