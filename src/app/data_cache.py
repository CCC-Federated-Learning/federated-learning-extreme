from typing import Callable

import torch
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


class DatasetCache:
    """Singleton MNIST dataset cache shared across PyTorch and XGBoost backends."""

    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def __init__(self):
        self._train = None
        self._test = None

    def get_datasets(self) -> tuple:
        if self._train is None:
            self._train = datasets.MNIST(
                root="./results/data", train=True, download=True, transform=self.pytorch_transforms
            )
            self._test = datasets.MNIST(
                root="./results/data", train=False, download=True, transform=self.pytorch_transforms
            )
        return self._train, self._test

    def clear(self) -> None:
        self._train = None
        self._test = None


class PartitionIndexCache:
    """Cache for client partition index lists to avoid recomputation."""

    def __init__(self):
        self._train: dict[tuple, list] = {}
        self._test: dict[tuple, list] = {}

    def get_or_build_train(self, key: tuple, builder: Callable[[], list]) -> list:
        if key not in self._train:
            self._train[key] = builder()
        return self._train[key]

    def get_or_build_test(self, key: tuple, builder: Callable[[], list]) -> list:
        if key not in self._test:
            self._test[key] = builder()
        return self._test[key]

    def clear(self) -> None:
        self._train.clear()
        self._test.clear()


_dataset_cache = DatasetCache()
_partition_cache = PartitionIndexCache()


def get_datasets() -> tuple:
    return _dataset_cache.get_datasets()


def get_partition_cache() -> PartitionIndexCache:
    return _partition_cache


def clear_all_caches() -> None:
    _dataset_cache.clear()
    _partition_cache.clear()
