"""Centralized data caching for MNIST dataset and partition indices.

This module provides unified caching for both PyTorch (task.py) and XGBoost (task_xgb.py)
to prevent duplication and ensure consistent behavior across different learner backends.
"""

from typing import Dict, Callable, Any, List
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor


class PartitionIndexCache:
    """Thread-safe partition index cache for train/test splits.
    
    Caches indices computed based on distribution strategy to avoid
    recomputation across multiple clients and backends.
    """

    def __init__(self):
        # Cache key: (distribution.name, num_partitions, dirichlet_alpha, seed)
        # Cache value: list of indices for each partition
        self._train_cache: Dict[tuple, List] = {}
        self._test_cache: Dict[tuple, List] = {}

    def get_or_build_train(self, cache_key: tuple, builder_func: Callable[[], List]) -> List:
        """Get cached train partition indices or build them.
        
        Args:
            cache_key: Tuple key for caching (distribution, num_partitions, alpha, seed)
            builder_func: Function that builds indices if not cached
            
        Returns:
            List of partition index lists for training data
        """
        if cache_key not in self._train_cache:
            self._train_cache[cache_key] = builder_func()
        return self._train_cache[cache_key]

    def get_or_build_test(self, cache_key: tuple, builder_func: Callable[[], List]) -> List:
        """Get cached test partition indices or build them.
        
        Args:
            cache_key: Tuple key for caching (distribution, num_partitions, alpha, seed)
            builder_func: Function that builds indices if not cached
            
        Returns:
            List of partition index lists for test data
        """
        if cache_key not in self._test_cache:
            self._test_cache[cache_key] = builder_func()
        return self._test_cache[cache_key]

    def clear(self):
        """Clear all cached partition indices."""
        self._train_cache.clear()
        self._test_cache.clear()


class DatasetCache:
    """Centralized MNIST dataset cache (train and test splits).
    
    Ensures MNIST is downloaded and loaded only once, shared across
    both PyTorch and XGBoost backends.
    """

    # MNIST normalization constants (standard values for grayscale images)
    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    def __init__(self):
        self._cached_train = None
        self._cached_test = None

    def get_datasets(self) -> tuple:
        """Load and cache MNIST train/test datasets.
        
        Returns:
            Tuple of (train_dataset, test_dataset) from torchvision.datasets.MNIST
        """
        if self._cached_train is None:
            self._cached_train = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=self.pytorch_transforms,
            )
            self._cached_test = datasets.MNIST(
                root="./data",
                train=False,
                download=True,
                transform=self.pytorch_transforms,
            )
        return self._cached_train, self._cached_test

    def clear(self):
        """Clear cached datasets."""
        self._cached_train = None
        self._cached_test = None


# ============================================================================
# Global Cache Instances
# ============================================================================
# Single instances shared across the entire application.
# - _dataset_cache: Manages MNIST raw data loading
# - _partition_cache: Manages client data partition indices


_dataset_cache = DatasetCache()
_partition_cache = PartitionIndexCache()


# ============================================================================
# Public API
# ============================================================================


def get_datasets() -> tuple:
    """Shared API to get MNIST datasets (train and test).
    
    This is called by both task.py and task_xgb.py to ensure
    a single dataset instance is used across the entire application.
    
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    return _dataset_cache.get_datasets()


def get_partition_cache() -> PartitionIndexCache:
    """Shared API to access the global partition cache.
    
    Returns:
        PartitionIndexCache instance for managing client partition indices
    """
    return _partition_cache


def clear_all_caches():
    """Clear all caches (useful for testing or cleanup).
    
    Clears both dataset and partition index caches. Use this when
    you need to reset the application state.
    """
    _dataset_cache.clear()
    _partition_cache.clear()
