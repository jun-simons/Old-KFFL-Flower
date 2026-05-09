"""Main dataset handler for KFFL.

Provides a unified API that:
  1. Loads a named dataset via its dataset-specific loader.
  2. Wraps the data in a PyTorch ``Dataset`` yielding ``(X, y, s)`` tuples,
     where ``s`` is a 1-D tensor of shape ``(k,)`` containing all k sensitive
     attribute codes for that sample.
  3. Partitions the data across federated clients (IID by default).
  4. Returns ``DataLoader`` instances ready for use in FAIR1, FAIR2,
     and the local-update step.

Usage
-----
>>> loaders = get_federated_loaders("adult", num_partitions=5, batch_size=64)
>>> loader_0 = loaders[0]
>>> for X_batch, y_batch, s_batch in loader_0:
...     # X_batch: (batch, d)   float32
...     # y_batch: (batch,)     long
...     # s_batch: (batch, k)   long
...     pass
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from .adult import DatasetBundle

# ---------------------------------------------------------------------------
# Registry: maps dataset name → module path
# ---------------------------------------------------------------------------

_DATASET_REGISTRY: Dict[str, str] = {
    "adult": "kffl.data.adult",
    "toy": "kffl.data.toy",
    "no_sensitive": "kffl.data.no_sensitive",
}


# ---------------------------------------------------------------------------
# PyTorch Dataset wrapper
# ---------------------------------------------------------------------------


class FairDataset(Dataset):
    """Wraps a ``DatasetBundle`` as a PyTorch map-style dataset.

    ``__getitem__`` returns ``(X[i], y[i], s[i])`` where:
      - ``X[i]`` has shape ``(d,)``, dtype float32
      - ``y[i]`` is a scalar long tensor
      - ``s[i]`` has shape ``(k,)``, dtype long  (k = number of sensitive attrs)
    """

    def __init__(self, bundle: DatasetBundle) -> None:
        self.X = torch.as_tensor(bundle.X, dtype=torch.float32)
        self.y = torch.as_tensor(bundle.y, dtype=torch.long)
        self.s = torch.as_tensor(bundle.s, dtype=torch.long)  # (n, k)
        self.feature_names = bundle.feature_names
        self.sensitive_names = bundle.sensitive_names
        self.target_name = bundle.target_name
        self.sensitive_labels = bundle.sensitive_labels

    @property
    def num_sensitive(self) -> int:
        """Number of sensitive attributes (k)."""
        return self.s.shape[1]

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], self.s[idx]  # s[idx] shape: (k,)


# ---------------------------------------------------------------------------
# Partitioning helpers
# ---------------------------------------------------------------------------


def _iid_partition_indices(
    n_samples: int,
    num_partitions: int,
    seed: int = 42,
) -> List[np.ndarray]:
    """Split sample indices into *num_partitions* roughly-equal IID shards."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    return [part.copy() for part in np.array_split(indices, num_partitions)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_dataset(
    name: str,
    sensitive_features: List[str] | None = None,
    target_feature: str | None = None,
) -> DatasetBundle:
    """Load a dataset by name, returning a ``DatasetBundle``.

    Parameters
    ----------
    name:
        Key in the dataset registry (e.g. ``"adult"``).
    sensitive_features:
        Override the default sensitive attributes for the dataset.
        Pass a list of column names, e.g. ``["sex", "race"]``.
    target_feature:
        Override the default target column for the dataset.
    """
    if name not in _DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. Available: {sorted(_DATASET_REGISTRY)}"
        )

    module = importlib.import_module(_DATASET_REGISTRY[name])
    kwargs: dict = {}
    if sensitive_features is not None:
        kwargs["sensitive_features"] = sensitive_features
    if target_feature is not None:
        kwargs["target_feature"] = target_feature
    return module.load(**kwargs)


def get_federated_loaders(
    name: str,
    num_partitions: int,
    batch_size: int = 64,
    sensitive_features: List[str] | None = None,
    target_feature: str | None = None,
    seed: int = 42,
    shuffle: bool = True,
) -> List[DataLoader]:
    """Load a dataset, partition it IID, and return one ``DataLoader`` per client.

    Parameters
    ----------
    name:
        Dataset name (registry key).
    num_partitions:
        Number of federated clients / partitions.
    batch_size:
        Batch size for each client's DataLoader.
    sensitive_features:
        List of sensitive attribute column names to pass to the loader.
    target_feature:
        Override the target column name.
    seed:
        Random seed for reproducible partitioning.
    shuffle:
        Whether to shuffle within each client's DataLoader.

    Returns
    -------
    List of ``DataLoader``, one per partition.  Each batch yields
    ``(X, y, s)`` with ``s`` of shape ``(batch, k)``.
    """
    bundle = load_dataset(name, sensitive_features, target_feature)
    full_dataset = FairDataset(bundle)

    partition_indices = _iid_partition_indices(
        len(full_dataset), num_partitions, seed=seed
    )

    loaders: List[DataLoader] = []
    for indices in partition_indices:
        subset = Subset(full_dataset, indices.tolist())
        loaders.append(
            DataLoader(subset, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        )
    return loaders
