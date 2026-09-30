"""JAX batching for LeRobot datasets, without PyTorch's DataLoader."""

from collections.abc import Iterator, Mapping, Sequence
import operator
from typing import Any

import jax
import numpy as np


def _collate(values: list[Any], device: jax.Device | None, path: str = "sample") -> Any:
    """Stack numeric leaves; retain text metadata as lists on the host."""
    first = values[0]
    if isinstance(first, Mapping):
        keys = first.keys()
        if any(not isinstance(value, Mapping) or value.keys() != keys for value in values):
            raise ValueError(f"Inconsistent dictionary keys at {path}")
        return {key: _collate([value[key] for value in values], device, f"{path}.{key}") for key in keys}

    if isinstance(first, (str, bytes)) or first is None:
        if any(type(value) is not type(first) for value in values):
            raise TypeError(f"Inconsistent metadata types at {path}")
        return values

    arrays = []
    for value in values:
        # LeRobot returns torch.Tensor leaves. Duck typing avoids a torch dependency.
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "biufc":
            raise TypeError(f"Unsupported dtype {array.dtype} at {path}; expected numeric data")
        arrays.append(array)
    try:
        batch = np.stack(arrays)
    except ValueError as error:
        raise ValueError(f"Cannot batch differently shaped values at {path}") from error
    return jax.device_put(batch, device)


class DataLoader:
    """Thin wrapper around an indexed LeRobot dataset with JAX conversion.

    Numeric leaves (including PyTorch tensors) are stacked on a new leading
    batch axis and placed on ``device``. Nested dictionaries are preserved;
    text such as ``task`` remains a list of strings. Image layout and values
    are preserved as supplied by the dataset. No normalization is applied.

    Each iteration is one epoch. With ``shuffle=True``, ``seed`` produces
    reproducible but different permutations across epochs. The final partial
    batch is included unless ``drop_last=True``. Batches load synchronously.

    Example::

        dataset = LeRobotDataset(repo_id, root=local_path)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, seed=0)
        for batch in loader:
            train_step(batch["observation.state"], batch["action"])
    """

    def __init__(
        self,
        dataset: Sequence[Any],
        batch_size: int = 32,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        drop_last: bool = False,
        device: jax.Device | None = None,
    ) -> None:
        if isinstance(batch_size, bool):
            raise TypeError("batch_size must be an integer")
        batch_size = operator.index(batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not hasattr(dataset, "__len__") or not hasattr(dataset, "__getitem__"):
            raise TypeError("dataset must support len(dataset) and dataset[index]")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        """Return the number of batches in an epoch."""
        size = len(self.dataset)
        return size // self.batch_size if self.drop_last else (size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Any]:
        size = len(self.dataset)
        indices = self._rng.permutation(size) if self.shuffle else range(size)
        stop = size - size % self.batch_size if self.drop_last else size
        return self._batches(indices, stop)

    def _batches(self, indices: Any, stop: int) -> Iterator[Any]:
        for start in range(0, stop, self.batch_size):
            samples = [self.dataset[int(index)] for index in indices[start : min(start + self.batch_size, stop)]]
            yield _collate(samples, self.device)
