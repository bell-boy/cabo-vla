import importlib.util
from pathlib import Path

import jax
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "dataloader", Path(__file__).parents[1] / "python/cabo_vla/data/dataloader.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
DataLoader = module.DataLoader


class Tensor:
    """Exercise the tensor protocol without installing PyTorch."""

    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self.value, dtype=np.float32)


def test_numeric_conversion_and_metadata():
    samples = [
        {
            "action": Tensor([i, i + 1]),
            "observation": {"image": np.full((3, 2, 2), i, dtype=np.uint8)},
            "task": "pick up",
            "index": i,
        }
        for i in range(3)
    ]
    loader = DataLoader(samples, batch_size=2, device=jax.devices("cpu")[0])
    batches = list(loader)
    assert len(loader) == len(batches) == 2
    assert isinstance(batches[0]["action"], jax.Array)
    np.testing.assert_array_equal(batches[0]["action"], [[0, 1], [1, 2]])
    assert batches[0]["action"].dtype == np.float32
    assert batches[0]["observation"]["image"].shape == (2, 3, 2, 2)
    assert batches[0]["observation"]["image"].dtype == np.uint8
    assert batches[0]["task"] == ["pick up", "pick up"]
    assert batches[1]["action"].shape == (1, 2)


def test_drop_last_and_empty_dataset():
    loader = DataLoader(list(range(5)), batch_size=2, drop_last=True)
    assert len(loader) == 2
    np.testing.assert_array_equal(np.concatenate(list(loader)), [0, 1, 2, 3])
    for drop_last in (False, True):
        loader = DataLoader([], batch_size=2, drop_last=drop_last)
        assert len(loader) == 0
        assert list(loader) == []


def test_shuffle_is_reproducible_and_covers_each_sample_each_epoch():
    a = DataLoader(list(range(20)), batch_size=3, shuffle=True, seed=7)
    b = DataLoader(list(range(20)), batch_size=3, shuffle=True, seed=7)
    epochs = []
    for _ in range(2):
        left, right = np.concatenate(list(a)), np.concatenate(list(b))
        np.testing.assert_array_equal(left, right)
        np.testing.assert_array_equal(np.sort(left), np.arange(20))
        epochs.append(left)
    assert not np.array_equal(*epochs)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True])
def test_invalid_batch_size(batch_size):
    with pytest.raises((TypeError, ValueError)):
        DataLoader([], batch_size=batch_size)


def test_bad_samples_report_feature_path():
    with pytest.raises(ValueError, match="sample.action"):
        list(DataLoader([{"action": [1]}, {"action": [1, 2]}]))
    with pytest.raises(ValueError, match="dictionary keys"):
        list(DataLoader([{"action": [1]}, {"state": [1]}]))
    with pytest.raises(TypeError, match="sample.action"):
        list(DataLoader([{"action": object()}]))


def test_iteration_is_lazy_and_independent():
    class Dataset:
        accessed = []

        def __len__(self):
            return 5

        def __getitem__(self, index):
            self.accessed.append(index)
            return index

    dataset = Dataset()
    loader = DataLoader(dataset, batch_size=2)
    first, second = iter(loader), iter(loader)
    assert dataset.accessed == []
    np.testing.assert_array_equal(next(first), [0, 1])
    assert dataset.accessed == [0, 1]
    np.testing.assert_array_equal(next(second), [0, 1])
    np.testing.assert_array_equal(next(first), [2, 3])
