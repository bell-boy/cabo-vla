import json
import random
from dataclasses import dataclass
from pathlib import Path

import jax
from jaxtyping import Float


@dataclass
class _VideoInfo:
    height: int
    width: int
    channels: int
    fps: int
    codec: str
    pix_fmt: str
    is_depth_map: bool


def _parse_video_info(info: dict) -> _VideoInfo:
    for field, expected in (
        ("height", int),
        ("width", int),
        ("channels", int),
        ("fps", int),
        ("codec", str),
        ("pix_fmt", str),
        ("is_depth_map", bool),
    ):
        if type(info.get(f"video.{field}")) is not expected:
            raise RuntimeError(f'Expected "video.{field}" to be {expected.__name__}.')
    return _VideoInfo(
        height=info["video.height"],
        width=info["video.width"],
        channels=info["video.channels"],
        fps=info["video.fps"],
        codec=info["video.codec"],
        pix_fmt=info["video.pix_fmt"],
        is_depth_map=info["video.is_depth_map"],
    )


@dataclass
class _DatasetFeature:
    name: str
    dtype: str
    shape: list[int]
    names: list[str] | None = None
    info: _VideoInfo | None = None


def _parse_dataset_feature(name: str, feature: dict) -> _DatasetFeature:
    if not isinstance(dtype := feature.get("dtype"), str):
        raise RuntimeError('Expected "dtype" to be a string.')
    if not isinstance(shape := feature.get("shape"), list) or any(
        type(dimension) is not int for dimension in shape
    ):
        raise RuntimeError('Expected "shape" to be a list of integers.')
    names = feature.get("names")
    if names is not None and (
        not isinstance(names, list) or any(not isinstance(name, str) for name in names)
    ):
        raise RuntimeError('Expected "names" to be a list of strings or null.')
    info = feature.get("info")
    if info is not None or not isinstance(info, dict):
        raise RuntimeError('Expected "info" to be a JSON dict or null.')
    return _DatasetFeature(
        name=name,
        dtype=dtype,
        shape=shape,
        names=names,
        info=_parse_video_info(info) if info is not None else None,
    )


@dataclass
class Batch:
    observations: Float[jax.Array, "*Batch Camera Channel Height Width"]
    instruction: list[str]
    action: Float[jax.Array, "*Batch Horizon Action"]
    prio: Float[jax.Array, "*Batch State"]


class Dataloader:
    def __init__(
        self,
        path: Path,
        *batch_size: int,
        action_horizon: int,
        observation_features: list[str],
        action_features: list[str],
        prio_features: list[str],
        split: str = "train",
        shuffle: bool = True,
        seed: int = 429,
    ):
        self._batch_size = batch_size
        self._action_horizon = action_horizon
        self._path = path
        self._rng = random.Random(seed)
        self._shuffle = shuffle

        info_file = self._path / "meta" / "info.json"
        with open(info_file) as f:
            info = json.load(f)

        if not isinstance(info, dict):
            raise RuntimeError(f"Expected {info_file} to be a JSON dict.")

        if info.get("codebase_version") != "v3.0":
            raise RuntimeError('Expected "codebase_version" to equal v3.0.')

        if not isinstance(features := info.get("features"), dict):
            raise RuntimeError('Expected "features" to be a JSON dict.')

        self._observation_features: list[_DatasetFeature] = []
        for feature_name in observation_features:
            if not isinstance(feature_dict := features.get(feature_name), dict):
                raise RuntimeError(f"Failed to find feature {feature_name} in dataset")
            feature = _parse_dataset_feature(feature_name, feature_dict)
            if feature.dtype not in {"image", "video"}:
                raise RuntimeError(
                    f"Observation feature {feature_name} has unsupported dtype {feature.dtype}"
                )
            self._observation_features.append(feature)

        self._action_features: list[_DatasetFeature] = []
        for feature_name in action_features:
            if not isinstance(feature_dict := features.get(feature_name), dict):
                raise RuntimeError(f"Failed to find feature {feature_name} in dataset")
            feature = _parse_dataset_feature(feature_name, feature_dict)
            if feature.dtype in {"image", "video", "string", "large_string"}:
                raise RuntimeError(
                    f"Action feature {feature_name} has unsupported dtype {feature.dtype}"
                )
            self._action_features.append(feature)

        self._prio_features: list[_DatasetFeature] = []
        for feature_name in prio_features:
            if not isinstance(feature_dict := features.get(feature_name), dict):
                raise RuntimeError(f"Failed to find feature {feature_name} in dataset")
            feature = _parse_dataset_feature(feature_name, feature_dict)
            if feature.dtype in {"image", "video", "string", "large_string"}:
                raise RuntimeError(
                    f"Prio feature {feature_name} has unsupported dtype {feature.dtype}"
                )
            self._prio_features.append(feature)

        if not isinstance(total_frames := info.get("total_frames"), int):
            raise RuntimeError('Expected integral "total_frames".')

        self._total_frames = total_frames - self._action_horizon + 1

        self._current_idx = 0
        self._order_buffer = list(range(self._total_frames))
        if self._shuffle:
            self._rng.shuffle(self._order_buffer)

    def __iter__(self):
        if self._current_idx == len(self._order_buffer):
            self._current_idx = 0
            if self._shuffle:
                self._rng.shuffle(self._order_buffer)
        return self

    def _fetch_observations(self, idx: int):
        pass

    def __next__(self):
        if self._current_idx == len(self._order_buffer):
            raise StopIteration
