import json
import random
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from pathlib import Path

import av
import jax
import jax.numpy as jnp
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from flax import struct
from jaxtyping import Float, Int
from PIL import Image, ImageOps


class NormalizationType(Enum):
    NONE = 1
    QUANTILE = 2
    MEANVAR = 3


class PaddingType(Enum):
    REPEAT_LAST = 1
    ZERO = 2


@struct.dataclass
class Batch:
    observations: Float[jax.Array, "Batch Camera Channel Height Width"]
    instruction: list[Int[jax.Array, " seq"]]
    action: Float[jax.Array, "Batch Horizon Action"]
    prio: Float[jax.Array, "Batch State"]
    timestamp: Float[jax.Array, "Batch"]


@dataclass
class _VideoInfo:
    height: int
    width: int
    channels: int
    fps: int
    codec: str
    pix_fmt: str
    is_depth_map: bool


@dataclass
class _Normalization:
    offset: jax.Array
    divisor: jax.Array
    quantile: bool

    def apply(self, value: jax.Array) -> jax.Array:
        normalized = (value - self.offset) / self.divisor
        return 2 * normalized - 1 if self.quantile else normalized


@dataclass
class _DatasetFeature:
    name: str
    dtype: str
    shape: list[int]
    names: list[str] | None = None
    info: _VideoInfo | None = None
    normalization: _Normalization | None = None


@dataclass
class _EpisodeVideoMeta:
    feature_name: str
    chunk_idx: int
    file_idx: int
    from_timestamp: float
    to_timestamp: float


@dataclass
class _Episode:
    data_chunk_index: int
    data_file_index: int
    data_from_index: int
    data_to_index: int
    video_metadata: list[_EpisodeVideoMeta]
    episode_index: int


@dataclass(frozen=True)
class DataloaderConfig:
    path: Path
    resolution: tuple[int, int]
    batch_size: int
    action_horizon: int
    observation_features: tuple[str, ...]
    action_features: tuple[tuple[str, NormalizationType], ...]
    prio_features: tuple[tuple[str, NormalizationType], ...]
    tokenizer: Callable[[str], Int[jax.Array, " seq"]]
    seq_len: int
    pad_token_id: int
    observation_range: tuple[float, float] = (0.0, 1.0)
    padding_type: PaddingType = PaddingType.REPEAT_LAST
    dtype: jax.typing.DTypeLike = jnp.float32
    split: str | tuple[int, ...] = "train"
    shuffle: bool = True
    seed: int = 429
    episode_indices: tuple[int, ...] | None = None


@dataclass
class DataloaderState:
    config: DataloaderConfig
    rng_state: tuple
    order_buffer: list[int]
    current_position: int


def _iter_episodes(path: Path, video_features: list[str], batch_size: int = 1024) -> Iterator[_Episode]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    metadata_path = path / "meta" / "episodes"
    if not metadata_path.is_dir():
        raise FileNotFoundError(f"No episode metadata Parquet files found in {metadata_path}")
    dataset = ds.dataset(metadata_path, format="parquet")
    if not dataset.files:
        raise FileNotFoundError(f"No episode metadata Parquet files found in {metadata_path}")

    required = (
        "episode_index",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    )
    columns = list(required) + [
        f"videos/{feature}/{field}" for feature in video_features for field in ("chunk_index", "file_index", "from_timestamp", "to_timestamp")
    ]
    for batch in dataset.to_batches(columns=columns, batch_size=batch_size):
        for row in batch.to_pylist():
            for key in required:
                if type(row.get(key)) is not int or row[key] < 0:
                    raise RuntimeError(f"Expected nonnegative integer {key!r} in {metadata_path}.")
            if row["dataset_to_index"] - row["dataset_from_index"] != row["length"]:
                raise RuntimeError(f"Episode length does not match frame bounds in {metadata_path}.")

            videos = []
            for feature in video_features:
                prefix = f"videos/{feature}"
                if row[f"{prefix}/chunk_index"] is None:
                    continue
                try:
                    videos.append(
                        _EpisodeVideoMeta(
                            feature_name=feature,
                            chunk_idx=row[f"{prefix}/chunk_index"],
                            file_idx=row[f"{prefix}/file_index"],
                            from_timestamp=row[f"{prefix}/from_timestamp"],
                            to_timestamp=row[f"{prefix}/to_timestamp"],
                        )
                    )
                except KeyError as error:
                    raise RuntimeError(f"Missing video metadata {error.args[0]!r} in {metadata_path}.") from error

            yield _Episode(
                data_chunk_index=row["data/chunk_index"],
                data_file_index=row["data/file_index"],
                data_from_index=row["dataset_from_index"],
                data_to_index=row["dataset_to_index"],
                video_metadata=videos,
                episode_index=row["episode_index"],
            )


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


def _parse_dataset_feature(name: str, feature: dict, mode: NormalizationType, stats: dict) -> _DatasetFeature:
    if not isinstance(dtype := feature.get("dtype"), str):
        raise RuntimeError('Expected "dtype" to be a string.')
    if not isinstance(shape := feature.get("shape"), list) or any(type(dimension) is not int for dimension in shape):
        raise RuntimeError('Expected "shape" to be a list of integers.')
    names = feature.get("names")
    if names is not None and (not isinstance(names, list) or any(not isinstance(name, str) for name in names)):
        raise RuntimeError('Expected "names" to be a list of strings or null.')
    info = feature.get("info")
    if info is not None and not isinstance(info, dict):
        raise RuntimeError('Expected "info" to be a JSON dict or null.')
    parsed = _DatasetFeature(
        name=name,
        dtype=dtype,
        shape=shape,
        names=names,
        info=_parse_video_info(info) if info is not None else None,
    )
    parsed.normalization = _parse_normalization(parsed, mode, stats)
    return parsed


def _read_row(
    path: Path,
    frame_index: int,
    columns: list[str] | None = None,
    *,
    num_rows: int = 1,
) -> dict[str, list]:
    if num_rows < 0:
        raise ValueError("num_rows must be nonnegative")
    if num_rows == 0:
        return {}
    selected_columns = None if columns is None else list(dict.fromkeys(["index", *columns]))
    table = ds.dataset(path, format="parquet").to_table(
        columns=selected_columns,
        filter=(ds.field("index") >= frame_index) & (ds.field("index") < frame_index + num_rows),
    )
    rows = table.sort_by("index").to_pydict()
    if rows["index"] != list(range(frame_index, frame_index + num_rows)):
        raise IndexError("Frame range not found")
    if columns is not None and "index" not in columns:
        rows.pop("index")
    return rows


def _read_video_frame(path: Path, timestamp: float) -> Image.Image:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise RuntimeError(f"No video stream found in {path}.")
        stream = container.streams.video[0]
        if stream.time_base is None or stream.average_rate is None:
            raise RuntimeError(f"Missing video timing information in {path}.")
        tolerance = 0.5 / float(stream.average_rate) + 1e-6
        container.seek(int(timestamp / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if abs(frame.time - timestamp) <= tolerance:
                return frame.to_image()
            if frame.time > timestamp + tolerance:
                break
    raise RuntimeError(f"No video frame found at {timestamp:.6f}s in {path}.")


def _parse_normalization(feature: _DatasetFeature, mode: NormalizationType, stats: dict) -> _Normalization | None:
    if mode == NormalizationType.NONE:
        return None
    keys = ("mean", "std") if mode == NormalizationType.MEANVAR else ("q01", "q99")
    values = []
    for key in keys:
        try:
            value = jnp.asarray(stats[feature.name][key], dtype=jnp.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid {key} statistics for {feature.name}.") from error
        if value.shape != tuple(feature.shape) or not bool(jnp.all(jnp.isfinite(value))):
            raise ValueError(f"Expected finite {key} statistics with shape {feature.shape} for {feature.name}.")
        values.append(value)
    offset, upper_or_std = values
    quantile = mode == NormalizationType.QUANTILE
    divisor = upper_or_std - offset if quantile else upper_or_std
    if bool(jnp.any(divisor < 0)) or not bool(jnp.all(jnp.isfinite(divisor))):
        raise ValueError(f"Invalid normalization scale for {feature.name}.")
    # Constant components have no usable scale. Subtract their offset without
    # amplifying numerical noise or dividing by zero.
    divisor = jnp.where(divisor == 0, 1, divisor)
    return _Normalization(offset, divisor, quantile)


class Dataloader:
    def __init__(
        self,
        config: DataloaderConfig,
        *,
        rng_state: tuple | None = None,
        order_buffer: list[int] | None = None,
        current_position: int = 0,
    ):
        if config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if config.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        self._config = config
        self._tokenizer = config.tokenizer
        self._batch_size = config.batch_size
        self._action_horizon = config.action_horizon
        self._path = config.path
        self._rng = random.Random(config.seed)
        if rng_state is not None:
            self._rng.setstate(rng_state)
        self._shuffle = config.shuffle
        self._dtype = config.dtype
        self._resolution = config.resolution
        self._observation_range = config.observation_range
        self._padding_type = config.padding_type
        if not jnp.issubdtype(config.dtype, jnp.floating):
            raise ValueError("dtype must be floating point for normalized observations")

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
        for feature_name in config.observation_features:
            if not isinstance(feature_dict := features.get(feature_name), dict):
                raise RuntimeError(f"Failed to find feature {feature_name} in dataset")
            feature = _parse_dataset_feature(feature_name, feature_dict, NormalizationType.NONE, {})
            if feature.dtype not in {"image", "video"}:
                raise RuntimeError(f"Observation feature {feature_name} has unsupported dtype {feature.dtype}")
            self._observation_features.append(feature)

        stats = {}
        if any(mode != NormalizationType.NONE for _, mode in config.action_features + config.prio_features):
            with open(self._path / "meta" / "stats.json") as f:
                stats = json.load(f)
            if not isinstance(stats, dict):
                raise ValueError("Expected meta/stats.json to contain a JSON dict.")

        self._action_features: list[_DatasetFeature] = []
        for feature_name, mode in config.action_features:
            if not isinstance(feature_dict := features.get(feature_name), dict):
                raise RuntimeError(f"Failed to find feature {feature_name} in dataset")
            feature = _parse_dataset_feature(feature_name, feature_dict, mode, stats)
            if feature.dtype in {"image", "video", "string", "large_string"}:
                raise RuntimeError(f"Action feature {feature_name} has unsupported dtype {feature.dtype}")
            self._action_features.append(feature)

        self._prio_features: list[_DatasetFeature] = []
        for feature_name, mode in config.prio_features:
            if not isinstance(feature_dict := features.get(feature_name), dict):
                raise RuntimeError(f"Failed to find feature {feature_name} in dataset")
            feature = _parse_dataset_feature(feature_name, feature_dict, mode, stats)
            if feature.dtype in {"image", "video", "string", "large_string"}:
                raise RuntimeError(f"Prio feature {feature_name} has unsupported dtype {feature.dtype}")
            self._prio_features.append(feature)

        if config.action_horizon <= 0:
            raise ValueError("action_horizon must be positive")

        split_episodes: range | set[int]
        split = config.split
        if isinstance(split, str):
            if not isinstance(splits := info.get("splits"), dict):
                raise RuntimeError('Expected "splits" to be a JSON dict.')
            if not isinstance(episode_range := splits.get(split), str):
                raise RuntimeError(f"Failed to find split {split!r} in dataset.")
            try:
                episode_from, episode_to = map(int, episode_range.split(":"))
            except ValueError as error:
                raise RuntimeError(f"Expected an episode range 'start:end' for split {split!r}.") from error
            if not 0 <= episode_from <= episode_to:
                raise RuntimeError(f"Invalid episode range for split {split!r}.")
            split_episodes = range(episode_from, episode_to)
        else:
            split_episodes = set(split)

        selected_episodes = set(config.episode_indices) if config.episode_indices is not None else None
        self._episodes = [
            episode
            for episode in _iter_episodes(
                self._path,
                [feature.name for feature in self._observation_features if feature.dtype == "video"],
            )
            if episode.episode_index in split_episodes
        ]
        if not isinstance(split, str) and split_episodes != {ep.episode_index for ep in self._episodes}:
            raise ValueError("Requested episodes must exist in the dataset.")
        if selected_episodes is not None:
            self._episodes = [ep for ep in self._episodes if ep.episode_index in selected_episodes]
        if selected_episodes is not None and selected_episodes != {ep.episode_index for ep in self._episodes}:
            raise ValueError("Requested episodes must exist in the selected split.")

        if not isinstance(data_path_template := info.get("data_path"), str):
            raise RuntimeError("Expected string data_path.")

        self._data_path_template = data_path_template
        self._video_path_template = info.get("video_path")
        if any(feature.dtype == "video" for feature in self._observation_features):
            if not isinstance(self._video_path_template, str):
                raise RuntimeError("Expected string video_path for video observations.")

        self._tasks: list[str] = pq.read_table(self._path / "meta" / "tasks.parquet").to_pandas().index.tolist()

        if order_buffer is None:
            self._order_buffer = [
                idx for episode in sorted(self._episodes, key=lambda ep: ep.data_from_index) for idx in range(episode.data_from_index, episode.data_to_index)
            ]
            if self._shuffle:
                self._rng.shuffle(self._order_buffer)
        else:
            self._order_buffer = order_buffer.copy()
        if not 0 <= current_position <= len(self._order_buffer):
            raise ValueError("current_position must be within the order buffer")
        self.current_position = current_position

    def get_state(self) -> DataloaderState:
        return DataloaderState(
            config=self._config,
            rng_state=self._rng.getstate(),
            order_buffer=self._order_buffer.copy(),
            current_position=self.current_position,
        )

    @classmethod
    def from_state(cls, state: DataloaderState) -> "Dataloader":
        return cls(
            state.config,
            rng_state=state.rng_state,
            order_buffer=state.order_buffer,
            current_position=state.current_position,
        )

    def __len__(self) -> int:
        return (len(self._order_buffer) + self._batch_size - 1) // self._batch_size

    def __iter__(self) -> Iterator[Batch]:
        while self.current_position < len(self._order_buffer):
            start = self.current_position
            indices = self._order_buffer[start : start + self._batch_size]
            samples = [self._fetch_single(idx) for idx in indices]
            batch = Batch(
                observations=jnp.stack([s[0] for s in samples], axis=0),
                instruction=[self._tokenize_instruction(s[3]) for s in samples],
                action=jnp.stack([s[1] for s in samples], axis=0),
                prio=jnp.stack([s[2] for s in samples], axis=0),
                timestamp=jnp.asarray([s[4] for s in samples], dtype=jnp.float32),
            )
            self.current_position += len(indices)
            yield batch
        self.current_position = 0
        if self._shuffle:
            self._rng.shuffle(self._order_buffer)

    def _tokenize_instruction(self, text: str) -> Int[jax.Array, " seq"]:
        tokens = self._tokenizer(text)[: self._config.seq_len]
        return jnp.pad(
            tokens,
            (0, self._config.seq_len - tokens.shape[0]),
            constant_values=self._config.pad_token_id,
        )

    def _fetch_single(self, idx: int) -> tuple[jax.Array, jax.Array, jax.Array, str, float]:
        episode: _Episode | None = None
        for ep in self._episodes:
            if ep.data_from_index <= idx and idx < ep.data_to_index:
                episode = ep
                break

        if episode is None:
            raise RuntimeError(f"Index {idx} is out of bounds.")

        image_features = [x.name for x in self._observation_features if x.dtype == "image"]
        prio_features = [x.name for x in self._prio_features]
        action_features = [x.name for x in self._action_features]

        non_action_features = image_features + prio_features + ["task_index", "timestamp"]
        feature_file = self._data_path_template.format(
            chunk_index=episode.data_chunk_index,
            file_index=episode.data_file_index,
        )
        feature_row = {name: values[0] for name, values in _read_row(self._path / feature_file, idx, columns=non_action_features).items()}
        action_count = min(self._action_horizon, episode.data_to_index - idx)
        action_rows = _read_row(
            self._path / feature_file,
            idx,
            columns=action_features,
            num_rows=action_count,
        )

        if not isinstance(task_id := feature_row.get("task_index"), int):
            raise RuntimeError(f"Expected integral task_id for frame {idx}.")
        task: str = self._tasks[task_id]

        frame_timestamp = feature_row.get("timestamp")
        if isinstance(frame_timestamp, list) and len(frame_timestamp) == 1:
            frame_timestamp = frame_timestamp[0]
        if not isinstance(frame_timestamp, (int, float)):
            raise RuntimeError(f"Expected scalar timestamp for frame {idx}.")
        frame_timestamp = float(frame_timestamp)

        actions: list[Float[jax.Array, "Horizon Action"]] = []
        for feature in self._action_features:
            feat = feature.name
            if not isinstance(action := action_rows.get(feat), list):
                raise RuntimeError(f"Failed to load action feature {feat} for frame {idx}.")
            action_tensor = jnp.asarray(action, dtype=jnp.promote_types(self._dtype, jnp.float32))
            if action_tensor.ndim == 1:
                action_tensor = action_tensor[:, None]
            if feature.normalization is not None:
                action_tensor = feature.normalization.apply(action_tensor)
            if action_count < self._action_horizon:
                action_tensor = jnp.pad(
                    action_tensor,
                    ((0, self._action_horizon - action_count),)
                    + ((0, 0),) * (action_tensor.ndim - 1),
                    mode="edge" if self._padding_type == PaddingType.REPEAT_LAST else "constant",
                )
            actions.append(action_tensor.astype(self._dtype))

        prioperception: list[Float[jax.Array, " State"]] = []
        for feature in self._prio_features:
            feat = feature.name
            if (prio := feature_row.get(feat)) is None:
                raise RuntimeError(f"Failed to load prio feature {feat} for frame {idx}.")
            if not isinstance(prio, list):
                prio = [prio]
            prio_tensor = jnp.asarray(prio, dtype=jnp.promote_types(self._dtype, jnp.float32))
            if feature.normalization is not None:
                prio_tensor = feature.normalization.apply(prio_tensor)
            prioperception.append(prio_tensor.astype(self._dtype))

        observations: list[Float[jax.Array, "Channel Height Width"]] = []
        video_metadata = {meta.feature_name: meta for meta in episode.video_metadata}
        for feature in self._observation_features:
            feat = feature.name
            if feature.dtype == "image":
                if not isinstance(image_struct := feature_row.get(feat), dict):
                    raise RuntimeError(f"Failed to load image feature {feat} for frame {idx}.")
                if not isinstance(img_bytes := image_struct.get("bytes"), bytes):
                    raise RuntimeError(f"Failed to load image bytes for feature {feat}.")
                source_image = Image.open(BytesIO(img_bytes))
            else:
                if (meta := video_metadata.get(feat)) is None:
                    raise RuntimeError(f"Missing video metadata for feature {feat} in episode {episode.episode_index}.")
                timestamp = meta.from_timestamp + frame_timestamp
                if not meta.from_timestamp <= timestamp < meta.to_timestamp:
                    raise RuntimeError(f"Video timestamp {timestamp} is outside episode {episode.episode_index} for feature {feat}.")
                assert isinstance(self._video_path_template, str)
                video_file = self._video_path_template.format(
                    video_key=feat,
                    chunk_index=meta.chunk_idx,
                    file_index=meta.file_idx,
                )
                source_image = _read_video_frame(self._path / video_file, timestamp)

            with source_image as image:
                image = ImageOps.pad(
                    image.convert("RGB"),
                    self._resolution,
                    method=Image.Resampling.BILINEAR,
                    color=(0, 0, 0),
                )
                image_tensor = (jnp.asarray(image, dtype=jnp.promote_types(self._dtype, jnp.float32)) / 255).transpose(2, 0, 1)
                lower, upper = self._observation_range
                image_tensor = image_tensor * (upper - lower) + lower
                observations.append(image_tensor.astype(self._dtype))
        return (
            jnp.stack(observations, axis=0) if observations else jnp.empty((0, 3, self._resolution[1], self._resolution[0]), self._dtype),
            jnp.concatenate(actions, axis=-1) if actions else jnp.empty((self._action_horizon, 0), self._dtype),
            jnp.concatenate(prioperception, axis=-1) if prioperception else jnp.empty((0,), self._dtype),
            task,
            feature_row["timestamp"],
        )
