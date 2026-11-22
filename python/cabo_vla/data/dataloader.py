import json
import multiprocessing as mp
import random
import traceback
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import suppress
from copy import copy
from dataclasses import dataclass, replace
from enum import Enum
from io import BytesIO
from multiprocessing.connection import Connection
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path

import av
import jax
import jax.numpy as jnp
import numpy as np
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
class Batch[ArrayType]:
    observations: Float[ArrayType, "Batch Camera Channel Height Width"]
    instruction: Int[ArrayType, "Batch Seq"]
    action: Float[ArrayType, "Batch Horizon Action"]
    prio: Float[ArrayType, "Batch State"]
    timestamp: Float[ArrayType, "Batch"]


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
    offset: np.ndarray
    divisor: np.ndarray
    quantile: bool

    def apply(self, value: np.ndarray):
        value -= self.offset
        value /= self.divisor
        if self.quantile:
            value *= 2
            value -= 1


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
    seq_len: int
    tokenizer: Callable[[str], Int[jax.Array, " seq"]] | None = None
    pad_token_id: int | None = None
    observation_range: tuple[float, float] = (0.0, 1.0)
    padding_type: PaddingType = PaddingType.REPEAT_LAST
    dtype: jax.typing.DTypeLike = jnp.float32
    split: str | tuple[int, ...] = "train"
    shuffle: bool = True
    seed: int = 429
    episode_indices: tuple[int, ...] | None = None
    num_workers: int = 1
    container_pool_size: int = 32


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


def _open_video(path: Path) -> av.container.InputContainer:
    container = av.open(str(path), mode="r")
    for stream in container.streams.video:
        stream.thread_count = 1
    return container


def _read_video_frame(container: av.container.InputContainer, timestamp: float) -> Image.Image:
    if not container.streams.video:
        raise RuntimeError(f"No video stream found in {container.name}.")
    stream = container.streams.video[0]
    if stream.time_base is None or stream.average_rate is None:
        raise RuntimeError(f"Missing video timing information in {container.name}.")
    tolerance = 0.5 / float(stream.average_rate) + 1e-6
    container.seek(int(timestamp / stream.time_base), stream=stream)
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        if abs(frame.time - timestamp) <= tolerance:
            return frame.to_image()
        if frame.time > timestamp + tolerance:
            break
    raise RuntimeError(f"No video frame found at {timestamp:.6f}s in {container.name}.")


def _parse_normalization(feature: _DatasetFeature, mode: NormalizationType, stats: dict) -> _Normalization | None:
    if mode == NormalizationType.NONE:
        return None
    keys = ("mean", "std") if mode == NormalizationType.MEANVAR else ("q01", "q99")
    values = []
    for key in keys:
        try:
            value = np.asarray(stats[feature.name][key], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing or invalid {key} statistics for {feature.name}.") from error
        if value.shape != tuple(feature.shape) or not bool(np.all(np.isfinite(value))):
            raise ValueError(f"Expected finite {key} statistics with shape {feature.shape} for {feature.name}.")
        values.append(value)
    offset, upper_or_std = values
    quantile = mode == NormalizationType.QUANTILE
    divisor = upper_or_std - offset if quantile else upper_or_std
    if bool(np.any(divisor < 0)) or not bool(np.all(np.isfinite(divisor))):
        raise ValueError(f"Invalid normalization scale for {feature.name}.")
    # Constant components have no usable scale. Subtract their offset without
    # amplifying numerical noise or dividing by zero.
    divisor = np.where(divisor == 0, 1, divisor)
    return _Normalization(offset, divisor, quantile)


class _DecoderCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.containers: OrderedDict[Path, av.container.InputContainer] = OrderedDict()

    def open(self, path: Path) -> av.container.InputContainer:
        if path in self.containers:
            self.containers.move_to_end(path)
            return self.containers[path]
        if len(self.containers) >= self.capacity:
            _, container = self.containers.popitem(last=False)
            container.close()
        container = _open_video(path)
        self.containers[path] = container
        return container

    def close(self) -> None:
        for container in self.containers.values():
            container.close()
        self.containers.clear()


def _batch_buffers(memory: SharedMemory, layout: np.dtype) -> list[Batch[np.ndarray]]:
    slots = np.ndarray((2,), dtype=layout, buffer=memory.buf)
    names = layout.names
    assert names is not None
    return [Batch(**{name: slots[name][slot] for name in names}) for slot in range(len(slots))]


def _process_worker(loader: "Dataloader", name: str, layout: np.dtype, connection: Connection) -> None:
    loader._decoder_cache = _DecoderCache(loader._config.container_pool_size)
    try:
        memory = SharedMemory(name=name)
        try:
            buffers = _batch_buffers(memory, layout)
            while (job := connection.recv()) is not None:
                slot, indices = job
                for batch_index, sample_index in indices:
                    loader._fetch_single(sample_index, buffers[slot], batch_index)
                connection.send(None)
        finally:
            memory.close()
    except Exception:
        with suppress(BrokenPipeError, EOFError):
            connection.send(traceback.format_exc())
    finally:
        connection.close()
        loader._decoder_cache.close()


class Dataloader:
    _decoder_cache: _DecoderCache

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
        if config.container_pool_size <= 0:
            raise ValueError("container_pool_size must be positive")
        if config.seq_len < 0:
            raise ValueError("seq_len must be nonnegative")
        if config.seq_len > 0 and (config.tokenizer is None or config.pad_token_id is None):
            raise ValueError("tokenizer and pad_token_id are required when seq_len is nonzero")
        self._config = config
        self._rng = random.Random(config.seed)
        if rng_state is not None:
            self._rng.setstate(rng_state)
        if not jnp.issubdtype(config.dtype, jnp.floating):
            raise ValueError("dtype must be floating point for normalized observations")

        info_file = config.path / "meta" / "info.json"
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
            with open(config.path / "meta" / "stats.json") as f:
                stats = json.load(f)
            if not isinstance(stats, dict):
                raise ValueError("Expected meta/stats.json to contain a JSON dict.")

        def numeric_features(specs: tuple[tuple[str, NormalizationType], ...], label: str) -> list[_DatasetFeature]:
            result = []
            for name, mode in specs:
                if not isinstance(feature_dict := features.get(name), dict):
                    raise RuntimeError(f"Failed to find feature {name} in dataset")
                feature = _parse_dataset_feature(name, feature_dict, mode, stats)
                if feature.dtype in {"image", "video", "string", "large_string"}:
                    raise RuntimeError(f"{label} feature {name} has unsupported dtype {feature.dtype}")
                result.append(feature)
            return result

        self._action_features = numeric_features(config.action_features, "Action")
        self._prio_features = numeric_features(config.prio_features, "Prio")

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
                config.path,
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

        tasks: list[str] = pq.read_table(config.path / "meta" / "tasks.parquet").to_pandas().index.tolist()

        if order_buffer is None:
            self._order_buffer = [
                idx for episode in sorted(self._episodes, key=lambda ep: ep.data_from_index) for idx in range(episode.data_from_index, episode.data_to_index)
            ]
            if config.shuffle:
                self._rng.shuffle(self._order_buffer)
        else:
            self._order_buffer = order_buffer.copy()
        if not 0 <= current_position <= len(self._order_buffer):
            raise ValueError("current_position must be within the order buffer")
        self.current_position = current_position

        if config.num_workers < 1:
            raise ValueError("num_workers must be positive")

        pad_token_id = config.pad_token_id if config.pad_token_id is not None else 0
        self._task_tokens = np.full((len(tasks), config.seq_len), pad_token_id, dtype=np.int32)
        if config.seq_len:
            assert config.tokenizer is not None
            for task, buffer in zip(tasks, self._task_tokens, strict=True):
                tokens = np.asarray(config.tokenizer(task), dtype=np.int32)[: config.seq_len]
                buffer[: len(tokens)] = tokens

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
        return (len(self._order_buffer) + self._config.batch_size - 1) // self._config.batch_size

    def __iter__(self) -> Iterator[Batch[jax.Array]]:
        if self.current_position < len(self._order_buffer):
            yield from self._iter_batches()
        self.current_position = 0
        if self._config.shuffle:
            self._rng.shuffle(self._order_buffer)

    def _iter_batches(self) -> Iterator[Batch[jax.Array]]:
        config = self._config
        action_prefix = (self._action_features[0].shape or (1,))[:-1] if self._action_features else ()
        action_dim = sum((feature.shape or (1,))[-1] for feature in self._action_features)
        prio_prefix = (self._prio_features[0].shape or (1,))[:-1] if self._prio_features else ()
        prio_dim = sum((feature.shape or (1,))[-1] for feature in self._prio_features)
        work_dtype = np.dtype(jnp.promote_types(config.dtype, jnp.float32))
        width, height = config.resolution
        layout = np.dtype(
            [
                ("observations", work_dtype, (config.batch_size, len(self._observation_features), 3, height, width)),
                ("instruction", np.int32, (config.batch_size, config.seq_len)),
                ("action", work_dtype, (config.batch_size, config.action_horizon, *action_prefix, action_dim)),
                ("prio", work_dtype, (config.batch_size, *prio_prefix, prio_dim)),
                ("timestamp", np.float32, (config.batch_size,)),
            ],
            align=True,
        )

        # Workers receive the precomputed tokens without the tokenizer.
        worker_loader = copy(self)
        worker_loader._config = replace(config, tokenizer=None)
        worker_loader._order_buffer = []

        context = mp.get_context("spawn")
        worker_count = min(config.num_workers, config.batch_size)
        workers = []
        completed = False
        memory = SharedMemory(create=True, size=2 * layout.itemsize)
        try:
            buffers = _batch_buffers(memory, layout)
            for _ in range(worker_count):
                parent, child = context.Pipe()
                process = context.Process(target=_process_worker, args=(worker_loader, memory.name, layout, child), daemon=True)
                with child:
                    try:
                        process.start()
                    except BaseException:
                        parent.close()
                        raise
                workers.append((process, parent))

            starts = range(self.current_position, len(self._order_buffer), config.batch_size)

            def dispatch(batch_number: int) -> None:
                start = starts[batch_number]
                indices = list(enumerate(self._order_buffer[start : start + config.batch_size]))
                for worker_index, (_, connection) in enumerate(workers):
                    connection.send((batch_number % 2, indices[worker_index::worker_count]))

            for batch_number in range(min(2, len(starts))):
                dispatch(batch_number)
            for batch_number, start in enumerate(starts):
                for process, connection in workers:
                    try:
                        error = connection.recv()
                    except (EOFError, ConnectionError) as exc:
                        raise RuntimeError(f"Dataloader worker {process.pid} exited before completing a batch.") from exc
                    if error is not None:
                        raise RuntimeError(f"Dataloader worker failed:\n{error}")

                size = min(config.batch_size, len(self._order_buffer) - start)
                buffer = buffers[batch_number % 2]
                batch = Batch(
                    observations=jnp.array(buffer.observations[:size], dtype=config.dtype, copy=True),
                    instruction=jnp.array(buffer.instruction[:size], copy=True),
                    action=jnp.array(buffer.action[:size], dtype=config.dtype, copy=True),
                    prio=jnp.array(buffer.prio[:size], dtype=config.dtype, copy=True),
                    timestamp=jnp.array(buffer.timestamp[:size], copy=True),
                )
                # Wait for device copies before letting workers overwrite this slot.
                jax.block_until_ready(batch)
                if batch_number + 2 < len(starts):
                    dispatch(batch_number + 2)
                self.current_position = start + size
                yield batch
            completed = True
        finally:
            for process, connection in workers:
                if completed:
                    with suppress(BrokenPipeError, EOFError):
                        connection.send(None)
                elif process.is_alive():
                    process.terminate()
                connection.close()
            for process, _ in workers:
                process.join(timeout=5)
                if process.is_alive():
                    process.kill()
                    process.join()
                process.close()
            memory.close()
            memory.unlink()

    def _fetch_single(self, idx: int, batch_buffer: Batch[np.ndarray], batch_index: int) -> None:
        episode = next((ep for ep in self._episodes if ep.data_from_index <= idx < ep.data_to_index), None)
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
        feature_row = {name: values[0] for name, values in _read_row(self._config.path / feature_file, idx, columns=non_action_features).items()}
        action_count = min(self._config.action_horizon, episode.data_to_index - idx)
        action_rows = _read_row(
            self._config.path / feature_file,
            idx,
            columns=action_features,
            num_rows=action_count,
        )

        if not isinstance(task_id := feature_row.get("task_index"), int):
            raise RuntimeError(f"Expected integral task_id for frame {idx}.")

        frame_timestamp = feature_row.get("timestamp")
        if isinstance(frame_timestamp, list) and len(frame_timestamp) == 1:
            frame_timestamp = frame_timestamp[0]
        if not isinstance(frame_timestamp, (int, float)):
            raise RuntimeError(f"Expected scalar timestamp for frame {idx}.")
        frame_timestamp = float(frame_timestamp)

        batch_buffer.instruction[batch_index] = self._task_tokens[task_id]
        batch_buffer.timestamp[batch_index] = frame_timestamp

        actions = batch_buffer.action[batch_index]
        start_dim = 0
        for feature in self._action_features:
            feat = feature.name
            if not isinstance(action := action_rows.get(feat), list):
                raise RuntimeError(f"Failed to load action feature {feat} for frame {idx}.")
            feature_length = (feature.shape or (1,))[-1]
            values = np.asarray(action)
            if values.ndim == 1:
                values = values[:, None]
            view = actions[:action_count, ..., start_dim : start_dim + feature_length]
            view[:] = values
            if feature.normalization is not None:
                feature.normalization.apply(view)
            start_dim += feature_length
        if action_count < self._config.action_horizon:
            match self._config.padding_type:
                case PaddingType.REPEAT_LAST:
                    actions[action_count:] = actions[action_count - 1]
                case PaddingType.ZERO:
                    actions[action_count:] = 0

        start_dim = 0
        for feature in self._prio_features:
            feat = feature.name
            if (prio := feature_row.get(feat)) is None:
                raise RuntimeError(f"Failed to load prio feature {feat} for frame {idx}.")
            feature_length = (feature.shape or (1,))[-1]
            view = batch_buffer.prio[batch_index, ..., start_dim : start_dim + feature_length]
            view[:] = prio
            if feature.normalization is not None:
                feature.normalization.apply(view)
            start_dim += feature_length

        video_metadata = {meta.feature_name: meta for meta in episode.video_metadata}
        for camera_index, feature in enumerate(self._observation_features):
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
                container = self._decoder_cache.open(self._config.path / video_file)
                source_image = _read_video_frame(container, timestamp)

            with source_image as image:
                image = ImageOps.pad(
                    image.convert("RGB"),
                    self._config.resolution,
                    method=Image.Resampling.BILINEAR,
                    color=(0, 0, 0),
                )
                view = batch_buffer.observations[batch_index, camera_index]
                view[:] = np.asarray(image).transpose(2, 0, 1)
                view /= 255
                lower, upper = self._config.observation_range
                view *= upper - lower
                view += lower
