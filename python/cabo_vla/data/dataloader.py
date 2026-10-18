import bisect
import contextlib
import json
import math
import multiprocessing
import queue
import random
import threading
import weakref
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from io import BytesIO
from multiprocessing import shared_memory
from pathlib import Path

import av
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
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
    offset: np.ndarray
    divisor: np.ndarray
    quantile: bool

    def apply(self, value: np.ndarray) -> np.ndarray:
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
    num_workers: int = 0
    # Batches decoded ahead of the consumer. Every batch is split into many small tasks, so a few batches keep all workers busy.
    prefetch_batches: int = 4
    drop_last: bool = False


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


def _load_data_file(path: Path, columns: list[str]) -> tuple[int, pa.Table]:
    table = pq.read_table(path, columns=list(dict.fromkeys(["index", *columns]))).sort_by("index")
    indices = table.column("index").to_numpy()
    if len(indices) and not np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices))):
        raise RuntimeError(f"Expected contiguous frame indices in {path}.")
    return (int(indices[0]) if len(indices) else 0), table


_video_containers: OrderedDict[Path, av.container.InputContainer] = OrderedDict()
_MAX_OPEN_VIDEOS = 64


def _open_video(path: Path) -> av.container.InputContainer:
    # Opening a container costs about as much as decoding the frame itself, so keep recently used files open (per process).
    if (container := _video_containers.pop(path, None)) is None:
        container = av.open(str(path))
        try:
            if not container.streams.video:
                raise RuntimeError(f"No video stream found in {path}.")
            stream = container.streams.video[0]
            # Parallelism comes from the loader's worker processes; codec-level threads (dav1d defaults to one per core) oversubscribe.
            stream.codec_context.thread_count = 1
            if stream.time_base is None or stream.average_rate is None:
                raise RuntimeError(f"Missing video timing information in {path}.")
        except Exception:
            container.close()
            raise
        while len(_video_containers) >= _MAX_OPEN_VIDEOS:
            _video_containers.popitem(last=False)[1].close()
    _video_containers[path] = container
    return container


def _read_video_frame(path: Path, timestamp: float) -> Image.Image:
    container = _open_video(path)
    stream = container.streams.video[0]
    tolerance = 0.5 / float(stream.average_rate) + 1e-6
    try:
        container.seek(int(timestamp / stream.time_base), stream=stream)
        for frame in container.decode(stream):
            if frame.time is None:
                continue
            if abs(frame.time - timestamp) <= tolerance:
                return frame.to_image()
            if frame.time > timestamp + tolerance:
                break
    except Exception:
        # The decoder state is unknown after a failure; reopen the file next time.
        _video_containers.pop(path, None)
        container.close()
        raise
    raise RuntimeError(f"No video frame found at {timestamp:.6f}s in {path}.")


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
    divisor = np.where(divisor == 0, 1, divisor).astype(np.float32)
    return _Normalization(offset, divisor, quantile)


_worker_loader: "Dataloader | None" = None
_worker_shm: shared_memory.SharedMemory | None = None
_worker_ring: np.ndarray | None = None


def _init_worker(loader: "Dataloader", shm_name: str, ring_shape: tuple[int, ...]) -> None:
    global _worker_loader, _worker_shm, _worker_ring
    _worker_loader = loader
    _worker_shm = shared_memory.SharedMemory(name=shm_name)
    _worker_ring = np.ndarray(ring_shape, dtype=np.uint8, buffer=_worker_shm.buf)


def _worker_fetch_chunk(slot: int, start: int, indices: list[int]) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    assert _worker_loader is not None and _worker_ring is not None
    return _worker_loader._fetch_chunk(indices, _worker_ring[slot, start : start + len(indices)])


def _release_pool(pool: ProcessPoolExecutor, shm: shared_memory.SharedMemory) -> None:
    pool.shutdown(wait=False, cancel_futures=True)
    shm.unlink()
    with contextlib.suppress(BufferError):
        shm.close()


class Dataloader:
    def __init__(
        self,
        config: DataloaderConfig,
        *,
        # Applied to every batch array, so it should only partition the leading (batch) axis. None means the default device.
        # Not part of the config: device handles don't survive pickling a saved state.
        sharding: jax.sharding.Sharding | None = None,
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
        self._num_workers = config.num_workers
        self._prefetch_batches = config.prefetch_batches
        self._drop_last = config.drop_last
        self._sharding = sharding
        self._np_dtype = np.dtype(config.dtype)
        self._compute_dtype = np.promote_types(self._np_dtype, np.float32)
        # A lookup table computed on the host: XLA rewrites x / 255 as x * (1 / 255), which misses the range endpoints by an ulp.
        lower, upper = config.observation_range
        self._pixel_table = (np.arange(256, dtype=self._compute_dtype) / 255 * (upper - lower) + lower).astype(self._np_dtype)
        self._data_files: dict[Path, tuple[int, pa.Table]] = {}
        self._pool: ProcessPoolExecutor | None = None
        self._shm: shared_memory.SharedMemory | None = None
        self._inflight: list[Future] = []
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

        self._episodes.sort(key=lambda ep: ep.data_from_index)
        self._episode_starts = [ep.data_from_index for ep in self._episodes]
        self._data_columns = (
            [x.name for x in self._observation_features if x.dtype == "image"]
            + [x.name for x in self._prio_features]
            + [x.name for x in self._action_features]
            + ["task_index", "timestamp"]
        )
        self._observation_shape = (len(self._observation_features), 3, self._resolution[1], self._resolution[0])

        if order_buffer is None:
            self._order_buffer = [idx for episode in self._episodes for idx in range(episode.data_from_index, episode.data_to_index)]
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
    def from_state(cls, state: DataloaderState, *, sharding: jax.sharding.Sharding | None = None) -> "Dataloader":
        return cls(
            state.config,
            sharding=sharding,
            rng_state=state.rng_state,
            order_buffer=state.order_buffer,
            current_position=state.current_position,
        )

    def __len__(self) -> int:
        n = len(self._order_buffer)
        return n // self._batch_size if self._drop_last else -(-n // self._batch_size)

    def __iter__(self) -> Iterator[Batch]:
        n = len(self._order_buffer)
        end = n - n % self._batch_size if self._drop_last else n
        batches = [self._order_buffer[start : start + self._batch_size] for start in range(self.current_position, end, self._batch_size)]
        if self._num_workers == 0:
            for indices in batches:
                observations = np.empty((len(indices), *self._observation_shape), dtype=np.uint8)
                batch = self._to_batch(observations, *self._fetch_chunk(indices, observations))
                self.current_position += len(indices)
                yield batch
        else:
            for batch in self._iter_parallel(batches):
                self.current_position += len(batch.instruction)
                yield batch
        self.current_position = 0
        if self._shuffle:
            self._rng.shuffle(self._order_buffer)

    def _iter_parallel(self, batches: list[list[int]]) -> Iterator[Batch]:
        # Workers write decoded pixels straight into a shared-memory ring (one slot per in-flight batch) and return only the small
        # arrays through the pool. Sending images through the result pipe tops out well below 1 GB/s, far slower than the decoding.
        slots = self._prefetch_batches + 1
        ring_shape = (slots, self._batch_size, *self._observation_shape)
        if self._pool is None or self._shm is None:
            self._shm = shared_memory.SharedMemory(create=True, size=max(1, math.prod(ring_shape)))
            # spawn, not fork: forking a process that has initialized JAX can deadlock.
            self._pool = ProcessPoolExecutor(
                self._num_workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker, initargs=(self, self._shm.name, ring_shape)
            )
            # Workers outlive an epoch: spawning them re-imports the training script, which takes tens of seconds.
            weakref.finalize(self, _release_pool, self._pool, self._shm)
        pool, shm = self._pool, self._shm
        # Tasks left over from an interrupted iteration still write into the ring; let them finish before reusing the slots.
        wait(self._inflight)
        self._inflight = []
        # Small tasks spread every batch over all workers, so batches finish in a steady stream instead of one wave per worker.
        chunk_size = max(1, -(-self._batch_size // (4 * self._num_workers)))
        pending: deque[tuple[int, list[Future]]] = deque()
        # A background thread copies finished batches to the device so the transfer overlaps the train step.
        ready: queue.Queue = queue.Queue(maxsize=2)
        stop = threading.Event()

        def put(item) -> bool:
            while not stop.is_set():
                try:
                    ready.put(item, timeout=0.1)
                    return True
                except queue.Full:
                    pass
            return False

        def submit(batch_index: int, indices: list[int]) -> tuple[int, list[Future]]:
            slot = batch_index % slots
            return slot, [pool.submit(_worker_fetch_chunk, slot, start, indices[start : start + chunk_size]) for start in range(0, len(indices), chunk_size)]

        def collate(slot: int, futures: list[Future]) -> Batch:
            results = [future.result() for future in futures]
            instruction = [task for result in results for task in result[2]]
            # A fresh view per batch: the segment can only be closed once no view of it is alive.
            observations = np.ndarray(ring_shape, dtype=np.uint8, buffer=shm.buf)[slot, : len(instruction)]
            action, prio, timestamp = (np.concatenate([result[i] for result in results], axis=0) for i in (0, 1, 3))
            return self._to_batch(observations, action, prio, instruction, timestamp)

        def produce():
            try:
                for batch_index, indices in enumerate(batches):
                    pending.append(submit(batch_index, indices))
                    if len(pending) > self._prefetch_batches and not put(collate(*pending.popleft())):
                        return
                while pending:
                    if not put(collate(*pending.popleft())):
                        return
                put(None)
            except BaseException as error:
                put(error)

        producer = threading.Thread(target=produce, daemon=True)
        producer.start()
        try:
            while (item := ready.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            producer.join()
            self._inflight = [future for _, futures in pending for future in futures if not future.cancel()]

    def _fetch_chunk(self, indices: list[int], observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
        samples = [self._fetch_single(idx, observations[i]) for i, idx in enumerate(indices)]
        return (
            np.stack([s[0] for s in samples], axis=0),
            np.stack([s[1] for s in samples], axis=0),
            [s[2] for s in samples],
            np.array([s[3] for s in samples], dtype=np.float32),
        )

    def _to_batch(self, observations: np.ndarray, action: np.ndarray, prio: np.ndarray, instruction: list[str], timestamp: np.ndarray) -> Batch:
        # Wait for the host-to-device copy only (it owns the shared-memory slot); the normalize below is left in flight so the
        # producer can run ahead instead of queueing behind the train step that is executing on the compute stream.
        pixels, device_action, device_prio, device_timestamp = jax.block_until_ready(jax.device_put((observations, action, prio, timestamp), self._sharding))
        # Pixels travel as uint8 and are normalized on the device: a quarter of the host traffic of float32 images.
        return Batch(
            observations=jnp.take(self._pixel_table, pixels),
            instruction=[self._tokenize_instruction(text) for text in instruction],
            action=device_action,
            prio=device_prio,
            timestamp=device_timestamp,
        )

    def __getstate__(self) -> dict:
        # Workers only decode on the host: device handles in a sharding don't survive pickling, and each process fills its own table cache.
        return {**self.__dict__, "_sharding": None, "_data_files": {}, "_pool": None, "_shm": None, "_inflight": []}

    def _data_file(self, path: Path) -> tuple[int, pa.Table]:
        if (cached := self._data_files.get(path)) is None:
            cached = self._data_files[path] = _load_data_file(path, self._data_columns)
        return cached

    def _read_rows(self, path: Path, frame_index: int, columns: list[str], num_rows: int = 1) -> dict[str, list]:
        first_index, table = self._data_file(path)
        rows = table.slice(frame_index - first_index, num_rows).select(columns).to_pydict()
        if frame_index < first_index or any(len(values) != num_rows for values in rows.values()):
            raise IndexError("Frame range not found")
        return rows

    def _tokenize_instruction(self, text: str) -> Int[jax.Array, " seq"]:
        tokens = self._tokenizer(text)[: self._config.seq_len]
        return jnp.pad(
            tokens,
            (0, self._config.seq_len - tokens.shape[0]),
            constant_values=self._config.pad_token_id,
        )

    def _fetch_single(self, idx: int, observations: np.ndarray) -> tuple[np.ndarray, np.ndarray, str, float]:
        position = bisect.bisect_right(self._episode_starts, idx) - 1
        episode = self._episodes[position] if position >= 0 else None
        if episode is None or not episode.data_from_index <= idx < episode.data_to_index:
            raise RuntimeError(f"Index {idx} is out of bounds.")

        image_features = [x.name for x in self._observation_features if x.dtype == "image"]
        prio_features = [x.name for x in self._prio_features]
        action_features = [x.name for x in self._action_features]

        non_action_features = image_features + prio_features + ["task_index", "timestamp"]
        feature_file = self._data_path_template.format(
            chunk_index=episode.data_chunk_index,
            file_index=episode.data_file_index,
        )
        feature_row = {name: values[0] for name, values in self._read_rows(self._path / feature_file, idx, non_action_features).items()}
        action_count = min(self._action_horizon, episode.data_to_index - idx)
        action_rows = self._read_rows(self._path / feature_file, idx, action_features, num_rows=action_count)

        if not isinstance(task_id := feature_row.get("task_index"), int):
            raise RuntimeError(f"Expected integral task_id for frame {idx}.")
        task: str = self._tasks[task_id]

        frame_timestamp = feature_row.get("timestamp")
        if isinstance(frame_timestamp, list) and len(frame_timestamp) == 1:
            frame_timestamp = frame_timestamp[0]
        if not isinstance(frame_timestamp, (int, float)):
            raise RuntimeError(f"Expected scalar timestamp for frame {idx}.")
        frame_timestamp = float(frame_timestamp)

        actions: list[Float[np.ndarray, "Horizon Action"]] = []
        for feature in self._action_features:
            feat = feature.name
            if not isinstance(action := action_rows.get(feat), list):
                raise RuntimeError(f"Failed to load action feature {feat} for frame {idx}.")
            action_tensor = np.asarray(action, dtype=self._compute_dtype)
            if action_tensor.ndim == 1:
                action_tensor = action_tensor[:, None]
            if feature.normalization is not None:
                action_tensor = feature.normalization.apply(action_tensor)
            if action_count < self._action_horizon:
                action_tensor = np.pad(
                    action_tensor,
                    ((0, self._action_horizon - action_count), (0, 0)),
                    mode="edge" if self._padding_type == PaddingType.REPEAT_LAST else "constant",
                )
            actions.append(action_tensor.astype(self._np_dtype))

        prioperception: list[Float[np.ndarray, " State"]] = []
        for feature in self._prio_features:
            feat = feature.name
            if (prio := feature_row.get(feat)) is None:
                raise RuntimeError(f"Failed to load prio feature {feat} for frame {idx}.")
            if not isinstance(prio, list):
                prio = [prio]
            prio_tensor = np.asarray(prio, dtype=self._compute_dtype)
            if feature.normalization is not None:
                prio_tensor = feature.normalization.apply(prio_tensor)
            prioperception.append(prio_tensor.astype(self._np_dtype))

        video_metadata = {meta.feature_name: meta for meta in episode.video_metadata}
        for camera, feature in enumerate(self._observation_features):
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
                observations[camera] = np.asarray(image).transpose(2, 0, 1)
        return (
            np.concatenate(actions, axis=-1) if actions else np.empty((self._action_horizon, 0), self._np_dtype),
            np.concatenate(prioperception, axis=-1) if prioperception else np.empty((0,), self._np_dtype),
            task,
            feature_row["timestamp"],
        )
