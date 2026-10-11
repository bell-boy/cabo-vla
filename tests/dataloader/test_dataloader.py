import json
from io import BytesIO
from pathlib import Path

import av
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from cabo_vla.data.dataloader import Dataloader, NormalizationType

EPISODE_LENGTHS = (4, 3)
TASKS = ("pick up", "put down")
BATCH_SIZE = 2
HORIZON = 3
ROW_GROUP_SIZE = 2
WIDTH, HEIGHT, CHANNELS = 8, 4, 3
FPS = 10
KEYFRAME_INTERVAL = 3
IMAGE_CAMERA = "observation.images.wrist"
VIDEO_CAMERA = "observation.images.front"
DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"

# Distinct feature ranges and component values reveal swaps and truncation.
# Scalars remain scalars on disk; vectors are stored as lists.
NUMERIC_FEATURES = {
    "action.arm": (100, 4),
    "action.gripper": (200, 1),
    "state.joints": (300, 5),
    "state.force": (400, 1),
}
FRAME_STRIDE = 10
CAMERA_BASES = {IMAGE_CAMERA: 20, VIDEO_CAMERA: 120}
PIXEL_FRAME_STRIDE = 10
PIXEL_CHANNEL_STRIDE = 20

# Hand-checked oracle for lengths (4, 3) and horizon 3. Do not compute this
# with the loader's episode/window algorithm: it should catch bugs there.
EXPECTED_WINDOWS = np.array([[0, 1, 2], [1, 2, 3], [4, 5, 6]])
EXPECTED_STARTS = EXPECTED_WINDOWS[:, 0]
EXPECTED_TASKS = [TASKS[0], TASKS[0], TASKS[1]]


def write_video(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True)
    with av.open(str(path), "w") as container:
        # Lossless RGB encoding allows exact pixel assertions after decoding.
        stream = container.add_stream("libx264rgb", rate=FPS)
        stream.width, stream.height = WIDTH, HEIGHT
        stream.pix_fmt = "rgb24"
        stream.options = {"crf": "0", "g": str(KEYFRAME_INTERVAL)}
        for pixels in frames:
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_tiny_dataset(root: Path) -> dict[str, np.ndarray]:
    """Write real files and retain their source arrays as expected-value inputs."""
    frame_count = sum(EPISODE_LENGTHS)
    frame_ids = np.arange(frame_count)
    source = {
        name: (base + FRAME_STRIDE * frame_ids[:, None] + np.arange(width)).astype(
            np.float32
        )
        for name, (base, width) in NUMERIC_FEATURES.items()
    }
    for name, base in CAMERA_BASES.items():
        # Vary frame, channel, row and column: even transposes and flips show up.
        source[name] = (
            base
            + PIXEL_FRAME_STRIDE * frame_ids[:, None, None, None]
            + np.arange(HEIGHT)[None, :, None, None]
            + np.arange(WIDTH)[None, None, :, None]
            + PIXEL_CHANNEL_STRIDE * np.arange(CHANNELS)[None, None, None, :]
        ).astype(np.uint8)

    metadata_path = root / "meta/episodes/chunk-000/file-000.parquet"
    data_path = root / DATA_PATH.format(chunk_index=0, file_index=0)
    metadata_path.parent.mkdir(parents=True)
    data_path.parent.mkdir(parents=True)
    features = {
        name: {"dtype": "float32", "shape": [width]}
        for name, (_, width) in NUMERIC_FEATURES.items()
    }
    features.update(
        {
            IMAGE_CAMERA: {"dtype": "image", "shape": [HEIGHT, WIDTH, CHANNELS]},
            VIDEO_CAMERA: {"dtype": "video", "shape": [HEIGHT, WIDTH, CHANNELS]},
        }
    )
    (root / "meta/info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "features": features,
                "splits": {"train": f"0:{len(EPISODE_LENGTHS)}"},
                "data_path": DATA_PATH,
                "video_path": VIDEO_PATH,
            }
        )
    )
    pd.DataFrame({"task_index": range(len(TASKS))}, index=np.asarray(TASKS)).to_parquet(
        root / "meta/tasks.parquet"
    )

    episodes, task_indices, timestamps = [], [], []
    start = 0
    for episode_index, length in enumerate(EPISODE_LENGTHS):
        stop = start + length
        episodes.append(
            {
                "episode_index": episode_index,
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": start,
                "dataset_to_index": stop,
                f"videos/{VIDEO_CAMERA}/chunk_index": 0,
                f"videos/{VIDEO_CAMERA}/file_index": 0,
                f"videos/{VIDEO_CAMERA}/from_timestamp": start / FPS,
                f"videos/{VIDEO_CAMERA}/to_timestamp": stop / FPS,
            }
        )
        task_indices.extend([episode_index] * length)
        # Frame timestamps restart each episode, but the video file is shared.
        timestamps.extend(np.arange(length) / FPS)
        start = stop
    pq.write_table(pa.Table.from_pylist(episodes), metadata_path)

    images = []
    for pixels in source[IMAGE_CAMERA]:
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, format="PNG")
        images.append({"bytes": buffer.getvalue(), "path": None})
    columns = {
        name: pa.array(
            values[:, 0] if values.shape[1] == 1 else values.tolist(),
            type=pa.float32() if values.shape[1] == 1 else pa.list_(pa.float32()),
        )
        for name, values in source.items()
        if name in NUMERIC_FEATURES
    }
    columns.update(
        {
            "task_index": pa.array(task_indices),
            "timestamp": pa.array(timestamps),
            IMAGE_CAMERA: pa.array(images),
        }
    )
    pq.write_table(pa.table(columns), data_path, row_group_size=ROW_GROUP_SIZE)
    write_video(
        root / VIDEO_PATH.format(video_key=VIDEO_CAMERA, chunk_index=0, file_index=0),
        source[VIDEO_CAMERA],
    )
    return source


@pytest.fixture
def tiny_dataset(tmp_path):
    return tmp_path, write_tiny_dataset(tmp_path)


def make_loader(path, *, actions, states, cameras=(), dtype=jnp.float32):
    return Dataloader(
        path,
        resolution=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        action_horizon=HORIZON,
        observation_features=list(cameras),
        action_features=[(name, NormalizationType.NONE) for name in actions],
        prio_features=[(name, NormalizationType.NONE) for name in states],
        shuffle=False,
        dtype=dtype,
    )


@pytest.mark.parametrize(
    ("actions", "states"),
    [
        pytest.param(
            ["action.arm", "action.gripper"],
            ["state.joints", "state.force"],
            id="all-features",
        ),
        pytest.param(
            ["action.gripper", "action.arm"],
            ["state.force", "state.joints"],
            id="reversed-feature-order",
        ),
        pytest.param(["action.arm"], ["state.joints"], id="vectors-only"),
        pytest.param(["action.gripper"], ["state.force"], id="scalars-only"),
    ],
)
def test_action_windows_and_numeric_feature_selection(tiny_dataset, actions, states):
    path, source = tiny_dataset
    batches = list(make_loader(path, actions=actions, states=states))

    assert len(batches) == 2
    first, last = batches
    action_width = sum(NUMERIC_FEATURES[name][1] for name in actions)
    state_width = sum(NUMERIC_FEATURES[name][1] for name in states)
    assert first.action.shape == (BATCH_SIZE, HORIZON, action_width)
    assert last.action.shape == (1, HORIZON, action_width)
    assert first.prio.shape == (BATCH_SIZE, state_width)
    assert last.prio.shape == (1, state_width)

    # Index the original arrays using the explicitly enumerated valid windows.
    expected_actions = np.concatenate(
        [source[name][EXPECTED_WINDOWS] for name in actions], axis=-1
    )
    expected_states = np.concatenate(
        [source[name][EXPECTED_STARTS] for name in states], axis=-1
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.action for batch in batches]), expected_actions
    )
    np.testing.assert_array_equal(
        np.concatenate([batch.prio for batch in batches]), expected_states
    )
    assert [task for batch in batches for task in batch.instruction] == EXPECTED_TASKS
    for batch in batches:
        assert batch.action.dtype == np.float32
        assert batch.prio.dtype == np.float32


@pytest.mark.parametrize(
    "cameras",
    [
        pytest.param([IMAGE_CAMERA], id="image-only"),
        pytest.param([VIDEO_CAMERA], id="video-only"),
        pytest.param([IMAGE_CAMERA, VIDEO_CAMERA], id="image-then-video"),
        pytest.param([VIDEO_CAMERA, IMAGE_CAMERA], id="video-then-image"),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float16])
def test_camera_pixels_selection_and_order(tiny_dataset, cameras, dtype):
    path, source = tiny_dataset
    batches = list(
        make_loader(
            path,
            actions=["action.arm"],
            states=["state.joints"],
            cameras=cameras,
            dtype=dtype,
        )
    )
    actual = np.concatenate([batch.observations for batch in batches])
    expected = np.stack(
        [source[name][EXPECTED_STARTS].transpose(0, 3, 1, 2) for name in cameras],
        axis=1,
    )
    assert actual.shape == (len(EXPECTED_STARTS), len(cameras), CHANNELS, HEIGHT, WIDTH)
    assert actual.dtype == dtype
    expected = (expected.astype(np.float32) / 255).astype(dtype)
    np.testing.assert_allclose(actual, expected, rtol=float(np.finfo(dtype).eps))
    # Check that image/video samples are paired with the right numeric frame.
    np.testing.assert_array_equal(
        np.concatenate([batch.prio for batch in batches]),
        source["state.joints"][EXPECTED_STARTS],
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_mixed_normalization_preserves_values_order_and_dtype(tiny_dataset, dtype):
    path, source = tiny_dataset
    # Different per-component scales catch accidental scalar broadcasting.
    mean = source["action.arm"][0]
    std = np.arange(1, mean.size + 1, dtype=np.float32)
    lower = source["state.joints"][1]
    upper = source["state.joints"][3]
    stats_path = path / "meta/stats.json"
    stats_path.write_text(
        json.dumps(
            {
                "action.arm": {"mean": mean.tolist(), "std": std.tolist()},
                "state.joints": {"q01": lower.tolist(), "q99": upper.tolist()},
            }
        )
    )
    loader = Dataloader(
        path,
        resolution=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        action_horizon=HORIZON,
        observation_features=[],
        action_features=[
            ("action.gripper", NormalizationType.NONE),
            ("action.arm", NormalizationType.MEANVAR),
        ],
        prio_features=[
            ("state.joints", NormalizationType.QUANTILE),
            ("state.force", NormalizationType.NONE),
        ],
        dtype=dtype,
        shuffle=False,
    )
    batches = list(loader)
    expected_actions = np.concatenate(
        [
            source["action.gripper"][EXPECTED_WINDOWS],
            (source["action.arm"][EXPECTED_WINDOWS] - mean) / std,
        ],
        axis=-1,
    )
    expected_states = np.concatenate(
        [
            2 * (source["state.joints"][EXPECTED_STARTS] - lower) / (upper - lower) - 1,
            source["state.force"][EXPECTED_STARTS],
        ],
        axis=-1,
    )
    # Includes values beyond the quantiles: the transform does not clip.
    for field, expected in (("action", expected_actions), ("prio", expected_states)):
        actual = np.concatenate([getattr(batch, field) for batch in batches])
        assert actual.dtype == dtype
        np.testing.assert_allclose(
            actual.astype(np.float32),
            np.asarray(jnp.asarray(expected, dtype=dtype), dtype=np.float32),
            rtol=1e-6,
        )
    assert batches[0].observations.dtype == dtype


@pytest.mark.parametrize(
    "mode,stats",
    [
        (NormalizationType.MEANVAR, {"mean": [100], "std": [0]}),
        (NormalizationType.QUANTILE, {"q01": [100], "q99": [100]}),
    ],
)
def test_zero_normalization_scale_uses_unit_divisor(tiny_dataset, mode, stats):
    path, source = tiny_dataset
    (path / "meta/stats.json").write_text(json.dumps({"action.gripper": stats}))
    loader = Dataloader(
        path,
        resolution=(WIDTH, HEIGHT),
        batch_size=BATCH_SIZE,
        action_horizon=HORIZON,
        observation_features=[],
        action_features=[("action.gripper", mode)],
        prio_features=[],
        shuffle=False,
    )
    actual = np.concatenate([batch.action for batch in loader])
    offset_key = "mean" if mode == NormalizationType.MEANVAR else "q01"
    expected = source["action.gripper"][EXPECTED_WINDOWS] - stats[offset_key]
    if mode == NormalizationType.QUANTILE:
        expected = 2 * expected - 1
    np.testing.assert_array_equal(actual, expected)
