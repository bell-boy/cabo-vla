# LeRobot Dataset v3.0: implementation-derived format specification

Reference implementation: Hugging Face LeRobot **release v0.4.4**. Dataset format marker: **`v3.0`**. Prepared 2026-10-09.

This is an unofficial specification reconstructed from the release's reader, writer, migration code, and statistics utilities. It describes the standard Parquet/MP4 representation. It is not a Hugging Face standard or a claim that every dataset labeled v3.0 conforms to the reference writer. Later releases may change behavior without changing the format marker.

**Terminology:** “MUST” below defines consistency requirements for the canonical representation described here; it does not imply that LeRobot validates every requirement. “Writer behavior” describes the reference implementation. “Recommendation” identifies an interoperability choice rather than an enforced format rule.

## 1. Data model

A dataset contains an ordered sequence of episodes. An episode contains an ordered sequence of frames. Each frame is one synchronized sample of the dataset's declared features, with a timestamp, episode-local frame number, global frame number, and task ID.

An episode's frame rows are contiguous in the global dataset and reside in one data Parquet shard in the reference writer. A shard can contain multiple episodes. Each camera's video is stored separately; its shard boundaries need not match the data shards or another camera's shards. Episode metadata connects these files.

The format does not prescribe joint units, coordinate frames, rotation representations, action semantics, or which state/action components a robot must have. An exporter must document those separately. Feature names alone do not establish units or semantics.

## 2. Files and paths

All paths below are relative to the dataset root.

| Artifact | Default path | Purpose |
|---|---|---|
| Dataset descriptor | `meta/info.json` | Schema, counts, FPS, shard path templates |
| Global statistics | `meta/stats.json` | Feature normalization statistics |
| Task table | `meta/tasks.parquet` | Task text to integer ID mapping |
| Episode tables | `meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` | One row per episode |
| Frame tables | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` | One row per frame |
| Video files | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` | Encoded frames for a camera |

`03d` means decimal formatting with a minimum width of three digits. Chunk and file indices start at zero. The reference helper advances the file index until it reaches `chunks_size`, then increments the chunk and resets the file index to zero.

Readers MUST resolve data and video paths from the templates in `info.json` and the episode's locator fields. They MUST NOT derive file numbers from episode IDs. Metadata shards use the fixed episode path pattern in the reference implementation.

The reference defaults are 1,000 files per chunk, a 100 MB target for data shards, and a 200 MB target for video shards. These are writer configuration defaults, not strict maximum sizes: rollover occurs between episodes and uses size estimates.

`meta/tasks.jsonl`, `meta/episodes.jsonl`, and `meta/episodes_stats.jsonl` belong to legacy layouts; they are not the canonical v3.0 outputs described here.

## 3. Dataset descriptor: `meta/info.json`

The descriptor is a UTF-8 JSON object. These are the fields emitted by the reference writer:

| Field | JSON type | Meaning |
|---|---|---|
| `codebase_version` | string | `"v3.0"`; this is the dataset compatibility marker, not the installed package release |
| `robot_type` | string or null | Robot identifier; may be unspecified |
| `total_episodes` | integer | Number of episodes |
| `total_frames` | integer | Number of frame rows across all episodes |
| `total_tasks` | integer | Number of entries in the task table |
| `chunks_size` | integer | Files per chunk before rollover |
| `data_files_size_in_mb` | number | Data shard size target |
| `video_files_size_in_mb` | number | Video shard size target |
| `fps` | integer in reference creation API | Dataset sample rate in frames per second |
| `splits` | object | Split names mapped to episode ranges, such as `{"train":"0:2"}` |
| `data_path` | string | Formattable relative Parquet path |
| `video_path` | string or null | Formattable relative MP4 path; null when videos are disabled |
| `features` | object | Mapping from feature name to feature descriptor |

Split ranges use episode indices, with an inclusive start and exclusive end. The standard writer updates the training split to `0:total_episodes`.

### 3.1 Feature descriptors

| Field | Type | Meaning |
|---|---|---|
| `dtype` | string | Numeric/string type understood by the underlying datasets stack, or the special type `image` or `video` |
| `shape` | array of integers | Logical per-frame shape |
| `names` | array or null | Labels for vector components or visual axes; not a universal unit specification |
| `info` | object, for encoded video | Properties of the encoded stream; the reference video convention is described in section 7, not a universal schema for every feature type |

Feature names MUST NOT contain `/`; dots are permitted. Slash-separated names used in episode metadata are separate from feature names.

**Important scalar rule:** a nonvisual feature with logical shape `[1]` maps to a scalar `datasets.Value`, not a one-element Parquet list. A numeric vector `[D]`, for `D > 1`, maps to a fixed-length sequence. Numeric arrays of rank 2–5 map to the corresponding Hugging Face array extension types. Do not assume every tensor-shaped feature has a plain fixed-size-list physical Arrow representation.

`video` features are declared in `features` but omitted from the frame Parquet schema. `image` features map to `datasets.Image()` and are included in the frame table.

The hardware helper uses visual shapes `[height, width, channels]` with `names: ["height", "width", "channels"]`. Runtime training tensors may use a different axis order. Exporters should preserve the declared shape and axis labels rather than infer them from a training tensor.

#### Element types

For nonvisual features, `dtype` describes each component's element type, while `shape` describes the logical per-frame dimensions. Common Hugging Face `datasets.Value` element types are:

| Category | `dtype` values |
|---|---|
| Boolean | `bool` |
| Signed integer | `int8`, `int16`, `int32`, `int64` |
| Unsigned integer | `uint8`, `uint16`, `uint32`, `uint64` |
| Floating point | `float16`, `float32`, `float64` |
| Text | `string`, `large_string` |

The number indicates the bit width. `float` and `double` are aliases for `float32` and `float64`, respectively. See the [Hugging Face `datasets.Value` documentation](https://huggingface.co/docs/datasets/v2.5.1/en/package_reference/main_classes#datasets.Value).

For example, a six-component state vector has:

```json
{
  "dtype": "float32",
  "shape": [6],
  "names": ["x", "y", "z", "roll", "pitch", "yaw"]
}
```

The underlying library also supports binary, null, date/time, duration, and decimal types. That broader support does not guarantee compatibility with LeRobot's recording, statistics, or training code. LeRobot handles `image` and `video` separately from these element types.

### 3.2 Minimal complete descriptor example

This example describes two episodes with three frames each, one state component, one action component, and no video. Each state/action value is a scalar in Parquet because its declared shape is `[1]`.

```json
{
  "codebase_version": "v3.0",
  "robot_type": null,
  "total_episodes": 2,
  "total_frames": 6,
  "total_tasks": 1,
  "chunks_size": 1000,
  "data_files_size_in_mb": 100,
  "video_files_size_in_mb": 200,
  "fps": 30,
  "splits": {"train": "0:2"},
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": null,
  "features": {
    "observation.state": {"dtype": "float32", "shape": [1], "names": ["joint_position"]},
    "action": {"dtype": "float32", "shape": [1], "names": ["joint_target"]},
    "timestamp": {"dtype": "float32", "shape": [1], "names": null},
    "frame_index": {"dtype": "int64", "shape": [1], "names": null},
    "episode_index": {"dtype": "int64", "shape": [1], "names": null},
    "index": {"dtype": "int64", "shape": [1], "names": null},
    "task_index": {"dtype": "int64", "shape": [1], "names": null}
  }
}
```

## 4. Frame Parquet schema

Every frame has the following standard scalar columns, plus the declared nonvideo features:

| Column | Type | Meaning |
|---|---|---|
| `timestamp` | float32 | Time in seconds relative to the episode's start |
| `frame_index` | int64 | Zero-based frame number within this episode |
| `episode_index` | int64 | Zero-based episode ID |
| `index` | int64 | Zero-based global frame ID across the dataset |
| `task_index` | int64 | Task ID resolved through `meta/tasks.parquet` |

For episode `e`, with length `L` and global start `S`, the canonical rows satisfy:

```text
frame_index = 0, 1, ..., L-1
episode_index = e
index = S + frame_index
```

Writer behavior: if the caller supplies no timestamp, `timestamp = frame_index / fps`. The caller can supply timestamps explicitly. For a regularly sampled interoperable export, timestamps SHOULD begin at zero and follow the advertised FPS within the decoder/sampling tolerance. The implementation's tolerance is a runtime option, not a field in this file format.

`task_index` is per frame; an episode may contain more than one task. `action` and `observation.state` are common feature names, not a complete universal schema for all datasets.

Reference writer data compression is Snappy with dictionary encoding. This is an implementation choice; compatible Parquet readers need not require that compression codec.

## 5. Task table: `meta/tasks.parquet`

The logical table associates each task description string with one integer `task_index`. IDs assigned by the writer start at zero and are contiguous. Every frame's task ID MUST resolve to an entry in this table.

**Physical representation in v0.4.4:** the writer creates a pandas DataFrame with a `task_index` column and task descriptions as the DataFrame index, then calls `to_parquet()`. It does not name that index. With the standard pandas/Arrow path, its physical field is commonly `__index_level_0__`, and pandas schema metadata identifies it as the index. A raw Arrow exporter must reproduce the index metadata or verify compatibility with the pandas reader; a plain `task` column is not automatically equivalent for this release.

Logical example:

| Task text (DataFrame index) | `task_index` |
|---|---:|
| Move the joint to its target | 0 |

The reader's task lookup uses the DataFrame index. Do not treat the physical index field's incidental spelling as the semantic task name. Later code may name this index explicitly.

## 6. Episode metadata Parquet schema

Each episode has one row in the metadata tables. The reference writer emits these fields. Ordinary integer values infer int64 through Arrow; task lists are lists of strings and video offsets infer float64. Statistics columns follow their serialized array values.

| Column | Logical type | Meaning |
|---|---|---|
| `episode_index` | integer | Episode ID |
| `length` | integer | Number of frame rows |
| `tasks` | list of strings | Unique task descriptions used in the episode; order is not significant |
| `data/chunk_index` | integer | Chunk containing this episode's data shard |
| `data/file_index` | integer | Data shard file index |
| `dataset_from_index` | integer | Inclusive global frame start |
| `dataset_to_index` | integer | Exclusive global frame end |
| `meta/episodes/chunk_index` | integer | Metadata shard chunk containing this row |
| `meta/episodes/file_index` | integer | Metadata shard file containing this row |
| `stats/{feature}/{statistic}` | numeric scalar/list structure | Flattened per-episode feature statistics |

For each declared video feature `K`, the writer additionally emits:

| Column | Logical type | Meaning |
|---|---|---|
| `videos/K/chunk_index` | integer | Video shard chunk for camera K |
| `videos/K/file_index` | integer | Video shard file for camera K |
| `videos/K/from_timestamp` | number | Episode start in that MP4, in seconds |
| `videos/K/to_timestamp` | number | Episode end boundary in that MP4, in seconds |

Here `K` is substituted literally, such as `observation.images.front`.

Required consistency relationships for a canonical complete dataset:

```text
length = dataset_to_index - dataset_from_index
first episode's dataset_from_index = 0
next episode's dataset_from_index = preceding episode's dataset_to_index
last episode's dataset_to_index = info.total_frames
sum(length) = info.total_frames
number of episode rows = info.total_episodes
```

The reference reader relies on ordered episode metadata for positional lookup; keep episode rows in increasing `episode_index` order across ordered shards. Do not confuse metadata shard indices with data shard indices.

Readers can locate an episode's data shard using its data locator, then select rows whose global `index` falls in `[dataset_from_index, dataset_to_index)`. These endpoints are global dataset coordinates, not row offsets local to the shard.

## 7. Encoded video representation

A video feature is stored as MP4, not as a per-frame path column or a `VideoFrame` struct in the data Parquet. The reference writer concatenates complete episode clips for a camera into shared MP4 files, rolling over between episodes.

For a frame with episode-relative timestamp `t`, the reader queries:

```text
video_path = info.video_path.format(
    video_key=K,
    chunk_index=episode["videos/K/chunk_index"],
    file_index=episode["videos/K/file_index"]
)
mp4_query_time = episode["videos/K/from_timestamp"] + t
```

`to_timestamp` describes an end boundary; it is not the timestamp of the last frame. In a conventional constant-FPS clip, the duration is `length / fps`, while the last frame is at `(length - 1) / fps` relative to the episode's start. The writer obtains clip duration from the encoded stream, so small time-base rounding differences can occur.

The v0.4.4 implementation populates `features[K].info` using `get_video_info()` from the encoded stream. There is no separately published schema for `info`; the following is the reference implementation's video metadata convention, not a universal `info` schema for every feature type.

| Keys | Value type |
|---|---|
| `video.height`, `video.width`, `video.channels`, `video.fps` | Integer |
| `video.codec`, `video.pix_fmt` | String |
| `video.is_depth_map` | Boolean |

The object is flat: `"video.width": 640` is a single literal dotted key, not a nested `video` object. It may also include audio information. Use actual encoded-stream properties. The format does not require one universal codec such as H.264; decoder support is an interoperability constraint.

Camera files can share neither file numbers nor rollover points with other cameras or with the data Parquet. Resolve every camera independently.

## 8. Embedded image representation

For `dtype: "image"`, the frame table contains a Hugging Face `datasets.Image()` field. Its Arrow representation is a struct with `bytes` (binary) and `path` (string) fields. The reference writer embeds image bytes before writing Parquet, so final frame tables can be self-contained.

The `images/.../episode-.../frame-....png` path pattern is used by the recording workflow for staging images. A directory of PNG files alone does not substitute for the declared frame image fields in the canonical output. Exact byte/path population follows the datasets library's image embedding behavior.

## 9. Statistics

`meta/stats.json` maps feature names to statistics objects. Values are serialized numeric arrays, including one-element arrays for scalar statistics/counts. String features are skipped.

The reference statistics utilities produce `min`, `max`, `mean`, `std`, `count`, and quantiles `q01`, `q10`, `q50`, `q90`, `q99`.

| Feature kind | Statistic shapes/interpretation |
|---|---|
| Scalar numeric | One-element arrays for summary values |
| Numeric vector `[D]` | Per-component arrays of length D |
| RGB image/video | Per-channel values of shape `[3,1,1]`, normalized to the pixel range `[0,1]` |
| Count | Shape `[1]`; count of samples represented by the estimator |

Numeric standard deviation is population standard deviation. Visual statistics may use sampled/downsampled images; their counts need not equal the episode's frame count. Do not interpret image statistics as an exact scan of every pixel in every frame.

Example for a scalar state feature, with values `0,1,2,0,1,2`:

```json
{
  "observation.state": {
    "min": [0.0],
    "max": [2.0],
    "mean": [1.0],
    "std": [0.8164965809],
    "count": [6]
  }
}
```

This is an illustrative entry, not the complete statistics output; the reference utilities also emit quantiles. Per-episode values appear in metadata columns such as `stats/observation.state/mean` and `stats/observation.state/count`.

Global means and variances are aggregated with sample counts. Reference quantiles are histogram approximations, and global quantiles are formed by weighted aggregation of episode quantiles. They are not guaranteed to be exact global empirical percentiles.

The reference `load_stats` returns null if the statistics file is absent. Nevertheless, exporters targeting training should provide usable statistics for features that policies normalize. Absence may be tolerated by a reader but does not guarantee a usable training dataset.

## 10. Worked indexing example

Two three-frame episodes share `data/chunk-000/file-000.parquet`:

| `index` | `episode_index` | `frame_index` | `timestamp` (approximately) | `task_index` |
|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0.000000 | 0 |
| 1 | 0 | 1 | 0.033333 | 0 |
| 2 | 0 | 2 | 0.066667 | 0 |
| 3 | 1 | 0 | 0.000000 | 0 |
| 4 | 1 | 1 | 0.033333 | 0 |
| 5 | 1 | 2 | 0.066667 | 0 |

The episode metadata's global intervals are `[0,3)` and `[3,6)`. Both data locators are `(chunk_index=0, file_index=0)`. State/action columns are omitted from this table only for readability.

If a 30-FPS camera were added and both clips shared one MP4, their conventional video intervals would be `[0.0,0.1)` and `[0.1,0.2)`. Episode 1, frame 2 would be queried at approximately `0.1 + 2/30 = 0.166667` seconds in that MP4. Adding this camera also requires its feature descriptor, video path template, and per-episode video locator/offset fields.

## 11. Export validation checklist

1. Check that the descriptor's counts match the data and episode tables and that its compatibility marker is `v3.0`.
2. Check the standard frame column types and the scalar `[1]` mapping; verify declared nonvideo features against the actual Parquet schema.
3. Check ordered, contiguous episode IDs/global frame IDs, episode lengths, and zero-based frame IDs within each episode.
4. Resolve every episode's data locator; verify that its global interval contains exactly the declared episode's rows.
5. Resolve every frame task ID through the pandas-compatible task table. Check that episode `tasks` matches the set of task descriptions used by its frames.
6. For each video feature, resolve its independent locator and offsets, and verify that requested frame times decode inside the corresponding clip interval.
7. Check that advertised FPS, visual dimensions, and codec metadata match the encoded streams.
8. Check statistics shapes and finite values for the features used by the intended policy; do not require statistics for strings.
9. Close all Parquet writers before distribution. The reference `finalize()` flushes buffered metadata and writes Parquet footers; an unfinished recording is not a completed dataset.

These checks define a useful canonical export profile, not a guarantee that the reference reader rejects every malformed input or that every existing Hub dataset passes them. Extensions such as optional subtask tables are outside this core profile. Unknown application fields should be documented separately.

## 12. Source map

All implementation links below are pinned to package release **v0.4.4**; that package implements dataset marker **v3.0**.

| Source | Relevant definitions |
|---|---|
| [datasets/utils.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/utils.py) | Paths, default features, `create_empty_dataset_info`, `get_hf_features_from_features`, `write_tasks`, `load_tasks`, feature validation |
| [datasets/lerobot_dataset.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/lerobot_dataset.py) | `CODEBASE_VERSION`, metadata creation/loading, `save_episode_tasks`, `save_episode`, `_save_episode_data`, `_save_episode_video`, `_query_videos` |
| [datasets/compute_stats.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/compute_stats.py) | Statistics shapes, normalization, sampling and aggregation |
| [datasets/video_utils.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/video_utils.py) | Stream properties and video duration/decoding utilities |
| [datasets/v30/convert_dataset_v21_to_v30.py](https://github.com/huggingface/lerobot/blob/v0.4.4/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py) | v2.1-to-v3.0 migration |

When targeting a different LeRobot package release, compare its reader and writer to these sources. In particular, verify the task-table index representation, feature serialization, video properties, and accepted optional metadata rather than assuming the shared `v3.0` marker specifies all release-dependent behavior.
