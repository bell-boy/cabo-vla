import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from tempfile import TemporaryDirectory
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parent
DATA = Path("/data")
VOLUME_NAME = "cabo-vla-q35vla"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), groups=["dev"])
    .env({"PYTHONPATH": "/workspace/python", "HF_HOME": str(DATA / "hf"), "JAX_COMPILATION_CACHE_DIR": str(DATA / "jax")})
    .add_local_dir(ROOT / "python", "/workspace/python", ignore=["**/__pycache__/**"])
)
app = modal.App("cabo-vla-q35vla", image=image)


@app.function(volumes={str(DATA): volume}, timeout=1800, cpu=4, memory=8192)
def prepare_inputs() -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    from cabo_vla.training_configs.q35vla import BACKBONE_FILES, BACKBONE_REPO, DATASET_REPO

    dataset = snapshot_download(DATASET_REPO, repo_type="dataset", max_workers=12)
    checkpoint = snapshot_download(BACKBONE_REPO, allow_patterns=BACKBONE_FILES, max_workers=12)
    volume.commit()
    return dataset, checkpoint


def training_command(dataset: str, checkpoint: str, output: str, batch_size: int, epochs: int, num_workers: int) -> list[str]:
    return [
        "python",
        "-u",
        "-m",
        "cabo_vla.training_configs.q35vla",
        "--dataset",
        dataset,
        "--checkpoint",
        checkpoint,
        "--output",
        output,
        "--batch-size",
        str(batch_size),
        "--epochs",
        str(epochs),
        "--num-workers",
        str(num_workers),
    ]


@app.function(gpu="H100!", volumes={str(DATA): volume}, timeout=24 * 3600, cpu=8, memory=24576)
def train_on_h100(dataset: str, checkpoint: str, batch_size: int, epochs: int, num_workers: int) -> str:
    output = DATA / "runs" / uuid4().hex
    subprocess.run(training_command(dataset, checkpoint, str(output), batch_size, epochs, num_workers), check=True, env={**os.environ, "JAX_PLATFORMS": "cuda"})
    volume.commit()
    return str(output.relative_to(DATA))


def gpu_activity(samples: Path, start: float, end: float) -> dict:
    readings = []
    for line in samples.read_text().splitlines():
        stamp, _, utilization = line.partition(", ")
        if utilization.isdigit():
            timestamp = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=UTC).timestamp()
            if start <= timestamp <= end:
                readings.append(int(utilization))
    return {"gpu_utilization_percent": mean(readings) if readings else None, "gpu_samples": len(readings)}


@app.function(gpu="H100!", volumes={str(DATA): volume}, timeout=7200, cpu=8, memory=24576)
def sweep_on_h100(dataset: str, checkpoint: str, batch_sizes: list[int], steps: int, warmup_steps: int, num_workers: int) -> dict:
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True).strip()
    print(f"Benchmark GPU: {gpu}", flush=True)
    results = []
    with TemporaryDirectory() as scratch:
        for batch_size in batch_sizes:
            print(f"Sweeping batch size {batch_size}", flush=True)
            output = Path(scratch) / str(batch_size)
            command = training_command(dataset, checkpoint, str(output), batch_size, 1, num_workers)
            command += ["--benchmark-steps", str(steps), "--warmup-steps", str(warmup_steps)]
            samples = output.with_suffix(".csv")
            query = ["nvidia-smi", "--query-gpu=timestamp,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "200"]
            with samples.open("w") as stream, subprocess.Popen(query, stdout=stream, env={**os.environ, "TZ": "UTC"}) as monitor:
                try:
                    result = subprocess.run(command, stderr=subprocess.PIPE, text=True, env={**os.environ, "JAX_PLATFORMS": "cuda"})
                finally:
                    monitor.terminate()
            if result.returncode == 0:
                metrics = json.loads((output / "throughput.json").read_text())
                metrics.update(gpu_activity(samples, metrics["window_start_unix"], metrics["window_end_unix"]))
                results.append(metrics)
            elif "RESOURCE_EXHAUSTED" in result.stderr or "CUDA_ERROR_OUT_OF_MEMORY" in result.stderr:
                print(f"Batch size {batch_size} ran out of GPU memory", flush=True)
                results.append({"batch_size": batch_size, "status": "out_of_memory"})
            else:
                raise RuntimeError(f"Batch size {batch_size} failed with exit code {result.returncode}:\n{result.stderr[-6000:]}")
    return {"gpu": gpu, "num_workers": num_workers, "results": results}


@app.local_entrypoint()
def main(
    batch_sizes: str = "1,2,4,8,16",
    steps: int = 20,
    warmup_steps: int = 5,
    num_workers: int = 4,
    epochs: int = 0,
    batch_size: int = 1,
):
    sizes = sorted({int(value) for value in batch_sizes.split(",")})
    if min(*sizes, steps, warmup_steps, num_workers, batch_size) < 1 or epochs < 0:
        raise ValueError("Batch sizes, steps, and workers must be positive. Epochs must be nonnegative.")
    dataset, checkpoint = prepare_inputs.remote()
    if epochs:
        output = train_on_h100.remote(dataset, checkpoint, batch_size, epochs, num_workers)
        print(f"Weights and log saved to Modal Volume {VOLUME_NAME} at {output}")
    else:
        print(json.dumps(sweep_on_h100.remote(dataset, checkpoint, sizes, steps, warmup_steps, num_workers), indent=2))
