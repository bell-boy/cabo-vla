import json
import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import modal

ROOT = Path(__file__).resolve().parent
DATA = Path("/data")
VOLUME_NAME = "cabo-vla-q35vla"
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image_env = {"PYTHONPATH": "/workspace/python", "HF_HOME": str(DATA / "hf"), "JAX_COMPILATION_CACHE_DIR": str(DATA / "jax")}
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(ROOT), groups=["dev"])
    .env(image_env)
    .add_local_dir(ROOT / "python", "/workspace/python", ignore=["**/__pycache__/**"])
)
eval_image = (
    modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("libosmesa6", "libgl1", "libglib2.0-0")
    .uv_sync(str(ROOT), groups=["dev"])
    .uv_pip_install("torch==2.5.1+cpu", index_url="https://download.pytorch.org/whl/cpu")
    .uv_pip_install("hf-libero==0.1.5", "torch==2.5.1+cpu", extra_index_url="https://download.pytorch.org/whl/cpu")
    .env(image_env)
    .add_local_dir(ROOT / "python", "/workspace/python", ignore=["**/__pycache__/**"])
)
app = modal.App("cabo-vla-q35vla", image=image)
wandb_secrets = [modal.Secret.from_local_environ(["WANDB_API_KEY"])] if os.environ.get("WANDB_API_KEY") else []


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


@app.function(image=eval_image, gpu=["L4", "A10"], volumes={str(DATA): volume}, timeout=1800, cpu=4, memory=16384, secrets=wandb_secrets)
def evaluate_checkpoint(
    weights: str,
    dataset: str,
    checkpoint: str,
    tasks: int = 5,
    trials: int = 2,
    max_steps: int = 220,
    tracking: dict | None = None,
    video_dir: str | None = None,
    temporal_decay: float | None = None,
) -> dict:
    volume.reload()
    with TemporaryDirectory() as scratch:
        output = Path(scratch) / "results.json"
        command = [
            "python",
            "-u",
            "-m",
            "cabo_vla.eval_libero_spatial",
            weights,
            "--stats",
            str(Path(dataset) / "meta" / "stats.json"),
            "--tokenizer",
            str(Path(checkpoint) / "tokenizer.json"),
            "--tasks",
            str(tasks),
            "--trials",
            str(trials),
            "--max-steps",
            str(max_steps),
            "--output",
            str(output),
        ]
        if video_dir:
            command += ["--video-dir", video_dir]
        if temporal_decay is not None:
            command += ["--temporal-decay", str(temporal_decay)]
        subprocess.run(
            command,
            input="n\n",
            text=True,
            check=True,
            env={
                **os.environ,
                "JAX_PLATFORMS": "cuda",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                "MUJOCO_GL": "osmesa",
                "PYOPENGL_PLATFORM": "osmesa",
                "LIBERO_CONFIG_PATH": scratch,
                "OPENBLAS_NUM_THREADS": "1",
            },
        )
        result = json.loads(output.read_text())
    if video_dir:
        (Path(video_dir) / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
    if tracking is not None:
        import wandb

        Path(weights).with_suffix(".eval.json").write_text(json.dumps(result, indent=2) + "\n")
        volume.commit()
        step = tracking["step"]
        with wandb.init(
            id=tracking["id"],
            project=tracking["project"],
            entity=tracking["entity"],
            resume="must",
            settings=wandb.Settings(mode="shared", x_primary=False, x_update_finish_state=False, x_label=f"libero-{step}"),
        ) as run:
            run.define_metric("libero_spatial/*", step_metric="checkpoint_step")
            metrics = {"checkpoint_step": step, "libero_spatial/success_rate": result["success_rate"]}
            for task in result["tasks"]:
                metrics[f"libero_spatial/task_{task['task_id']}_success_rate"] = task["successes"] / task["trials"]
            run.log(metrics)
    return result


@app.function(
    gpu="H100!",
    volumes={str(DATA): volume},
    timeout=24 * 3600,
    cpu=8,
    memory=24576,
    secrets=wandb_secrets,
)
def train_on_h100(
    dataset: str,
    checkpoint: str,
    batch_size: int,
    epochs: int,
    num_workers: int,
    wandb_project: str = "",
    memory_fraction: float = 0.75,
    run_id: str = "",
    eval_checkpoints: bool = False,
    initial_weights: str = "",
    checkpoint_every_quarter: bool = False,
    seed: int = 42,
) -> str:
    output = DATA / "runs" / (run_id or uuid4().hex)
    command = training_command(dataset, checkpoint, str(output), batch_size, epochs, num_workers)
    command += ["--seed", str(seed)]
    if initial_weights:
        command += ["--initial-weights", initial_weights]
    if checkpoint_every_quarter:
        command += ["--checkpoint-every-quarter"]
    if wandb_project:
        command += ["--wandb-project", wandb_project]
    if eval_checkpoints:
        command += ["--eval-checkpoints"]
    print(f"Training output: {output.relative_to(DATA)}", flush=True)
    env = {**os.environ, "JAX_PLATFORMS": "cuda", "XLA_CLIENT_MEM_FRACTION": str(memory_fraction)}
    with subprocess.Popen(command, stdout=subprocess.PIPE, text=True, env=env) as process:
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                if line.startswith("Checkpoint saved: "):
                    volume.commit()
                    print("Checkpoint committed to Modal Volume", flush=True)
                if line.startswith("Checkpoint evaluation: "):
                    request = json.loads(line.removeprefix("Checkpoint evaluation: "))
                    try:
                        call = evaluate_checkpoint.spawn(request["weights"], dataset, checkpoint, tracking=request)
                        print(f"LIBERO evaluation queued at step {request['step']}: {call.object_id}", flush=True)
                    except Exception as error:
                        print(f"Could not queue LIBERO evaluation at step {request['step']}: {error}", flush=True)
            if return_code := process.wait():
                raise subprocess.CalledProcessError(return_code, command)
        finally:
            if process.poll() is None:
                process.terminate()
    volume.commit()
    return str(output.relative_to(DATA))


@app.local_entrypoint()
def main(
    num_workers: int = 4,
    epochs: int = 1,
    batch_size: int = 1,
    wandb_project: str = "",
    memory_fraction: float = 0.75,
    eval_checkpoints: bool = False,
    initial_weights: str = "",
    checkpoint_every_quarter: bool = False,
    seed: int = 42,
):
    if min(epochs, num_workers, batch_size) < 1:
        raise ValueError("Epochs, batch size, and worker count must be positive.")
    if not 0 < memory_fraction <= 1:
        raise ValueError("Memory fraction must be in (0, 1].")
    if wandb_project and not os.environ.get("WANDB_API_KEY"):
        raise ValueError("W&B logging requires WANDB_API_KEY in the local environment.")
    if eval_checkpoints and not wandb_project:
        raise ValueError("Checkpoint evaluation requires --wandb-project.")
    dataset, checkpoint = prepare_inputs.remote()
    output = train_on_h100.remote(
        dataset,
        checkpoint,
        batch_size,
        epochs,
        num_workers,
        wandb_project,
        memory_fraction,
        eval_checkpoints=eval_checkpoints,
        initial_weights=initial_weights,
        checkpoint_every_quarter=checkpoint_every_quarter,
        seed=seed,
    )
    print(f"Weights and log saved to Modal Volume {VOLUME_NAME} at {output}")
