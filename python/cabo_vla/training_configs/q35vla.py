import argparse
import json
from dataclasses import asdict, fields
from math import isfinite
from pathlib import Path
from time import perf_counter, time

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from huggingface_hub import snapshot_download
from safetensors.flax import save_file
from tokenizers import Tokenizer

from cabo_vla.data.dataloader import Batch, DataloaderConfig, NormalizationType
from cabo_vla.load_model import load_hf_qwen35
from cabo_vla.models.qwen35vla.qwen35 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from cabo_vla.models.qwen35vla.qwen35vla import Qwen3_5VLA, Qwen3_5VLAConfig
from cabo_vla.train import TrainConfig, train

DATASET_REPO = "lerobot/libero"
BACKBONE_REPO = "Qwen/Qwen3.5-0.8B"
BACKBONE_FILES = ["config.json", "tokenizer*.json", "*.safetensors", "*.safetensors.index.json"]


def load_config(checkpoint: Path, pad_token_id: int) -> Qwen3_5VLAConfig:
    config = json.loads((checkpoint / "config.json").read_text())
    text = config["text_config"]
    text_kwargs = {field.name: text[field.name] for field in fields(Qwen3_5TextConfig) if field.name in text}
    rope = text["rope_parameters"]
    text_kwargs.update(
        rope_theta=rope["rope_theta"],
        partial_rotary_factor=rope["partial_rotary_factor"],
        mrope_section=tuple(rope["mrope_section"]),
    )
    vision = config["vision_config"]
    vision_kwargs = {field.name: vision[field.name] for field in fields(Qwen3_5VisionConfig) if field.name in vision}
    return Qwen3_5VLAConfig(
        qwen_config=Qwen3_5Config(
            text_config=Qwen3_5TextConfig(**text_kwargs),
            vision_config=Qwen3_5VisionConfig(**vision_kwargs),
            image_token_id=config["image_token_id"],
            video_token_id=config["video_token_id"],
        ),
        state_size=8,
        action_size=7,
        pad_token_id=pad_token_id,
    )


def create_model(cfg: Qwen3_5VLAConfig, checkpoint: Path, *, rngs: nnx.Rngs) -> Qwen3_5VLA:
    backbone = load_hf_qwen35(cfg.qwen_config, checkpoint)
    model = Qwen3_5VLA(cfg, rngs, backbone=backbone)
    for _, module in nnx.iter_modules(model):
        if isinstance(module, (nnx.Linear, nnx.Conv, nnx.Embed, nnx.LayerNorm, nnx.RMSNorm)):
            module.dtype = jnp.bfloat16
    return model


def loss_fn(
    params: nnx.State,
    frozen_state: nnx.State,
    graphdef: nnx.GraphDef,
    batch: Batch[jax.Array],
    *,
    key: jax.Array,
) -> jax.Array:
    model = nnx.merge(graphdef, params, frozen_state)
    noise_key, time_key = jax.random.split(key)
    noise = jax.random.normal(noise_key, batch.action.shape, dtype=jnp.float32)
    time = jax.random.uniform(time_key, (batch.action.shape[0],))
    amount = time[:, None, None]
    noisy_actions = (1 - amount) * batch.action + amount * noise
    velocity = model(batch, action_t=noisy_actions, traj_time=time)
    return jnp.mean(jnp.square(velocity.astype(jnp.float32) - (noise - batch.action)))


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3.5 VLA on LeRobot LIBERO.")
    parser.add_argument("--dataset", type=Path, help="Local LeRobot LIBERO dataset directory")
    parser.add_argument("--checkpoint", type=Path, help="Local Qwen3.5-0.8B Hugging Face snapshot")
    parser.add_argument("--output", type=Path, default=Path("runs/q35vla"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--benchmark-steps", type=int, help="Measure training throughput for this many optimizer steps")
    parser.add_argument("--warmup-steps", type=int, default=5, help="Exclude this many initial steps from benchmark timing")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.num_workers, args.warmup_steps) < 1:
        parser.error("Epochs, batch size, worker count, and warmup steps must be positive")
    if args.benchmark_steps is not None and args.benchmark_steps < 1:
        parser.error("--benchmark-steps must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    log_file = args.output / "train.log"
    model_file = args.output / "model.safetensors"
    if log_file.exists() or model_file.exists():
        raise FileExistsError("Use a fresh output directory to preserve previous runs.")
    if jax.config.jax_compilation_cache_dir is None:
        jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    checkpoint = args.checkpoint or Path(snapshot_download(BACKBONE_REPO, allow_patterns=BACKBONE_FILES))
    tokenizer = Tokenizer.from_file(str(checkpoint / "tokenizer.json"))
    pad_token = json.loads((checkpoint / "tokenizer_config.json").read_text())["pad_token"]
    pad_token_id = tokenizer.token_to_id(pad_token)
    if pad_token_id is None:
        raise ValueError(f"Checkpoint tokenizer has no ID for pad token {pad_token!r}")
    vla_config = load_config(checkpoint, pad_token_id)

    def tokenize_instruction(instruction: str) -> jax.Array:
        ids = tokenizer.encode(instruction, add_special_tokens=False).ids
        return jnp.asarray(ids, dtype=jnp.int32)

    dataset = args.dataset or Path(snapshot_download(DATASET_REPO, repo_type="dataset"))
    config = TrainConfig(
        data_config=DataloaderConfig(
            path=dataset,
            resolution=(256, 256),
            batch_size=args.batch_size,
            action_horizon=vla_config.action_chunk_size,
            observation_features=("observation.images.image", "observation.images.image2"),
            action_features=(("action", NormalizationType.QUANTILE),),
            prio_features=(("observation.state", NormalizationType.QUANTILE),),
            tokenizer=tokenize_instruction,
            seq_len=64,
            pad_token_id=pad_token_id,
            observation_range=vla_config.observation_range,
            seed=42,
            num_workers=args.num_workers,
        ),
        log_file=log_file,
        model_type=create_model,
        model_kwargs={"cfg": vla_config, "checkpoint": checkpoint},
        optimizer=optax.adamw(learning_rate=1e-5),
        loss_fn=loss_fn,
        seed=42,
        n_epochs=args.epochs,
    )
    if args.benchmark_steps is not None:
        target_steps = args.warmup_steps + args.benchmark_steps
        started = elapsed = 0.0
        final_loss = 0.0
        window_start = window_end = 0.0
        batch_wait = optimizer_time = 0.0

        def record_timing(step: int, wait: float, compute: float):
            nonlocal batch_wait, optimizer_time
            if step > args.warmup_steps:
                batch_wait += wait
                optimizer_time += compute

        def record_step(step: int, loss: float):
            nonlocal started, elapsed, final_loss, window_start, window_end
            if not isfinite(loss):
                raise RuntimeError(f"Non-finite training loss at step {step}: {loss}")
            if step == args.warmup_steps:
                started = perf_counter()
                window_start = time()
            if step == target_steps:
                window_end = time()
                elapsed = perf_counter() - started
                final_loss = loss

        train(config, max_steps=target_steps, on_step=record_step, on_step_timing=record_timing)
        if elapsed <= 0:
            raise RuntimeError(f"Training ended before {target_steps} benchmark steps completed")
        result = {
            "peak_bytes_in_use": (jax.devices()[0].memory_stats() or {}).get("peak_bytes_in_use"),
            "device": jax.devices()[0].device_kind,
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.benchmark_steps,
            "elapsed_seconds": elapsed,
            "window_start_unix": window_start,
            "window_end_unix": window_end,
            "dataloader_workers": min(args.num_workers, args.batch_size),
            "batch_wait_seconds": batch_wait,
            "optimizer_seconds": optimizer_time,
            "steps_per_second": args.benchmark_steps / elapsed,
            "examples_per_second": args.batch_size * args.benchmark_steps / elapsed,
            "final_loss": final_loss,
        }
        (args.output / "throughput.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return None

    state = train(config)
    weights = {".".join(map(str, path)): param[...] for path, param in state.params.flat_state()}
    temporary_file = model_file.with_suffix(".tmp")
    save_file(weights, temporary_file, metadata={"vla_config": json.dumps(asdict(vla_config))})
    temporary_file.replace(model_file)
    return state


if __name__ == "__main__":
    main()
