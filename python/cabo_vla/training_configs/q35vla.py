import argparse
import json
from dataclasses import asdict, fields
from math import ceil
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.flax import save_file
from tokenizers import Tokenizer

from cabo_vla.data.dataloader import Batch, DataloaderConfig, NormalizationType
from cabo_vla.load_model import load_hf_qwen35
from cabo_vla.models.qwen35vla.qwen35 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
from cabo_vla.models.qwen35vla.qwen35vla import Qwen3_5VLA, Qwen3_5VLAConfig
from cabo_vla.train import TrainConfig, TrainState, train

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


def create_model(cfg: Qwen3_5VLAConfig, checkpoint: Path | None, *, rngs: nnx.Rngs, initial_weights: Path | None = None) -> Qwen3_5VLA:
    if initial_weights is not None:
        loaded_config, model = load_vla_checkpoint(initial_weights)
        if loaded_config != cfg:
            raise ValueError("The VLA checkpoint configuration does not match the training configuration")
        return model

    backbone = load_hf_qwen35(cfg.qwen_config, checkpoint) if checkpoint is not None else None
    model = Qwen3_5VLA(cfg, rngs, backbone=backbone)
    for _, module in nnx.iter_modules(model):
        if isinstance(module, (nnx.Linear, nnx.Conv, nnx.Embed, nnx.LayerNorm, nnx.RMSNorm)):
            module.dtype = jnp.bfloat16
    return model


def load_vla_checkpoint(checkpoint: Path) -> tuple[Qwen3_5VLAConfig, Qwen3_5VLA]:
    with safe_open(checkpoint, framework="flax") as weights:
        config = json.loads((weights.metadata() or {})["vla_config"])
        backbone = config.pop("qwen_config")
        text = backbone.pop("text_config")
        text["mrope_section"] = tuple(text["mrope_section"])
        vision = backbone.pop("vision_config")
        config["observation_range"] = tuple(config["observation_range"])
        cfg = Qwen3_5VLAConfig(
            qwen_config=Qwen3_5Config(Qwen3_5TextConfig(**text), Qwen3_5VisionConfig(**vision), **backbone),
            **config,
        )
        abstract = nnx.eval_shape(lambda: create_model(cfg, None, rngs=nnx.Rngs(0)))
        graph, params = nnx.split(abstract)
        loaded = []
        for path, param in params.flat_state():
            name = ".".join(map(str, path))
            value = weights.get_tensor(name)
            if value.shape != param.shape:
                raise ValueError(f"{name}: checkpoint shape {value.shape} differs from expected {param.shape}")
            loaded.append((path, param.replace(value=value)))
    return cfg, nnx.merge(graph, nnx.State.from_flat_path(loaded))


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
    parser.add_argument("--initial-weights", type=Path, help="Warm-start from a Cabo VLA safetensors checkpoint")
    parser.add_argument("--checkpoint-every-quarter", action="store_true", help="Save weights at each quarter of every epoch")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("runs/q35vla"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eval-checkpoints", action="store_true", help="Queue Spatial rollouts through the Modal launcher")
    parser.add_argument("--wandb-project", help="Log training loss to this Weights & Biases project")
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.num_workers) < 1:
        parser.error("Epochs, batch size, and worker count must be positive")

    if args.eval_checkpoints and not args.wandb_project:
        parser.error("--eval-checkpoints requires --wandb-project")

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
            seed=args.seed,
            num_workers=args.num_workers,
        ),
        log_file=log_file,
        model_type=create_model,
        model_kwargs={"cfg": vla_config, "checkpoint": checkpoint, "initial_weights": args.initial_weights},
        optimizer=optax.adamw(learning_rate=1e-5),
        loss_fn=loss_fn,
        seed=args.seed,
        n_epochs=args.epochs,
    )
    checkpoint_names: dict[int, str] = {}

    def save_checkpoint(step: int, total_steps: int, state: TrainState):
        if step == 0:
            if args.checkpoint_every_quarter:
                per_epoch = total_steps // args.epochs
                for epoch in range(args.epochs):
                    for quarter in range(1, 5):
                        milestone = epoch * per_epoch + (per_epoch * quarter + 3) // 4
                        checkpoint_names[milestone] = f"model-epoch-{epoch + 1:02d}-{quarter * 25:03d}pct.safetensors"
                checkpoint_names[total_steps] = model_file.name
            else:
                checkpoint_names.update(
                    {
                        0: "model-start.safetensors",
                        ceil(total_steps / 2): "model-half.safetensors",
                        ceil(total_steps * 3 / 4): "model-75pct.safetensors",
                        total_steps: model_file.name,
                    }
                )
        filename = checkpoint_names.get(step)
        if filename is None:
            return
        destination = args.output / filename
        weights = {".".join(map(str, path)): param[...] for path, param in state.params.flat_state()}
        temporary_file = destination.with_suffix(".tmp")
        save_file(weights, temporary_file, metadata={"vla_config": json.dumps(asdict(vla_config)), "step": str(step), "total_steps": str(total_steps)})
        temporary_file.replace(destination)
        print(f"Checkpoint saved: {destination}", flush=True)
        return destination

    if args.wandb_project:
        import wandb

        with wandb.init(
            project=args.wandb_project,
            name=args.output.name,
            dir=str(args.output),
            settings=wandb.Settings(mode="shared", x_label="trainer") if args.eval_checkpoints else None,
            config={
                "backbone": BACKBONE_REPO,
                "checkpoint_revision": checkpoint.name,
                "initial_weights": str(args.initial_weights) if args.initial_weights is not None else None,
                "checkpoint_every_quarter": args.checkpoint_every_quarter,
                "dataset": DATASET_REPO,
                "dataset_revision": dataset.name,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "dataloader_workers": min(args.num_workers, args.batch_size),
                "learning_rate": 1e-5,
                "seed": config.seed,
            },
        ) as run:
            print(f"W&B run: {run.url}", flush=True)

            def checkpoint_and_queue_eval(step: int, total_steps: int, state: TrainState):
                weights = save_checkpoint(step, total_steps, state)
                if weights is not None and args.eval_checkpoints:
                    request = {"weights": str(weights), "step": step, "id": run.id, "project": run.project, "entity": run.entity}
                    print(f"Checkpoint evaluation: {json.dumps(request)}", flush=True)

            def log_loss(step: int, loss: float):
                if args.eval_checkpoints:
                    run.log({"train/step": step, "train/loss": loss})
                else:
                    run.log({"train/loss": loss}, step=step)

            if args.eval_checkpoints:
                run.define_metric("train/loss", step_metric="train/step")
            state = train(config, on_step=log_loss, on_state=checkpoint_and_queue_eval)
    else:
        state = train(config, on_state=save_checkpoint)
    return state


if __name__ == "__main__":
    main()
