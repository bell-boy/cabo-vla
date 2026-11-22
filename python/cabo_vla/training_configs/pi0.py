import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from huggingface_hub import snapshot_download

from cabo_vla.config import VLAConfig
from cabo_vla.data.dataloader import Batch, DataloaderConfig, NormalizationType
from cabo_vla.load_model import load_hf_vit
from cabo_vla.models.pi0 import VLA
from cabo_vla.train import TrainConfig, train


def loss_fn(
    params: nnx.State,
    frozen_state: nnx.State,
    graphdef: nnx.GraphDef,
    batch: Batch,
    *,
    key: jax.Array,
) -> jax.Array:
    noise_key, time_key = jax.random.split(key)
    model = nnx.merge(graphdef, params, frozen_state)
    _, cache = model.encode_state_imgs(imgs=batch.observations, state=batch.prio, time=batch.timestamp)
    noise = jax.random.normal(noise_key, shape=batch.action.shape, dtype=jnp.float32)
    time = jax.random.uniform(time_key, shape=(batch.action.shape[0], 1, 1))
    noisy_actions = time * noise + (1 - time) * batch.action
    target_velocity = noise - batch.action
    predicted_velocity = model.run_action_head(cache=cache, action_t=noisy_actions, traj_time=time[:, 0, 0])
    return jnp.mean((predicted_velocity - target_velocity) ** 2)


def create_model(cfg: VLAConfig, *, rngs: nnx.Rngs) -> VLA:
    model = VLA(cfg, rngs=rngs)
    load_hf_vit(model.ViT, repo="google/vit-base-patch16-224")
    for _, module in nnx.iter_modules(model):
        if isinstance(module, (nnx.Linear, nnx.Conv)):
            module.dtype = jnp.bfloat16
    return model


def ignore_instruction(_: str) -> jax.Array:
    return jnp.empty((0,), dtype=jnp.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/pi0"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "train.log").exists():
        raise FileExistsError("Use a fresh output directory to preserve previous logs.")
    jax.config.update("jax_compilation_cache_dir", str(Path.home() / ".cache/jax"))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

    vla_config = VLAConfig(state_size=8, action_size=7)
    dataset = args.dataset if args.dataset is not None else Path(snapshot_download("lerobot/libero", repo_type="dataset"))
    config = TrainConfig(
        data_config=DataloaderConfig(
            path=dataset,
            resolution=(vla_config.vit.input_size, vla_config.vit.input_size),
            batch_size=32,
            action_horizon=vla_config.action_chunk_size,
            observation_features=("observation.images.image", "observation.images.image2"),
            action_features=(("action", NormalizationType.QUANTILE),),
            prio_features=(("observation.state", NormalizationType.QUANTILE),),
            tokenizer=ignore_instruction,
            seq_len=1,
            pad_token_id=0,
            observation_range=(-1.0, 1.0),
            seed=42,
        ),
        log_file=args.output / "train.log",
        model_type=create_model,
        model_kwargs={"cfg": vla_config},
        optimizer=optax.adam(learning_rate=1e-4),
        loss_fn=loss_fn,
        seed=42,
        n_epochs=args.epochs,
    )
    return train(config)


if __name__ == "__main__":
    main()
