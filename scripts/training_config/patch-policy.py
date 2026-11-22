import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import optax
from flax import nnx, serialization

from cabo_vla.config import ViTConfig
from cabo_vla.data.dataloader import (
    Batch,
    DataloaderConfig,
    NormalizationType,
    PaddingType,
)
from cabo_vla.load_model import load_hf_vit
from cabo_vla.models.diffusion_policy import DiffusionPolicy
from cabo_vla.models.vit import ViT
from cabo_vla.train import TrainConfig, train


def loss_fn(
    params: nnx.State,
    frozen_state: nnx.State,
    graphdef: nnx.GraphDef,
    batch: Batch,
    *,
    key: jax.Array,
) -> jax.Array:
    time_key, noise_key = jax.random.split(key)
    model = nnx.merge(graphdef, params, frozen_state)
    time = jax.random.uniform(time_key, shape=(batch.action.shape[0],), minval=0.0, maxval=1.0)
    noise = jax.random.normal(noise_key, shape=batch.action.shape, dtype=batch.action.dtype)
    # Match sampling: Gaussian noise at time 0, clean actions at time 1.
    interpolation_time = time[:, None, None]
    noisy_actions = (1 - interpolation_time) * noise + interpolation_time * batch.action
    target_velocity = batch.action - noise
    predicted_velocity = model(batch.observations, noisy_actions, time)
    return jnp.mean((predicted_velocity - target_velocity) ** 2)


def ignore_instruction(_: str) -> jax.Array:
    return jnp.empty((0,), dtype=jnp.int32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/lerobot_libero"))
    parser.add_argument("--output", type=Path, default=Path("runs/patch-policy"))
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "train.log").exists():
        raise FileExistsError("Use a fresh output directory to preserve previous logs.")
    vit_revision = "3f49326eb077187dfe1c2a2bb15fbd74e6ab91e3"
    seed = 0
    vit_config = ViTConfig()
    _, vision_key, _ = jax.random.split(jax.random.key(seed), 3)
    vision_encoder = ViT(cfg=vit_config, cls_out_params=None, rngs=nnx.Rngs(params=vision_key))
    load_hf_vit(vision_encoder, revision=vit_revision)
    config = TrainConfig(
        data_config=DataloaderConfig(
            path=args.dataset,
            observation_features=(
                "observation.images.image",
                "observation.images.image2",
            ),
            action_features=(("action", NormalizationType.MEANVAR),),
            prio_features=(),
            tokenizer=ignore_instruction,
            seq_len=1,
            pad_token_id=0,
            resolution=(vit_config.input_size, vit_config.input_size),
            batch_size=4,
            observation_range=(-1, 1),
            padding_type=PaddingType.ZERO,
            action_horizon=10,
            seed=seed,
            episode_indices=tuple(range(args.episodes)),
        ),
        log_file=args.output / "train.log",
        model_type=DiffusionPolicy,
        model_kwargs={
            "vision_encoder": vision_encoder,
            "action_dim": 7,
            "action_horizon": 10,
            "num_cameras": 2,
        },
        optimizer=optax.adam(learning_rate=1e-3),
        loss_fn=loss_fn,
        frozen_weight_filter=nnx.PathContains("vision_encoder"),
        seed=seed,
        n_epochs=args.epochs,
    )

    (args.output / "config.json").write_text(
        json.dumps(
            {
                **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "seed": seed,
                "vit_revision": vit_revision,
                "batch_size": 4,
                "action_horizon": 10,
                "learning_rate": 1e-3,
                "optimizer": "adam",
                "vision_encoder_frozen": True,
                "policy": "DiffusionPolicy",
                "d_model": 256,
                "n_layers": 4,
                "n_heads": 8,
                "d_hidden": 1024,
                "episode_indices": list(range(args.episodes)),
                "devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
    )
    state = train(config)
    (args.output / "last.msgpack").write_bytes(
        serialization.to_bytes(
            {
                "params": nnx.to_pure_dict(state.params),
                "key": jax.random.key_data(state.key),
            }
        )
    )


if __name__ == "__main__":
    main()
