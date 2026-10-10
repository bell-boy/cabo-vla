import os

import jax
import jax.numpy as jnp
from flax import nnx

from cabo_vla.config import ViTConfig, VLAConfig
from cabo_vla.load_model import load_hf_vit
from cabo_vla.models.pi0 import VLA
from cabo_vla.models.vit import ViT

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def test_vit():
    vit_config = ViTConfig()
    vit = ViT(vit_config, rngs=nnx.Rngs(0))
    load_hf_vit(vit, repo="google/vit-base-patch16-224")
    sample_input = jnp.ones((16, 3, vit_config.input_size, vit_config.input_size))
    print(f"Sample input shape: {sample_input.shape}")
    graphdef, state = nnx.split(vit)

    @jax.jit
    def forward(state, x):
        return nnx.merge(graphdef, state)(x)

    output = forward(state, sample_input)
    print(f"Output shape: {output.shape}")


def test_vla():
    vla_config = VLAConfig()
    vla = VLA(vla_config, rngs=nnx.Rngs(0))

    @jax.jit
    def forward(state, imgs, st, t, action_t):
        model = nnx.merge(graphdef, state)
        _, cache = model.encode_state_imgs(imgs, st, t)
        dt = 0.1

        def action_step(action_t, diff_idx):
            out = model.run_action_head(cache=cache, action_t=action_t, traj_time=t)
            return action_t + dt * out, None

        return jax.lax.scan(action_step, action_t, jnp.arange(10))[0]

    load_hf_vit(vla.ViT, repo="google/vit-base-patch16-224")

    batch_size = 16

    sample_imgs = jnp.ones((batch_size, 4, 3, vla_config.vit.input_size, vla_config.vit.input_size))  # Batch of 16, 4 cameras
    sample_state = jnp.ones((batch_size, vla_config.state_size))
    sample_time = jnp.ones((batch_size,))

    graphdef, state = nnx.split(vla)

    print(f"Sample images shape: {sample_imgs.shape}")
    print(f"Sample state shape: {sample_state.shape}")
    print(f"Sample time shape: {sample_time.shape}")

    action_t = jnp.ones((batch_size, vla_config.action_chunk_size, vla_config.action_size))

    out = forward(state, sample_imgs, sample_state, sample_time, action_t)

    print(f"Action output shape: {out.shape}")


def main():
    test_vit()
    test_vla()


if __name__ == "__main__":
    main()
