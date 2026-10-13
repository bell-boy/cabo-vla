import dataclasses
import functools
import os

import flax.nnx as nnx
import jax
import optax
from einops import rearrange, repeat
from flax import struct
from jax import numpy as jnp

from cabo_vla.config import VLAConfig
from cabo_vla.data.dataloader import Batch
from cabo_vla.load_model import load_hf_vit
from cabo_vla.models.pi0 import VLA

jax.config.update("jax_compilation_cache_dir", os.path.expanduser("~/.cache/jax"))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


@struct.dataclass
class TrainConfig:
    vla: VLAConfig
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    seed: int = 42
    draws_per_action_chunk: int = 1  # https://abc.bot/


@struct.dataclass
class TrainState:
    model_graph: nnx.GraphDef
    model_state: nnx.State
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState
    key: jax.Array
    step: int = 0


@functools.partial(jax.jit, static_argnames="config", donate_argnums=(0,))
def train_step(state: TrainState, config: TrainConfig, batch: Batch):
    key, noise_key, t_key = jax.random.split(state.key, 3)

    # TODO: add actual forward and backward pass
    def loss_fn(model_state: nnx.State, batch: Batch):

        model: VLA = nnx.merge(state.model_graph, model_state)
        _, cache = model.encode_state_imgs(imgs=batch.observations, state=batch.prio, time=batch.timestamp)  # B N D

        x = batch.action
        assert config.draws_per_action_chunk == 1, "for now, we only support draws_per_action_chunk=1, see comment in train.py for more details"
        noise = jax.random.normal(
            noise_key, shape=(x.shape[0], config.draws_per_action_chunk, config.vla.action_chunk_size, config.vla.action_size), dtype=jnp.float32
        )  # batch draws horizon actions
        t = jax.random.uniform(
            t_key, shape=(x.shape[0], config.draws_per_action_chunk, 1, 1), minval=0, maxval=1
        )  # batch draws 1 1 (because we denoise each chunk at the same time)
        noise = rearrange(noise, "b d h a -> (b d) h a")
        t = rearrange(t, "b d 1 1 -> (b d) 1 1")
        x_t = noise * t + repeat(x, "b h a -> (b d) h a", d=config.draws_per_action_chunk)
        target = noise - x
        output = model.run_action_head(cache=cache, action_t=x_t, traj_time=t)
        # TODO: @Honyant: I'd like to have multiple draws of the noise per action so that we can reuse the same state/img prefix encoding for multiple action predictions. This emperically, as demonstrated by https://abc.bot/, improves training loss/gpu hour in their setup.
        # you could flatten dims, and in the action head, check that batch size must be a multiple of the cache batch size, then broadcast cache, but the impl for broadcasting attn, so for now, I'm just skipping this multiple draw idea. you could also have grouped queries then mask, but you'd need a proper kernel. either case, i think you need a custom kernel, but maybe there's a simpler way? for now assert draws_per_action_chunk=1, and we can add this later if we want to do multiple draws per action chunk. it's probably not that bad to implement it
        loss = jnp.mean((output - target) ** 2)
        return loss

    value, grads = jax.value_and_grad(loss_fn)(state.model_state, batch)
    updates, new_optimizer_state = state.optimizer.update(grads, state.optimizer_state, state.model_state)
    new_model_state = optax.apply_updates(state.model_state, updates)
    new_state = dataclasses.replace(state, model_state=new_model_state, optimizer_state=new_optimizer_state, key=key, step=state.step + 1)

    info = {"loss": value, "step": state.step}
    return new_state, info


def main():
    vla_config = VLAConfig()
    train_config = TrainConfig(vla=vla_config)
    init_key, data_key, train_key = jax.random.split(jax.random.key(train_config.seed), 3)

    vla = VLA(vla_config, rngs=nnx.Rngs(init_key))
    load_hf_vit(vla.ViT, repo="google/vit-base-patch16-224")
    for _, m in nnx.iter_modules(vla):
        if isinstance(m, (nnx.Linear, nnx.Conv)):
            m.dtype = jnp.bfloat16
    graphdef, state = nnx.split(vla)

    optimizer = optax.adam(learning_rate=train_config.learning_rate)
    optimizer_state = optimizer.init(state)

    train_state = TrainState(model_graph=graphdef, model_state=state, optimizer=optimizer, optimizer_state=optimizer_state, key=train_key)
    data_rngs = nnx.Rngs(data_key)  # only used outside jit, for the mock batches

    for epoch in range(train_config.num_epochs):
        for step in range(100):
            batch = Batch(
                observations=jax.random.normal(data_rngs(), (train_config.batch_size, 2, 3, 224, 224)),
                instruction=None,
                prio=jax.random.normal(data_rngs(), (train_config.batch_size, 32)),
                action=jax.random.normal(data_rngs(), (train_config.batch_size, train_config.vla.action_chunk_size, train_config.vla.action_size)),
                timestamp=jax.random.uniform(data_rngs(), (train_config.batch_size, 1)),
            )
            train_state, info = train_step(train_state, train_config, batch)
            print(f"Epoch {epoch}, Step {step}, Loss: {info['loss']}")


if __name__ == "__main__":
    main()
