import functools
import os

import flax.nnx as nnx
import jax
import optax
from flax import struct

from cabo_vla.config import VLAConfig
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


@struct.dataclass
class TrainState:
    model_graph: nnx.GraphDef
    model_state: nnx.State
    optimizer: optax.GradientTransformation = struct.field(pytree_node=False)
    optimizer_state: optax.OptState
    rngs: nnx.Rngs
    step: int = 0


@functools.partial(jax.jit, static_argnames="config")
def train_step(state: TrainState, config: TrainConfig, rngs: nnx.Rngs, batch):
    model = nnx.merge(state.model_graph, state.model_state)
    rngs_step = rngs.step()
    random_action_target = jax.random.normal(rngs_step, (config.batch_size, config.vla.action_size))
    # TODO: add actual forward and backward pass
    info = {"loss": 0, "step": state.step, "random_action_target": random_action_target}
    return state, info


def main():
    # init model optimizer
    # init dataloader (mock dataloder for now)
    rngs = nnx.Rngs(0, step=0)
    vla_config = VLAConfig()
    vla = VLA(vla_config, rngs=rngs)
    load_hf_vit(vla.ViT, repo="google/vit-base-patch16-224")
    graphdef, state = nnx.split(vla)
    optimizer = optax.adam(learning_rate=1e-4)
    optimizer_state = optimizer.init(state)

    train_state = TrainState(model_graph=graphdef, model_state=state, optimizer=optimizer, optimizer_state=optimizer_state, rngs=rngs)
    train_config = TrainConfig(vla=vla_config)

    for epoch in range(10):
        for step in range(100):
            batch = {
                "imgs": jax.random.normal(rngs.step(), (32, 4, 3, 224, 224)),
                "state": jax.random.normal(rngs.step(), (32, 10)),
                "time": jax.random.normal(rngs.step(), (32,)),
            }
            train_state, info = train_step(train_state, train_config, rngs, batch)
            print(f"Epoch {epoch}, Step {step}, Loss: {info['loss']}")


if __name__ == "__main__":
    main()
