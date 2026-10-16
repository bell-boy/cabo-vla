import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Protocol

import jax
import optax
from flax import nnx
from tqdm import tqdm, trange

from cabo_vla.data.dataloader import (
    Batch,
    Dataloader,
    DataloaderConfig,
    DataloaderState,
)


class LossFn(Protocol):
    def __call__(
        self,
        params: nnx.State,
        frozen_state: nnx.State,
        graphdef: nnx.GraphDef,
        batch: Batch,
        *,
        key: jax.Array,
    ) -> jax.Array: ...


@dataclass
class TrainConfig:
    data_config: DataloaderConfig
    log_file: Path
    model_type: Callable[..., nnx.Module]
    model_kwargs: dict
    optimizer: optax.GradientTransformation
    loss_fn: LossFn
    seed: int = 0
    n_epochs: int = 1
    frozen_weight_filter: nnx.filterlib.Filter = lambda _, __: False


@dataclass
class TrainState:
    graphdef: nnx.GraphDef
    params: nnx.State
    frozen_state: nnx.State
    optimizer_state: optax.OptState
    dataloader_state: DataloaderState
    key: jax.Array


class TrainStepOutput(NamedTuple):
    loss: jax.Array
    params: nnx.State
    optimizer_state: optax.OptState
    key: jax.Array


def train(config: TrainConfig):
    key = jax.random.key(config.seed)
    key, _, model_key = jax.random.split(key, 3)
    logging.basicConfig(filename=config.log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info(f"Loading data at {config.data_config.path}")

    dataloader = Dataloader(config.data_config)

    @functools.partial(jax.jit, static_argnums=0, donate_argnums=1)
    def train_step(graphdef, params, frozen_state, optimizer_state, batch, key) -> TrainStepOutput:
        key, loss_key = jax.random.split(key)
        loss, grad = jax.value_and_grad(config.loss_fn)(params, frozen_state, graphdef, batch, key=loss_key)
        updates, optimizer_state = config.optimizer.update(grad, optimizer_state, params=params)
        new_params: nnx.State = optax.apply_updates(params, updates)  # type: ignore
        return TrainStepOutput(loss, new_params, optimizer_state, key)

    model = config.model_type(**config.model_kwargs, rngs=nnx.Rngs(params=model_key))
    graphdef, params, frozen_state = nnx.split(model, nnx.All(nnx.Param, nnx.Not(config.frozen_weight_filter)), ...)
    optimizer_state = config.optimizer.init(params)

    state = TrainState(
        graphdef=graphdef,
        params=params,
        frozen_state=frozen_state,
        optimizer_state=optimizer_state,
        dataloader_state=dataloader.get_state(),
        key=key,
    )

    for epoch in trange(config.n_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.n_epochs}")
        for batch in tqdm(dataloader, leave=False):
            train_output = train_step(
                state.graphdef,
                state.params,
                frozen_state,
                state.optimizer_state,
                batch,
                state.key,
            )
            state = TrainState(
                graphdef=state.graphdef,
                params=train_output.params,
                frozen_state=frozen_state,
                optimizer_state=train_output.optimizer_state,
                dataloader_state=dataloader.get_state(),
                key=train_output.key,
            )
            logger.info(f"Loss: {train_output.loss.item()}")
    logger.info("Training complete.")
    return state


if __name__ == "__main__":
    from cabo_vla.training_configs.pi0 import main

    main()
