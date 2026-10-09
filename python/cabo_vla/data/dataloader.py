"""JAX batching for LeRobot datasets, without PyTorch's DataLoader."""

from dataclasses import dataclass

import jax
from jaxtyping import Float


@dataclass
class Batch:
    observations: Float[jax.Array, "Batch Camera Channel Height Width"]
    instruction: str
    action: Float[jax.Array, "Batch Horizon Action"]
    prio: Float[jax.Array, "Batch State"]
    timestamp: Float[jax.Array, "Batch"]
