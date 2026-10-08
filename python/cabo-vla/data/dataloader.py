"""JAX batching for LeRobot datasets, without PyTorch's DataLoader."""

from collections.abc import Iterator, Mapping, Sequence
import operator
from typing import Any
from dataclasses import dataclass
from jaxtyping import Float

import jax
import numpy as np


@dataclass
class Batch:
    observations: Float[jax.Array, "Batch Camera Channel Height Width"]
    instruction: str
    action: Float[jax.Array, "Batch Horizon Action"]
    prio: Float[jax.Array, "Batch State"]
