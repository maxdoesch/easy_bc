import abc
from typing import Dict, Optional

import chex
import jax.numpy as jnp
from flax import nnx


class BasePolicy(nnx.Module, abc.ABC):
    @abc.abstractmethod
    def compute_loss(
        self, batch: Dict[str, jnp.ndarray], rng: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_action(
        self, batch: Dict[str, jnp.ndarray], rng: Optional[chex.PRNGKey] = None
    ) -> jnp.ndarray:
        raise NotImplementedError
