import abc
from typing import Dict, Optional

import chex
from flax import nnx
import jax.numpy as jnp


class BasePolicy(nnx.Module, abc.ABC):
    @abc.abstractmethod
    def compute_loss(
        self, batch: Dict[str, jnp.ndarray], rng: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        raise NotImplementedError

    @abc.abstractmethod
    def sample_action(self, batch: Dict[str, jnp.ndarray]) -> jnp.ndarray:
        raise NotImplementedError
