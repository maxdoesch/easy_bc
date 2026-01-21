from typing import Dict, Optional
from typing_extensions import override

import optax
import chex
import jax.numpy as jnp
from flax import nnx

from easy_bc.policies.modules import RGBEncoder
from easy_bc.policies.policy import BasePolicy
from easy_bc.policies.regression.configuration_regression import RegressionConfig


class RegressionPolicy(BasePolicy):
    """A simple regression policy network."""

    def __init__(self, config: RegressionConfig, rngs: nnx.Rngs):
        super().__init__()

        self.config = config

        action_dim = next(iter(self.config.output_features.values())).shape[0]
        images_shape = next(iter(config.image_features.values())).shape  # C, H, W

        out_feature_dim = action_dim

        self.encoder = RGBEncoder(images_shape, out_feature_dim, rngs)

    @override
    def compute_loss(
        self, batch: Dict[str, jnp.ndarray], rng: Optional[chex.PRNGKey] = None
    ) -> chex.Array:
        img_key = next(iter(self.config.image_features.keys()))
        x = batch[img_key]

        actions = batch["action"]

        pred_actions = self(x)

        # TODO: propely handle horizon dimension
        pred_actions = jnp.reshape(pred_actions, actions.shape)

        loss = optax.l2_loss(predictions=pred_actions, targets=actions)

        return loss

    @override
    def sample_action(
        self, batch: Dict[str, jnp.ndarray], rng: Optional[chex.PRNGKey] = None
    ) -> jnp.ndarray:
        img_key = next(iter(self.config.image_features.keys()))
        x = batch[img_key]

        pred_actions = self(x)
        return pred_actions

    def __call__(self, x):
        x = self.encoder(x)
        return x
