from typing import Dict
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
    def compute_loss(self, batch: Dict[str, jnp.ndarray]) -> chex.Array:
        img_key = next(iter(self.config.image_features.keys()))
        x = batch[img_key]

        pred_actions = self.encoder(x)
        actions = batch["action"]

        loss = optax.l2_loss(predictions=pred_actions, targets=actions)

        return loss

    def __call__(self, x):
        img_key = next(iter(self.config.image_features.keys()))
        x = x[img_key]

        x = self.encoder(x)
        return x
