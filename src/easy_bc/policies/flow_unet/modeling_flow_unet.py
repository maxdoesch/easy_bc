from typing import Dict
import jax
import optax
from typing_extensions import override

import chex
import jax.numpy as jnp
from flax import nnx

from easy_bc.policies.flow_unet.configuration_flow_unet import FlowUnetConfig
from easy_bc.policies.policy import BasePolicy
from easy_bc.policies.modules import ConditionalUnet1D, Conv1DBlock, RGBEncoder


class FlowUnetPolicy(BasePolicy):
    """FlowUnet policy model."""

    def __init__(self, config: FlowUnetConfig, rngs: nnx.Rngs):
        super().__init__()

        self.config = config

        action_dim = next(iter(self.config.output_features.values())).shape[0]
        images_shape = next(iter(config.image_features.values())).shape  # C, H, W

        self.rgb_encoder = RGBEncoder(
            images_shape=images_shape,
            out_feature_dim=self.config.img_feature_dim,
            rngs=rngs,
        )

        # TODO: support env state features
        self.unet = ConditionalUnet1D(
            feature_dim=self.config.latent_dim,
            cond_dim=self.config.img_feature_dim,
            down_dims=self.config.down_dims,
            kernel_size=self.config.kernel_size,
            num_groups=self.config.n_groups,
            rngs=rngs,
        )

        self.action_in_projection = nnx.Sequential(
            nnx.Conv(
                in_features=action_dim,
                out_features=self.config.latent_dim,
                kernel_size=(1,),
                strides=(1,),
                rngs=rngs,
            ),
            Conv1DBlock(
                input_dim=self.config.latent_dim,
                output_dim=self.config.latent_dim,
                kernel_size=self.config.kernel_size,
                num_groups=self.config.n_groups,
                rngs=rngs,
            ),
        )

        self.action_out_projection = nnx.Sequential(
            Conv1DBlock(
                input_dim=self.config.latent_dim,
                output_dim=self.config.latent_dim,
                kernel_size=self.config.kernel_size,
                num_groups=self.config.n_groups,
                rngs=rngs,
            ),
            nnx.Conv(
                in_features=self.config.latent_dim,
                out_features=action_dim,
                kernel_size=(1,),
                strides=(1,),
                rngs=rngs,
            ),
        )

    def pred_action_flow(
        self, x_t: jnp.ndarray, cond: jnp.ndarray, timestep: jnp.ndarray
    ):
        latent_actions = self.action_in_projection(x_t)  # B, H, latent_dim
        pred_latent_actions = self.unet(
            latent_actions, cond, timestep
        )  # B, H, latent_dim
        v_t = self.action_out_projection(pred_latent_actions)  # B, H, action_dim

        return v_t

    @override
    def compute_loss(
        self, batch: Dict[str, jnp.ndarray], rng: chex.PRNGKey
    ) -> chex.Array:
        noise_rng, time_rng = jax.random.split(rng, 2)
        img_key = next(iter(self.config.image_features.keys()))
        img = batch[img_key]  # B, C, H, W

        action_key = next(iter(self.config.output_features.keys()))
        actions = batch[action_key]  # B, H, action_dim

        B, H, _ = actions.shape

        assert H == self.config.horizon

        noise = jax.random.normal(
            noise_rng, actions.shape, dtype=actions.dtype
        )  # B, H, action_dim
        time = (
            jax.random.beta(time_rng, 1.5, 1, (B,), dtype=actions.dtype) * 0.999 + 0.001
        )  # B,

        time_expanded = time[..., None, None]  # B, 1, 1
        x_t = time_expanded * noise + (1 - time_expanded) * actions  # B, H, action_dim
        u_t = noise - actions  # B, H, action_dim

        img_feature = self.rgb_encoder(img)  # B, img_feature_dim

        v_t = self.pred_action_flow(x_t, img_feature, time)  # B, H, action_dim

        loss = optax.l2_loss(predictions=v_t, targets=u_t).mean(axis=-1)

        return loss

    @override
    def sample_action(
        self, batch: Dict[str, jnp.ndarray], rng: chex.PRNGKey
    ) -> jnp.ndarray:
        img_key = next(iter(self.config.image_features.keys()))
        img = batch[img_key]  # B, C, H, W

        B, H, action_dim = (
            img.shape[0],
            self.config.horizon,
            self.config.action_feature.shape[0],  # pyright: ignore
        )
        dtype = img.dtype

        dt = 1.0 / self.config.num_inference_steps

        noise = jax.random.normal(
            rng,
            (B, H, action_dim),
            dtype=dtype,
        )

        img_feature = self.rgb_encoder(img)  # B, img_feature_dim

        def step(carry):
            x_t, t = carry

            time_batched = jnp.full((B,), t, dtype=dtype)
            v_t = self.pred_action_flow(
                x_t, img_feature, time_batched
            )  # B, H, action_dim

            return x_t - dt * v_t, t - dt

        def cond(carry):
            _, t = carry
            return t >= dt / 2.0

        x_0, _ = nnx.while_loop(cond, step, (noise, 1.0))

        return x_0
