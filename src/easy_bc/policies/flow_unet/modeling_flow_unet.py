from typing import Dict
import jax
from lerobot.utils.constants import ACTION, OBS_STATE
import optax
from typing_extensions import override

import chex
import jax.numpy as jnp
from flax import nnx

from easy_bc.policies.flow_unet.configuration_flow_unet import FlowUnetConfig
from easy_bc.policies.policy import BasePolicy
from easy_bc.policies.modules import (
    ConditionalUnet1D,
    Conv1DBlock,
    RGBEncoder,
    SinusoidalPosEmb,
)


class FlowUnetPolicy(BasePolicy):
    """FlowUnet policy model."""

    def __init__(self, config: FlowUnetConfig, rngs: nnx.Rngs):
        super().__init__()

        self.config = config

        action_dim = next(iter(self.config.output_features.values())).shape[0]
        images_shape = next(iter(config.image_features.values())).shape  # C, H, W
        state_dim = (
            config.robot_state_feature.shape[0] if config.robot_state_feature else 0
        )

        self.rgb_encoder = RGBEncoder(
            images_shape=images_shape,
            out_feature_dim=self.config.img_feature_dim,
            rngs=rngs,
        )

        # TODO: support env state features
        self.unet = ConditionalUnet1D(
            feature_dim=self.config.latent_dim,
            cond_dim=state_dim
            + self.config.img_feature_dim
            + self.config.time_embedding_dim,
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

        self.time_embedding = nnx.Sequential(
            SinusoidalPosEmb(self.config.time_embedding_dim),
            nnx.Linear(
                in_features=self.config.time_embedding_dim,
                out_features=self.config.time_embedding_dim * 4,
                rngs=rngs,
            ),
            nnx.swish,
            nnx.Linear(
                in_features=self.config.time_embedding_dim * 4,
                out_features=self.config.time_embedding_dim,
                rngs=rngs,
            ),
        )

    def pred_action_flow(
        self, x_t: jnp.ndarray, cond: jnp.ndarray, timestep: jnp.ndarray
    ):
        timestep_embed = self.time_embedding(timestep)  # B, time_embedding_dim

        cond = jnp.concatenate(
            [cond, timestep_embed], axis=-1
        )  # B, img_feature_dim + time_embedding_dim

        latent_actions = self.action_in_projection(x_t)  # B, H, latent_dim
        pred_latent_actions = self.unet(latent_actions, cond)  # B, H, latent_dim
        v_t = self.action_out_projection(pred_latent_actions)  # B, H, action_dim

        return v_t

    @override
    def compute_loss(
        self, batch: Dict[str, jnp.ndarray], rng: chex.PRNGKey
    ) -> chex.Array:
        noise_rng, time_rng = jax.random.split(rng, 2)

        img_key = next(iter(self.config.image_features.keys()))
        img = batch[img_key]  # B, C, H, W

        state = batch[OBS_STATE]

        actions = batch[ACTION]  # B, H, action_dim

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

        cond = jnp.concatenate(
            [state, img_feature], axis=-1
        )  # B, state_dim + img_feature_dim

        v_t = self.pred_action_flow(x_t, cond, time)  # B, H, action_dim

        loss = optax.l2_loss(predictions=v_t, targets=u_t).mean(axis=-1)

        return loss

    @override
    def sample_action(
        self, batch: Dict[str, jnp.ndarray], rng: chex.PRNGKey
    ) -> jnp.ndarray:
        img_key = next(iter(self.config.image_features.keys()))
        img = batch[img_key]  # B, C, H, W

        state = batch[OBS_STATE]

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

        cond = jnp.concatenate(
            [state, img_feature], axis=-1
        )  # B, state_dim + img_feature_dim

        def step(carry):
            x_t, t = carry

            time_batched = jnp.full((B,), t, dtype=dtype)
            v_t = self.pred_action_flow(x_t, cond, time_batched)  # B, H, action_dim

            return x_t - dt * v_t, t - dt

        def condition(carry):
            _, t = carry
            return t >= dt / 2.0

        x_0, _ = nnx.while_loop(condition, step, (noise, 1.0))

        return x_0
