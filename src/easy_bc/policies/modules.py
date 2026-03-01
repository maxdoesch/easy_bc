from math import sqrt
from typing import Optional, cast

import einops
import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import PRNGKeyArray


def get_output_shape(module: nnx.Module, input_shape: tuple) -> tuple:
    """Utility function to get the output shape of a module given an input shape."""
    dummy_input = jnp.zeros((1, *input_shape))
    output = module(dummy_input)
    return output.shape[1:]


class SinusoidalPosEmb(nnx.Module):
    """
    Sinusoidal Positional Embedding
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        self.dim = dim

    def __call__(self, x):
        half_dim = self.dim // 2
        emb = jnp.log(10000) / (half_dim - 1)
        emb = jnp.exp(jnp.arange(half_dim) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
        return emb


class Conv1DBlock(nnx.Module):
    """
    1D Convolution Block: Conv1D → Norm → activation
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rngs: nnx.Rngs,
        kernel_size: int = 3,
        num_groups: int = 8,
    ):
        super().__init__()

        self.conv = nnx.Conv(
            in_features=input_dim,
            out_features=output_dim,
            kernel_size=(kernel_size,),
            rngs=rngs,
        )

        self.norm = nnx.GroupNorm(
            num_features=output_dim, num_groups=num_groups, rngs=rngs
        )

    def __call__(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = nnx.swish(x)
        return x


class ConditionalResidual1DBlock(nnx.Module):
    """
    Conditional Residual 1D Block: Conv1DBlock → FiLM → Conv1DBlock → Residual Add
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        cond_dim: int,
        rngs: nnx.Rngs,
        kernel_size: int = 3,
        num_groups: int = 8,
    ):
        super().__init__()

        self.conv1 = Conv1DBlock(
            input_dim=input_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            rngs=rngs,
        )

        self.cond_proj = nnx.Linear(
            in_features=cond_dim,
            out_features=2 * output_dim,
            rngs=rngs,
        )

        self.conv2 = Conv1DBlock(
            input_dim=output_dim,
            output_dim=output_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            rngs=rngs,
        )

        self.residual_conv = (
            nnx.Conv(
                in_features=input_dim,
                out_features=output_dim,
                kernel_size=(1,),
                rngs=rngs,
            )
            if input_dim != output_dim
            else nnx.identity
        )

    def __call__(self, x, cond):
        """
        x: [B, H, input_dim]
        cond: [B, cond_dim]
        """
        out = self.conv1(x)  # [B, H, output_dim]

        cond_out = self.cond_proj(cond)  # [B, 2 * output_dim]
        scale, shift = jnp.split(cond_out, 2, axis=-1)

        out = out * scale[:, None, :] + shift[:, None, :]

        out = self.conv2(out)

        x = self.residual_conv(x)

        return out + x


class ConditionalUnet1D(nnx.Module):
    """
    Conditional 1D U-Net
    """

    def __init__(
        self,
        feature_dim: int,
        cond_dim: int,
        down_dims: tuple[int, ...],
        kernel_size: int,
        num_groups: int,
        rngs: nnx.Rngs,
    ):
        super().__init__()

        down_dims = (feature_dim, *down_dims)

        self.down_blocks = nnx.List()
        i = 1
        for prev_dim, dim in zip(down_dims[:-1], down_dims[1:]):
            block = ConditionalResidual1DBlock(
                input_dim=prev_dim,
                output_dim=dim,
                cond_dim=cond_dim,
                kernel_size=kernel_size,
                num_groups=num_groups,
                rngs=rngs,
            )
            if i == len(down_dims) - 1:
                # No downsampling in the last block
                downsample = nnx.identity
            else:
                # TODO: downsampling may fail when H is not devisible by 2 enough times
                downsample = nnx.Conv(
                    in_features=dim,
                    out_features=dim,
                    kernel_size=(3,),
                    strides=(2,),
                    rngs=rngs,
                )
            self.down_blocks.append(nnx.List([block, downsample]))

            i += 1

        self.mid_block = ConditionalResidual1DBlock(
            input_dim=down_dims[-1],
            output_dim=down_dims[-1],
            cond_dim=cond_dim,
            kernel_size=kernel_size,
            num_groups=num_groups,
            rngs=rngs,
        )

        self.up_blocks = nnx.List()
        i = 1
        for prev_dim, dim in zip(reversed(down_dims[1:]), reversed(down_dims[:-1])):
            block = nnx.Sequential(
                ConditionalResidual1DBlock(
                    input_dim=prev_dim * 2,
                    output_dim=dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    num_groups=num_groups,
                    rngs=rngs,
                ),
                nnx.identity
                if i == len(down_dims) - 1
                else nnx.ConvTranspose(
                    in_features=dim,
                    out_features=dim,
                    kernel_size=(3,),
                    strides=(2,),
                    rngs=rngs,
                ),
            )
            self.up_blocks.append(block)

            i += 1

    def __call__(self, x, cond):
        """
        x: [B, H, input_dim]
        cond: [B, cond_dim]
        """

        skip_connections = []
        for block, downsample in self.down_blocks:
            x = block(x, cond)
            skip_connections.append(x)
            x = downsample(x)

        x = self.mid_block(x, cond)

        for block in self.up_blocks:
            skip_x = skip_connections.pop()
            x = jnp.concatenate([x, skip_x], axis=-1)
            x = block(x, cond)

        return x


class EncoderStem(nnx.Module):
    def __init__(
        self, input_dim: int, output_dim: int, rngs: nnx.Rngs, num_groups: int = 8
    ):
        """Conv 3×3, stride 2, out: output_dim → Norm → activation"""
        super().__init__()

        self.conv = nnx.Conv(
            in_features=input_dim,
            out_features=output_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            rngs=rngs,
        )

        self.norm = nnx.GroupNorm(
            num_features=output_dim, num_groups=num_groups, rngs=rngs
        )

    def __call__(self, x):
        x = self.conv(x)
        x = nnx.swish(x)
        x = self.norm(x)
        return x


class EncoderBlock(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        stride: int,
        rngs: nnx.Rngs,
        num_groups: int = 8,
    ):
        """
        Depthwise Conv 3×3 (groups = channels), stride s → Norm → activation
        Pointwise Conv 1×1 (mix channels) → Norm → activation
        """
        super().__init__()

        self.conv_dw = nnx.Conv(
            in_features=input_dim,
            out_features=output_dim,
            kernel_size=(3, 3),
            strides=(stride, stride),
            feature_group_count=input_dim,  # depthwise
            rngs=rngs,
        )

        self.norm_dw = nnx.GroupNorm(
            num_features=output_dim, num_groups=num_groups, rngs=rngs
        )

        self.conv_pw = nnx.Conv(
            in_features=output_dim,
            out_features=output_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            rngs=rngs,
        )
        self.norm_pw = nnx.GroupNorm(
            num_features=output_dim, num_groups=num_groups, rngs=rngs
        )

    def __call__(self, x):
        x = self.conv_dw(x)
        x = self.norm_dw(x)
        x = nnx.swish(x)

        x = self.conv_pw(x)
        x = self.norm_pw(x)
        x = nnx.swish(x)

        return x


class SpatialSoftmax(nnx.Module):
    """
    Spatial Soft Argmax operation described in
    "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    """

    def __init__(self, input_shape: tuple, num_keypoints: int, rngs: nnx.Rngs):
        """
        input_shape: (H, W, C)
        num_keypoints: int
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_h, self._in_w, self._in_c = input_shape

        self.projection = nnx.Conv(
            in_features=self._in_c,
            out_features=num_keypoints,
            kernel_size=(1, 1),
            rngs=rngs,
        )
        self._out_c = num_keypoints

        pos_x, pos_y = jnp.meshgrid(
            jnp.linspace(-1.0, 1.0, self._in_w),
            jnp.linspace(-1.0, 1.0, self._in_h),  # (H, W), (H, W)
        )

        pos_x = einops.rearrange(pos_x, "H W -> (H W) 1")  # (H*W, 1)
        pos_y = einops.rearrange(pos_y, "H W -> (H W) 1")  # (H*W, 1)

        self.pos_grid = jnp.concatenate([pos_x, pos_y], axis=1)  # (H*W, 2)

    def __call__(self, x):
        """
        x: [B, H, W, C]
        returns: [B, num_keypoints * 2]
        """

        x = self.projection(x)  # [B, H, W, num_keypoints]
        b, h, w, n = x.shape

        x = einops.rearrange(x, "B H W N -> (B N) (H W)")  # [B * num_keypoints, H * W]

        attn = nnx.softmax(x, axis=-1)  # [B * num_keypoints, H * W]

        expected_xy = attn @ self.pos_grid  # [B * num_keypoints, 2]

        feature_keypoints = einops.rearrange(
            expected_xy, "(B N) D -> B (N D)", N=n, D=2
        )  # [B, num_keypoints * 2]

        return feature_keypoints


class DinoRGBEncoder(nnx.Module):
    def __init__(self, images_shape: tuple, out_feature_dim: int, rngs: nnx.Rngs):
        """
        images_shape: (C, H, W)
        out_feature_dim: int, must be even
        """
        super().__init__()

        try:
            from equimo.io import load_model
            from equimo.models import VisionTransformer
        except ImportError as e:
            raise ImportError("DinoRGBEncoder requires equimo.") from e

        dino_model = load_model(
            "vit", identifier="dinov3_vits16_pretrain_lvd1689m", inference_mode=True
        )

        # cast to VisionTransforer
        self.dino_model: VisionTransformer = cast(VisionTransformer, dino_model)
        self.dino_model = nnx.data(self.dino_model)

        assert out_feature_dim % 2 == 0, "out_feature_dim must be even."

        self.images_shape = images_shape
        feature_dim = self.dino_model.dim
        self.patch_hw: int = int(sqrt(self.dino_model.num_patches))

        assert (
            self.images_shape[1] % self.patch_hw == 0
            and self.images_shape[2] % self.patch_hw == 0
        ), f"Image height and width must be divisible by patch size {self.patch_hw}."

        feature_map_shape = (
            self.patch_hw,
            self.patch_hw,
            feature_dim,
        )

        self.spatial_softmax = SpatialSoftmax(
            input_shape=feature_map_shape,
            num_keypoints=out_feature_dim // 2,
            rngs=rngs,
        )

        self.head = nnx.Linear(
            in_features=out_feature_dim,
            out_features=out_feature_dim,
            rngs=rngs,
        )

    def __call__(self, x, rng: PRNGKeyArray):
        """
        x: [B, C, H, W]
        returns: [B, out_feature_dim]
        """

        assert x.shape[1:] == self.images_shape, (
            f"Expected input shape (C, H, W) = {self.images_shape}, got {x.shape[1:]}"
        )

        x = jax.vmap(self.dino_model.forward_features, in_axes=(0, None, None))(
            x, rng, True
        )
        x = x["x_norm_patchtokens"]
        x = jax.lax.stop_gradient(x)

        x = einops.rearrange(
            x, "B (H W) C -> B H W C", H=self.patch_hw, W=self.patch_hw
        )

        x = self.spatial_softmax(x)  # [B, out_feature_dim]
        x = self.head(x)  # [B, out_feature_dim]

        return x


class RGBEncoder(nnx.Module):
    def __init__(self, images_shape: tuple, out_feature_dim: int, rngs: nnx.Rngs):
        """
        images_shape: (C, H, W)
        out_feature_dim: int, must be even
        """
        super().__init__()

        assert out_feature_dim % 2 == 0, "out_feature_dim must be even."

        self.feature_dims = [32, 64, 128, 128, 256, 256, 512]

        self.encoder_stem = EncoderStem(
            input_dim=images_shape[0], output_dim=self.feature_dims[0], rngs=rngs
        )

        self.encoder_blocks = nnx.List()
        for prev_feature_dim, feature_dim in zip(
            self.feature_dims[:-1], self.feature_dims[1:]
        ):
            stride = feature_dim // prev_feature_dim
            block = EncoderBlock(
                input_dim=prev_feature_dim,
                output_dim=feature_dim,
                stride=stride,
                rngs=rngs,
            )

            self.encoder_blocks.append(block)

        feature_map_shape = get_output_shape(
            nnx.Sequential(self.encoder_stem, *self.encoder_blocks),
            (*images_shape[1:], images_shape[0]),
        )  # H, W, C

        self.spatial_softmax = SpatialSoftmax(
            input_shape=feature_map_shape,
            num_keypoints=out_feature_dim // 2,
            rngs=rngs,
        )

        self.head = nnx.Linear(
            in_features=out_feature_dim,
            out_features=out_feature_dim,
            rngs=rngs,
        )

    def __call__(self, x, rng: Optional[PRNGKeyArray] = None):
        """
        x: [B, C, H, W]
        returns: [B, out_feature_dim]
        """

        x = einops.rearrange(x, "B C H W -> B H W C")
        x = self.encoder_stem(x)
        for block in self.encoder_blocks:
            x = block(x)

        x = self.spatial_softmax(x)  # [B, out_feature_dim]

        x = self.head(x)  # [B, out_feature_dim]
        return x
