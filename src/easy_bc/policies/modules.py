import einops
import jax.numpy as jnp
from flax import nnx


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
        cond_dim = cond_dim + 1  # for timestep embedding

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

    def __call__(self, x, cond, timestep):
        """
        x: [B, H, input_dim]
        cond: [B, cond_dim]
        timestep: [B,]
        """

        # TODO: Use a proper timestep embedding instead of just concatenating the raw timestep
        cond = jnp.concatenate([cond, timestep[:, None].astype(cond.dtype)], axis=-1)

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


class RGBEncoder(nnx.Module):
    def __init__(self, images_shape: tuple, out_feature_dim: int, rngs: nnx.Rngs):
        super().__init__()

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

        self.head = nnx.Linear(
            in_features=self.feature_dims[-1],
            out_features=out_feature_dim,
            rngs=rngs,
        )

    def __call__(self, x):
        x = einops.rearrange(x, "B C H W -> B H W C")
        x = self.encoder_stem(x)
        for block in self.encoder_blocks:
            x = block(x)

        # TODO: Replace the global average pooling with a spatial softmax pooling to maintain spatial information
        x = nnx.avg_pool(x, window_shape=x.shape[1:3])  # Global average pool
        x = einops.rearrange(x, "B 1 1 C -> B C")
        x = self.head(x)
        return x
