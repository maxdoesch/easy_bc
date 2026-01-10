import einops
from flax import nnx

from easy_bc.policies.regression.configuration_regression import RegressionConfig


class EncoderStem(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, rngs: nnx.Rngs):
        """Conv 3×3, stride 2, out: output_dim → Norm → SiLU/ReLU"""
        super().__init__()

        self.conv = nnx.Conv(
            in_features=input_dim,
            out_features=output_dim,
            kernel_size=(3, 3),
            strides=(2, 2),
            rngs=rngs,
        )

        # TODO: maybe switch to GroupNorm
        self.norm = nnx.BatchNorm(
            num_features=output_dim, use_running_average=True, rngs=rngs
        )

    def __call__(self, x):
        x = self.conv(x)
        x = nnx.relu(x)
        x = self.norm(x)
        return x


class EncoderBlock(nnx.Module):
    def __init__(self, input_dim: int, output_dim: int, stride: int, rngs: nnx.Rngs):
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

        self.norm_dw = nnx.BatchNorm(
            num_features=output_dim, use_running_average=True, rngs=rngs
        )

        self.conv_pw = nnx.Conv(
            in_features=output_dim,
            out_features=output_dim,
            kernel_size=(1, 1),
            strides=(1, 1),
            rngs=rngs,
        )

        self.norm_pw = nnx.BatchNorm(
            num_features=output_dim, use_running_average=True, rngs=rngs
        )

    def __call__(self, x):
        x = self.conv_dw(x)
        x = self.norm_dw(x)
        x = nnx.relu(x)

        x = self.conv_pw(x)
        x = self.norm_pw(x)
        x = nnx.relu(x)

        return x


class RGBEncoder(nnx.Module):
    def __init__(self, config: RegressionConfig, rngs: nnx.Rngs):
        super().__init__()

        images_shape = next(iter(config.image_features.values())).shape  # C, H, W

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
            out_features=config.out_feature_dim,
            rngs=rngs,
        )

    def __call__(self, x):
        x = einops.rearrange(x, "B C H W -> B H W C")
        x = self.encoder_stem(x)
        for block in self.encoder_blocks:
            x = block(x)
        x = nnx.avg_pool(x, window_shape=x.shape[1:3])  # Global average pool
        x = einops.rearrange(x, "B 1 1 C -> B C")
        x = self.head(x)
        return x


class RegressionPolicy(nnx.Module):
    """A simple regression policy network."""

    def __init__(self, config: RegressionConfig, rngs: nnx.Rngs):
        super().__init__()

        self.config = config

        action_dim = next(iter(self.config.output_features.values())).shape[0]

        self.config.out_feature_dim = action_dim

        self.encoder = RGBEncoder(config, rngs)

    def __call__(self, x):
        img_key = next(iter(self.config.image_features.keys()))
        x = x[img_key]

        x = self.encoder(x)
        return x
