from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("flow_unet")
@dataclass
class FlowUnetConfig(PreTrainedConfig):
    """Configuration class for FlowUnetPolicy."""

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    image_resolution: tuple[int, int] | None = (224, 224)
    crop_shape: tuple[int, int] | None = None

    horizon: int = 16

    # RGBEncoder
    img_feature_dim: int = 64

    time_embedding_dim: int = 64
    latent_dim: int = 64

    # Unet.
    down_dims: tuple[int, ...] = (128, 256, 512)
    kernel_size: int = 5
    n_groups: int = 8

    num_inference_steps: int = 10

    def __post_init__(self):
        super().__post_init__()

        pass

    def get_optimizer_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

        # Check that all input images have the same shape.
        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

    def get_scheduler_preset(self) -> None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None
