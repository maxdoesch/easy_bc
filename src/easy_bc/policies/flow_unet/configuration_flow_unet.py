from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamConfig


@PreTrainedConfig.register_subclass("flow_unet")
@dataclass
class FlowUnetConfig(PreTrainedConfig):
    """Configuration class for FlowUnetPolicy.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - "action" is required as an output key.

    Args:
        crop_shape: (H, W) shape to crop images to as a preprocessing step for the vision backbone. Must fit
            within the image size. If None, no cropping is done.
        down_dims: Feature dimension for each stage of temporal downsampling in the flow unet modeling Unet.
            You may provide a variable number of dimensions, therefore also controlling the degree of
            downsampling.
        kernel_size: The convolutional kernel size of the flow unet modeling Unet.
        n_groups: Number of groups used in the group norm of the Unet's convolutional blocks.
    """

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    horizon: int = 16

    # RGBEncoder
    img_feature_dim: int = 512

    # Unet.
    latent_dim: int = 64
    down_dims: tuple[int, ...] = (64, 128, 256)
    kernel_size: int = 5
    n_groups: int = 8

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    num_inference_steps: int = 10

    def __post_init__(self):
        super().__post_init__()

        pass

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if (
                    self.crop_shape[0] > image_ft.shape[1]
                    or self.crop_shape[1] > image_ft.shape[2]
                ):
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
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
