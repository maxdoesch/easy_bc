from typing import Any, Type

from easy_bc.policies.flow_unet.configuration_flow_unet import FlowUnetConfig
from easy_bc.policies.regression.configuration_regression import RegressionConfig
from easy_bc.policies.flow_unet.modeling_flow_unet import FlowUnetPolicy
from easy_bc.policies.regression.modeling_regression import RegressionPolicy
from easy_bc.policies.flow_unet.processors_flow_unet import (
    make_processors_flow_unet_pre_post_processors,
)
from easy_bc.policies.regression.processors_regression import (
    make_processors_regression_pre_post_processors,
)

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.processor import PolicyAction, PolicyProcessorPipeline


def make_policy_config(
    name: str, dataset_metadata: LeRobotDatasetMetadata, **kwargs: Any
) -> PreTrainedConfig:
    """Factory method to create policy configurations.

    Args:
        name: The name of the policy configuration to create.
        **kwargs: Keyword arguments passed to the config constructor.

    Returns:
        An instance of the requested policy configuration.

    Raises:
        ValueError: If the specified policy configuration name is not recognized.
    """

    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    policy_configs: dict[str, Type[PreTrainedConfig]] = {
        "regression": RegressionConfig,
        "flow_unet": FlowUnetConfig,
    }

    try:
        return policy_configs[name](
            input_features=input_features, output_features=output_features, **kwargs
        )
    except KeyError as e:
        raise ValueError(f"Unknown policy config name: {name}") from e


def make_policy(config: PreTrainedConfig, **kwargs: Any) -> Any:
    """Factory method to create policy instances.

    Args:
        config: The policy configuration instance.
        **kwargs: Keyword arguments passed to the policy constructor.

    Returns:
        An instance of the requested policy.

    Raises:
        ValueError: If the specified policy configuration type is not recognized.
    """

    policy_classes: dict[Type[PreTrainedConfig], Type[Any]] = {
        RegressionConfig: RegressionPolicy,
        FlowUnetConfig: FlowUnetPolicy,
    }

    policy_class = policy_classes.get(type(config))
    if policy_class is None:
        raise ValueError(f"Unknown policy config type: {type(config)}")

    return policy_class(config, **kwargs)


def make_pre_post_processors(
    config: PreTrainedConfig, dataset_metadata: LeRobotDatasetMetadata
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    policy_processor_factories: dict[Type[PreTrainedConfig], Any] = {
        RegressionConfig: make_processors_regression_pre_post_processors,
        FlowUnetConfig: make_processors_flow_unet_pre_post_processors,
    }

    factory = policy_processor_factories.get(type(config))
    if factory is None:
        raise ValueError(f"Unknown policy config type: {type(config)}")
    return factory(
        config,
        dataset_stats=dataset_metadata.stats,  # pyright: ignore
    )
