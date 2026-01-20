import pytest
import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

from easy_bc.policies.flow_unet.modeling_flow_unet import FlowUnetPolicy
from easy_bc.policies.flow_unet.configuration_flow_unet import FlowUnetConfig


def _keep_keys(config) -> set[str]:
    return set(config.input_features) | set(config.output_features)


def _as_batched_jnp(x):
    return jnp.expand_dims(jnp.asarray(x), axis=0)


def _jit_parity_method(obj, method_name: str, batch, *, rtol=1e-4, atol=1e-5):
    gdef, state = nnx.split(obj)
    eager = nnx.merge(gdef, state)
    compiled = nnx.merge(gdef, state)

    eager_fn = getattr(eager, method_name)
    compiled_fn = getattr(compiled, method_name)

    rng = jax.random.PRNGKey(0)

    y_eager = eager_fn(batch, rng)
    y_jit = jax.jit(compiled_fn)(batch, rng)

    assert y_eager.shape == y_jit.shape
    assert jnp.allclose(y_eager, y_jit, rtol=rtol, atol=atol)
    return y_eager, y_jit


@pytest.fixture(scope="session")
def dataset_metadata():
    return LeRobotDatasetMetadata("lerobot/pusht")


@pytest.fixture(scope="session")
def flow_unet_config(dataset_metadata) -> FlowUnetConfig:
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {
        k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {k: ft for k, ft in features.items() if k not in output_features}

    return FlowUnetConfig(
        input_features=input_features,
        output_features=output_features,
        img_feature_dim=32,
        latent_dim=64,
        down_dims=(64, 128),
        kernel_size=5,
        n_groups=8,
    )


@pytest.fixture(scope="session")
def dataset(dataset_metadata: LeRobotDatasetMetadata, flow_unet_config: FlowUnetConfig):
    return LeRobotDataset(
        "lerobot/pusht",
        delta_timestamps={
            key: [i / dataset_metadata.fps for i in range(flow_unet_config.horizon)]
            for key in flow_unet_config.output_features.keys()
        },
    )


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


@pytest.fixture
def policy(flow_unet_config: FlowUnetConfig, rngs: nnx.Rngs) -> FlowUnetPolicy:
    return FlowUnetPolicy(config=flow_unet_config, rngs=rngs)


@pytest.fixture
def batch(flow_unet_config: FlowUnetConfig, dataset) -> dict:
    keys = _keep_keys(flow_unet_config)
    sample = {k: v for k, v in dataset[0].items() if k in keys}
    return jax.tree_util.tree_map(_as_batched_jnp, sample)


def test_flow_unet_policy_compute_loss(policy: FlowUnetPolicy, batch: dict):
    action_key = next(iter(policy.config.output_features.keys()))
    action = batch[action_key]

    print(action.shape)

    B, H, _ = action.shape

    rng = jax.random.PRNGKey(0)
    loss = policy.compute_loss(batch, rng)

    assert loss.shape == (1, H)
    assert jnp.isfinite(loss).all()


def test_flow_unet_policy_compute_loss_jittable(policy: FlowUnetPolicy, batch: dict):
    _jit_parity_method(policy, "compute_loss", batch, rtol=1e-4, atol=1e-5)


def test_flow_unet_policy_sample_action(policy: FlowUnetPolicy, batch: dict):
    img_key = next(iter(policy.config.image_features.keys()))
    img = batch[img_key]
    B, H, action_dim = (
        img.shape[0],
        policy.config.horizon,
        policy.config.action_feature.shape[0],  # pyright: ignore
    )

    rng = jax.random.PRNGKey(0)
    actions = policy.sample_action(batch, rng)

    assert actions.shape == (B, H, action_dim)
    assert jnp.isfinite(actions).all()


def test_flow_unet_policy_sample_action_jittable(policy: FlowUnetPolicy, batch: dict):
    _jit_parity_method(policy, "sample_action", batch, rtol=1e-4, atol=1e-5)


def test_flow_unet_policy_checkpointing(policy: FlowUnetPolicy, batch: dict):
    options = ocp.CheckpointManagerOptions()
    with ocp.CheckpointManager(
        ocp.test_utils.erase_and_create_empty("/tmp/ckpt1/"),
        options=options,
    ) as mngr:
        graphdef, policy_state = nnx.split(policy)

        mngr.save(0, args=ocp.args.StandardSave(policy_state))  # pyright: ignore

        restored_state = mngr.restore(
            0,
            args=ocp.args.StandardRestore(policy_state),  # pyright: ignore
        )

        restored_policy = nnx.merge(graphdef, restored_state)

    rng = jax.random.PRNGKey(0)

    action_orig = policy.sample_action(batch, rng)
    action_restored = restored_policy.sample_action(batch, rng)

    assert jnp.allclose(action_orig, action_restored)
