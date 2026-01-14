import pytest

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

from easy_bc.policies.regression.modeling_regression import RegressionPolicy
from easy_bc.policies.regression.configuration_regression import RegressionConfig


def _keep_keys(config: RegressionConfig) -> set[str]:
    return set(config.input_features) | set(config.output_features)


def _action_dim(config: RegressionConfig) -> int:
    return next(iter(config.output_features.values())).shape[0]


def _as_batched_jnp(x):
    return jnp.expand_dims(jnp.asarray(x), axis=0)


def _jit_parity(module, x, *, rtol=1e-4, atol=1e-5):
    gdef, state = nnx.split(module)
    eager = nnx.merge(gdef, state)
    compiled = nnx.merge(gdef, state)
    y_eager = eager(x)
    y_jit = jax.jit(compiled)(x)
    assert y_eager.shape == y_jit.shape
    assert jnp.allclose(y_eager, y_jit, rtol=rtol, atol=atol)
    return y_eager, y_jit


@pytest.fixture(scope="session")
def dataset_metadata():
    return LeRobotDatasetMetadata("lerobot/pusht")


@pytest.fixture(scope="session")
def regression_config(dataset_metadata) -> RegressionConfig:
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        k: ft for k, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {k: ft for k, ft in features.items() if k not in output_features}
    return RegressionConfig(
        input_features=input_features, output_features=output_features
    )


@pytest.fixture(scope="session")
def dataset():
    return LeRobotDataset("lerobot/pusht")


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


@pytest.fixture
def policy(regression_config: RegressionConfig, rngs) -> RegressionPolicy:
    return RegressionPolicy(config=regression_config, rngs=rngs)


@pytest.fixture
def batch(regression_config: RegressionConfig, dataset) -> dict:
    keys = _keep_keys(regression_config)
    sample = {k: v for k, v in dataset[0].items() if k in keys}
    return jax.tree_util.tree_map(_as_batched_jnp, sample)


def test_regression_policy_forward_pass(policy: RegressionPolicy, batch: dict):
    action = policy(batch)
    assert action.shape == (1, _action_dim(policy.config))


def test_regression_policy_jittable(policy: RegressionPolicy, batch: dict):
    _jit_parity(policy, batch, rtol=1e-4, atol=1e-5)


def test_regression_policy_checkpointing(policy: RegressionPolicy, batch: dict):
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

    action_orig = policy(batch)
    action_restored = restored_policy(batch)

    assert jnp.allclose(action_orig, action_restored)
