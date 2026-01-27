import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from flax import nnx
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features

from easy_bc.policies.regression.configuration_regression import RegressionConfig
from easy_bc.policies.regression.modeling_regression import RegressionPolicy


def _keep_keys(config: RegressionConfig) -> set[str]:
    return set(config.input_features) | set(config.output_features)


def _action_dim(config: RegressionConfig) -> int:
    return next(iter(config.output_features.values())).shape[0]


def _as_batched_jnp(x):
    return jnp.expand_dims(jnp.asarray(x), axis=0)


def _jit_parity_method(obj, method_name: str, batch, *, rtol=1e-4, atol=1e-5):
    """
    Check eager vs jitted parity for a specific method on an NNX module/policy.
    """
    gdef, state = nnx.split(obj)
    eager = nnx.merge(gdef, state)
    compiled = nnx.merge(gdef, state)

    eager_fn = getattr(eager, method_name)
    compiled_fn = getattr(compiled, method_name)

    y_eager = eager_fn(batch)
    y_jit = jax.jit(compiled_fn)(batch)

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


def test_regression_policy_sample_action(policy: RegressionPolicy, batch: dict):
    action = policy.sample_action(batch)
    assert action.shape == (1, _action_dim(policy.config))


def test_regression_policy_compute_loss(policy: RegressionPolicy, batch: dict):
    loss = policy.compute_loss(batch)
    assert loss.shape == (1, _action_dim(policy.config))
    assert jnp.isfinite(loss).all()


def test_regression_policy_sample_action_jittable(
    policy: RegressionPolicy, batch: dict
):
    _jit_parity_method(policy, "sample_action", batch, rtol=1e-4, atol=1e-5)


def test_regression_policy_compute_loss_jittable(policy: RegressionPolicy, batch: dict):
    _jit_parity_method(policy, "compute_loss", batch, rtol=1e-4, atol=1e-5)


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

    action_orig = policy.sample_action(batch)
    action_restored = restored_policy.sample_action(batch)

    assert jnp.allclose(action_orig, action_restored)
