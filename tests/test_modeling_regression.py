import pytest

import jax
import jax.numpy as jnp
from flax import nnx
import orbax.checkpoint as ocp

from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.configs.types import FeatureType

from easy_bc.modeling_regression import (
    EncoderStem,
    EncoderBlock,
    RGBEncoder,
    RegressionPolicy,
)
from easy_bc.configuration_regression import RegressionConfig


def _keep_keys(config: RegressionConfig) -> set[str]:
    return set(config.input_features) | set(config.output_features)


def _first_image_shape(config: RegressionConfig):
    return next(iter(config.image_features.values())).shape


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


@pytest.mark.parametrize(
    "input_dim,output_dim,H,W,B",
    [
        (3, 32, 64, 64, 2),
        (3, 16, 63, 65, 1),
        (8, 64, 128, 96, 4),
    ],
)
def test_encoder_stem_forward_pass(input_dim, output_dim, H, W, B, rngs):
    stem = EncoderStem(input_dim=input_dim, output_dim=output_dim, rngs=rngs)
    x = jax.random.normal(
        jax.random.PRNGKey(1), (B, H, W, input_dim), dtype=jnp.float32
    )
    y = stem(x)

    exp_H = (H + 1) // 2
    exp_W = (W + 1) // 2

    assert y.shape == (B, exp_H, exp_W, output_dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_encoder_stem_wrong_input_channels(rngs):
    stem = EncoderStem(input_dim=3, output_dim=32, rngs=rngs)
    x = jnp.zeros((1, 32, 32, 4), dtype=jnp.float32)
    with pytest.raises(Exception):
        stem(x)


@pytest.mark.parametrize(
    "input_dim,output_dim,stride,H,W,B",
    [
        (16, 16, 1, 32, 32, 2),
        (16, 32, 2, 32, 32, 2),
        (8, 24, 2, 63, 65, 1),
    ],
)
def test_encoder_block_forward_shape_and_values(
    input_dim, output_dim, stride, H, W, B, rngs
):
    block = EncoderBlock(
        input_dim=input_dim, output_dim=output_dim, stride=stride, rngs=rngs
    )
    x = jax.random.normal(
        jax.random.PRNGKey(1), (B, H, W, input_dim), dtype=jnp.float32
    )
    y = block(x)

    exp_H = (H + stride - 1) // stride
    exp_W = (W + stride - 1) // stride

    assert y.shape == (B, exp_H, exp_W, output_dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_encoder_block_rejects_mismatched_channels(rngs):
    block = EncoderBlock(input_dim=16, output_dim=16, stride=1, rngs=rngs)
    x = jnp.zeros((1, 32, 32, 15), dtype=jnp.float32)
    with pytest.raises(Exception):
        block(x)


@pytest.mark.parametrize("B,out_dim", [(2, 10), (1, 128), (4, 32)])
def test_rgb_encoder_forward_pass(regression_config, rngs, B, out_dim):
    regression_config.out_feature_dim = out_dim
    enc = RGBEncoder(regression_config, rngs=rngs)

    c, h, w = _first_image_shape(regression_config)
    x = jax.random.normal(jax.random.PRNGKey(1), (B, c, h, w), dtype=jnp.float32)
    y = enc(x)

    assert y.shape == (B, out_dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_rgb_encoder_jittable(regression_config, rngs):
    regression_config.out_feature_dim = 16
    enc = RGBEncoder(regression_config, rngs=rngs)

    c, h, w = _first_image_shape(regression_config)
    x = jax.random.normal(jax.random.PRNGKey(1), (2, c, h, w), dtype=jnp.float32)

    y_eager, y_jit = _jit_parity(enc, x, rtol=1e-4, atol=1e-5)
    assert y_eager.shape == y_jit.shape == (2, 16)


def test_rgb_encoder_wrong_input_channels_raises(regression_config, rngs):
    enc = RGBEncoder(regression_config, rngs=rngs)
    x_bad = jnp.zeros((1, 4, 64, 64), dtype=jnp.float32)
    with pytest.raises(Exception):
        enc(x_bad)


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
