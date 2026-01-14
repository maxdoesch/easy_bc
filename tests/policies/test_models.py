import pytest

import jax
import jax.numpy as jnp
import flax.nnx as nnx


from easy_bc.policies.models import EncoderStem, EncoderBlock, RGBEncoder


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


def _jit_parity(module, x, *, rtol=1e-4, atol=1e-5):
    gdef, state = nnx.split(module)
    eager = nnx.merge(gdef, state)
    compiled = nnx.merge(gdef, state)
    y_eager = eager(x)
    y_jit = jax.jit(compiled)(x)
    assert y_eager.shape == y_jit.shape
    assert jnp.allclose(y_eager, y_jit, rtol=rtol, atol=atol)
    return y_eager, y_jit


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


@pytest.mark.parametrize(
    "B,H,W,C,out_dim", [(2, 64, 64, 3, 10), (1, 63, 65, 3, 16), (4, 128, 96, 3, 8)]
)
def test_rgb_encoder_forward_pass(B, H, W, C, out_dim, rngs):
    img_shape = (C, H, W)
    enc = RGBEncoder(img_shape, out_dim, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(1), (B, C, H, W), dtype=jnp.float32)
    y = enc(x)

    assert y.shape == (B, out_dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_rgb_encoder_jittable(rngs):
    out_feature_dim = 16
    image_shape = (3, 64, 64)
    enc = RGBEncoder(image_shape, out_feature_dim, rngs=rngs)

    x = jax.random.normal(jax.random.PRNGKey(1), (2, *image_shape), dtype=jnp.float32)

    y_eager, y_jit = _jit_parity(enc, x, rtol=1e-4, atol=1e-5)
    assert y_eager.shape == y_jit.shape == (2, 16)


def test_rgb_encoder_wrong_input_channels_raises(rngs):
    out_feature_dim = 16
    image_shape = (3, 64, 64)

    enc = RGBEncoder(image_shape, out_feature_dim, rngs=rngs)
    x_bad = jnp.zeros((1, 4, 64, 64), dtype=jnp.float32)
    with pytest.raises(Exception):
        enc(x_bad)
