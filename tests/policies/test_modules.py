import flax.nnx as nnx
import jax
import jax.numpy as jnp
import pytest

from easy_bc.policies.modules import (
    ConditionalResidual1DBlock,
    ConditionalUnet1D,
    Conv1DBlock,
    EncoderBlock,
    EncoderStem,
    RGBEncoder,
    SinusoidalPosEmb,
    SpatialSoftmax,
)


@pytest.fixture
def rngs():
    return nnx.Rngs(0)


def _jit_parity(
    module: nnx.Module, args: tuple, *, rtol: float = 1e-4, atol: float = 1e-5
):
    """
    module: nnx.Module
    args: tuple of inputs to the module
    """
    gdef, state = nnx.split(module)
    eager = nnx.merge(gdef, state)
    compiled = nnx.merge(gdef, state)

    y_eager = eager(*args)
    y_jit = jax.jit(compiled)(*args)

    assert jnp.allclose(y_eager, y_jit, rtol=rtol, atol=atol)
    assert y_eager.shape == y_jit.shape

    return y_eager, y_jit


@pytest.mark.parametrize(
    "B,dim",
    [
        (1, 8),
        (2, 32),
        (4, 64),
    ],
)
def test_sinusoidal_pos_emb_forward(B, dim):
    model = SinusoidalPosEmb(dim=dim)

    x = jax.random.uniform(
        jax.random.PRNGKey(0),
        (B,),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )

    y = model(x)

    assert y.shape == (B, dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_sinusoidal_pos_emb_jittable():
    dim = 64
    model = SinusoidalPosEmb(dim=dim)

    B = 4
    x = jax.random.uniform(
        jax.random.PRNGKey(1),
        (B,),
        minval=0.0,
        maxval=1.0,
        dtype=jnp.float32,
    )

    _jit_parity(model, (x,), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "B,H,feature_dim,cond_dim,down_dims,kernel_size,num_groups",
    [
        (2, 32, 8, 7, (32, 64), 3, 8),
        (1, 17, 16, 5, (32,), 1, 4),
        (4, 64, 32, 9, (64, 128), 5, 8),
    ],
)
def test_conditional_unet1d_forward(
    B, H, feature_dim, cond_dim, down_dims, kernel_size, num_groups, rngs
):
    model = ConditionalUnet1D(
        feature_dim=feature_dim,
        cond_dim=cond_dim,
        down_dims=down_dims,
        kernel_size=kernel_size,
        num_groups=num_groups,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (B, H, feature_dim), dtype=jnp.float32)
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim), dtype=jnp.float32)

    y = model(x, cond)

    assert y.shape == (B, H, feature_dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_conditional_unet1d_jittable(rngs):
    feature_dim, cond_dim = 8, 7
    model = ConditionalUnet1D(
        feature_dim=feature_dim,
        cond_dim=cond_dim,
        down_dims=(32, 64),
        kernel_size=3,
        num_groups=8,
        rngs=rngs,
    )

    B, H = 2, 32
    x = jax.random.normal(
        jax.random.PRNGKey(10), (B, H, feature_dim), dtype=jnp.float32
    )
    cond = jax.random.normal(jax.random.PRNGKey(11), (B, cond_dim), dtype=jnp.float32)

    _jit_parity(model, (x, cond), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "B,H,input_dim,output_dim,kernel_size",
    [
        (2, 32, 8, 16, 3),
        (1, 17, 16, 16, 1),
        (4, 64, 32, 64, 5),
    ],
)
def test_conv1d_block_forward(B, H, input_dim, output_dim, kernel_size, rngs):
    block = Conv1DBlock(
        input_dim=input_dim,
        output_dim=output_dim,
        kernel_size=kernel_size,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (B, H, input_dim), dtype=jnp.float32)
    y = block(x)

    assert y.shape == (B, H, output_dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_conv1d_block_jittable(rngs):
    block = Conv1DBlock(
        input_dim=8,
        output_dim=16,
        kernel_size=3,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (2, 32, 8), dtype=jnp.float32)

    _jit_parity(block, (x,), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "B,H,input_dim,output_dim,cond_dim",
    [
        (2, 32, 8, 16, 12),
        (1, 17, 16, 16, 8),  # identity residual
        (4, 64, 32, 64, 32),
    ],
)
def test_conditional_residual_1d_block_forward(
    B, H, input_dim, output_dim, cond_dim, rngs
):
    block = ConditionalResidual1DBlock(
        input_dim=input_dim,
        output_dim=output_dim,
        cond_dim=cond_dim,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (B, H, input_dim), dtype=jnp.float32)
    cond = jax.random.normal(jax.random.PRNGKey(1), (B, cond_dim), dtype=jnp.float32)

    y = block(x, cond)

    assert y.shape == (B, H, output_dim)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()


def test_conditional_residual_1d_block_jittable(rngs):
    block = ConditionalResidual1DBlock(
        input_dim=8,
        output_dim=16,
        cond_dim=10,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (2, 32, 8), dtype=jnp.float32)
    cond = jax.random.normal(jax.random.PRNGKey(1), (2, 10), dtype=jnp.float32)

    _jit_parity(block, (x, cond), rtol=1e-4, atol=1e-5)


def test_conditional_residual_1d_block_wrong_cond_shape_raises(rngs):
    block = ConditionalResidual1DBlock(
        input_dim=8,
        output_dim=16,
        cond_dim=10,
        rngs=rngs,
    )

    x = jnp.zeros((2, 32, 8), dtype=jnp.float32)
    bad_cond = jnp.zeros((2, 11), dtype=jnp.float32)

    with pytest.raises(Exception):
        block(x, bad_cond)


@pytest.mark.parametrize(
    "input_dim,output_dim,H,W,B",
    [
        (3, 32, 64, 64, 2),
        (3, 16, 63, 65, 1),
        (8, 64, 128, 96, 4),
    ],
)
def test_encoder_stem_forward(input_dim, output_dim, H, W, B, rngs):
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
def test_encoder_block_forward(input_dim, output_dim, stride, H, W, B, rngs):
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
    "B,H,W,C,num_keypoints",
    [
        (2, 32, 32, 3, 8),
        (1, 63, 65, 3, 16),
        (4, 128, 96, 3, 8),
    ],
)
def test_spatial_softmax_forward(B, H, W, C, num_keypoints, rngs):
    model = SpatialSoftmax(
        input_shape=(H, W, C),
        num_keypoints=num_keypoints,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(0), (B, H, W, C), dtype=jnp.float32)
    y = model(x)

    assert y.shape == (B, num_keypoints * 2)
    assert y.dtype == x.dtype
    assert jnp.isfinite(y).all()

    assert jnp.all(y >= -1.0001)
    assert jnp.all(y <= 1.0001)


def test_spatial_softmax_jittable(rngs):
    B, H, W, C, num_keypoints = 2, 32, 32, 3, 8
    model = SpatialSoftmax(
        input_shape=(H, W, C),
        num_keypoints=num_keypoints,
        rngs=rngs,
    )

    x = jax.random.normal(jax.random.PRNGKey(1), (B, H, W, C), dtype=jnp.float32)

    _jit_parity(model, (x,), rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "B,H,W,C,out_dim", [(2, 64, 64, 3, 10), (1, 63, 65, 3, 16), (4, 128, 96, 3, 8)]
)
def test_rgb_encoder_forward(B, H, W, C, out_dim, rngs):
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

    _jit_parity(enc, (x,), rtol=1e-4, atol=1e-5)


def test_rgb_encoder_wrong_input_channels_raises(rngs):
    out_feature_dim = 16
    image_shape = (3, 64, 64)

    enc = RGBEncoder(image_shape, out_feature_dim, rngs=rngs)
    x_bad = jnp.zeros((1, 4, 64, 64), dtype=jnp.float32)
    with pytest.raises(Exception):
        enc(x_bad)
