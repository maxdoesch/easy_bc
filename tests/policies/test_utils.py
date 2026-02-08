import jax
import jax.numpy as jnp
import numpy as np
import pytest

from easy_bc.policies.utils import crop_image, resize_with_pad


def _coord_image_uint8(C: int, H: int, W: int) -> jnp.ndarray:
    """Image where pixel value encodes (y, x) so crops are easy to verify."""
    yy = jnp.arange(H, dtype=jnp.uint16)[:, None]
    xx = jnp.arange(W, dtype=jnp.uint16)[None, :]
    base = (yy * 10 + xx).astype(jnp.uint8)  # safe for small H,W
    return jnp.broadcast_to(base[None, ...], (C, H, W))


def _coord_image_float(C: int, H: int, W: int) -> jnp.ndarray:
    yy = jnp.arange(H, dtype=jnp.float32)[:, None]
    xx = jnp.arange(W, dtype=jnp.float32)[None, :]
    base = (yy * 10.0 + xx) / (10.0 * max(H - 1, 1) + max(W - 1, 1))  # ~[0,1]
    return jnp.broadcast_to(base[None, ...], (C, H, W))


# -------------------------
# crop_image tests
# -------------------------


def test_crop_image_center_shape_no_batch():
    x = _coord_image_uint8(C=1, H=7, W=9)  # (C,H,W)
    out = crop_image(x, shape=(3, 4), random=False)
    assert out.shape == (1, 3, 4)


def test_crop_image_center_composition_no_batch():
    C, H, W = 1, 7, 9
    ch, cw = 3, 4
    x = _coord_image_uint8(C=C, H=H, W=W)

    max_y, max_x = H - ch, W - cw
    y0, x0 = max_y // 2, max_x // 2
    expected = x[:, y0 : y0 + ch, x0 : x0 + cw]

    out = crop_image(x, shape=(ch, cw), random=False)
    np.testing.assert_array_equal(np.array(out), np.array(expected))


def test_crop_image_center_batch_shape_and_composition():
    C, H, W = 2, 8, 8
    ch, cw = 5, 3
    x0 = _coord_image_uint8(C=C, H=H, W=W)
    x1 = _coord_image_uint8(C=C, H=H, W=W) + 1
    xb = jnp.stack([x0, x1], axis=0)  # (B,C,H,W)

    out = crop_image(xb, shape=(ch, cw), random=False)
    assert out.shape == (2, C, ch, cw)

    max_y, max_x = H - ch, W - cw
    y0, x0s = max_y // 2, max_x // 2
    expected0 = xb[0, :, y0 : y0 + ch, x0s : x0s + cw]
    expected1 = xb[1, :, y0 : y0 + ch, x0s : x0s + cw]
    np.testing.assert_array_equal(np.array(out[0]), np.array(expected0))
    np.testing.assert_array_equal(np.array(out[1]), np.array(expected1))


def test_crop_image_random_deterministic_given_rng():
    C, H, W = 1, 10, 10
    ch, cw = 4, 6
    xb = jnp.stack(
        [_coord_image_uint8(C=C, H=H, W=W), _coord_image_uint8(C=C, H=H, W=W) + 3],
        axis=0,
    )  # (B,C,H,W)

    rng = jax.random.PRNGKey(0)
    out1 = crop_image(xb, shape=(ch, cw), random=True, rng=rng)
    out2 = crop_image(xb, shape=(ch, cw), random=True, rng=rng)
    np.testing.assert_array_equal(np.array(out1), np.array(out2))


def test_crop_image_random_requires_rng():
    x = _coord_image_uint8(C=1, H=6, W=6)
    with pytest.raises(ValueError, match="rng required"):
        _ = crop_image(x, shape=(3, 3), random=True, rng=None)


def test_crop_image_invalid_rank_raises():
    with pytest.raises(ValueError, match="Invalid shape"):
        _ = crop_image(jnp.zeros((1, 2, 3, 4, 5)), shape=(2, 2))


# -------------------------
# resize_with_pad tests
# -------------------------


def test_resize_with_pad_shape_no_batch_and_batch():
    x = _coord_image_uint8(C=3, H=7, W=5)  # (C,H,W)
    out = resize_with_pad(x, (10, 10), method=jax.image.ResizeMethod.NEAREST)
    assert out.shape == (3, 10, 10)

    xb = jnp.stack([x, x], axis=0)  # (B,C,H,W)
    outb = resize_with_pad(xb, (10, 10), method=jax.image.ResizeMethod.NEAREST)
    assert outb.shape == (2, 3, 10, 10)


def test_resize_with_pad_uint8_padding_is_zero_and_centered():
    # pick a shape that forces padding on width (narrow resized image in a wide canvas)
    # original: W > H -> resizing to square keeps aspect, leaving pad on height or width depending
    x = _coord_image_uint8(C=1, H=4, W=8)  # wide
    target_h, target_w = 10, 10

    out = resize_with_pad(
        x, (target_h, target_w), method=jax.image.ResizeMethod.NEAREST
    )
    assert out.dtype == jnp.uint8
    assert out.shape == (1, target_h, target_w)

    out_np = np.array(out[0])

    # padding should be exactly zeros (uint8)
    # detect the "content box" as nonzero pixels (coord image has nonzero except at (0,0))
    # be robust: use rows/cols where any pixel is nonzero
    rows = np.where(out_np.any(axis=1))[0]
    cols = np.where(out_np.any(axis=0))[0]
    assert rows.size > 0 and cols.size > 0

    r0, r1 = rows[0], rows[-1]
    c0, c1 = cols[0], cols[-1]

    # outside content box should be all zeros
    top = out_np[:r0, :]
    bot = out_np[r1 + 1 :, :]
    left = out_np[:, :c0]
    right = out_np[:, c1 + 1 :]
    assert np.all(top == 0)
    assert np.all(bot == 0)
    assert np.all(left == 0)
    assert np.all(right == 0)

    # content box should be centered (pads differ by at most 1)
    pad_top = r0
    pad_bot = target_h - (r1 + 1)
    pad_left = c0
    pad_right = target_w - (c1 + 1)
    assert abs(pad_top - pad_bot) <= 1
    assert abs(pad_left - pad_right) <= 1


def test_resize_with_pad_float32_clips_to_unit_interval():
    x = _coord_image_float(C=1, H=5, W=7) * 2.0 - 0.5  # deliberately outside [0,1]
    out = resize_with_pad(x, (8, 8), method=jax.image.ResizeMethod.LINEAR)
    assert out.dtype == jnp.float32
    assert out.shape == (1, 8, 8)
    out_np = np.array(out)
    assert out_np.min() >= 0.0 - 1e-6
    assert out_np.max() <= 1.0 + 1e-6


def test_resize_with_pad_unsupported_dtype_raises():
    x = jnp.zeros((1, 3, 4), dtype=jnp.float16)
    with pytest.raises(ValueError, match="Unsupported image dtype"):
        _ = resize_with_pad(x, (8, 8), method=jax.image.ResizeMethod.NEAREST)
