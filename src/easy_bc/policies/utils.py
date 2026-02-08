from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp


def resize_with_pad(
    images: jnp.ndarray,
    shape: Tuple[int, int],
    method: jax.image.ResizeMethod = jax.image.ResizeMethod.LINEAR,
) -> jnp.ndarray:
    has_batch_dim = images.ndim == 4
    if not has_batch_dim:
        images = images[None]  # type: ignore
    B, C, H, W = images.shape

    height, width = shape
    ratio = max(W / width, H / height)
    resized_height = int(H / ratio)
    resized_width = int(W / ratio)
    resized_images = jax.image.resize(
        images,
        (B, C, resized_height, resized_width),
        method=method,
    )
    if images.dtype == jnp.uint8:
        # round from float back to uint8
        resized_images = jnp.round(resized_images).clip(0, 255).astype(jnp.uint8)
    elif images.dtype == jnp.float32:
        resized_images = resized_images.clip(0.0, 1.0)
    else:
        raise ValueError(f"Unsupported image dtype: {images.dtype}")

    pad_h0, remainder_h = divmod(height - resized_height, 2)
    pad_h1 = pad_h0 + remainder_h
    pad_w0, remainder_w = divmod(width - resized_width, 2)
    pad_w1 = pad_w0 + remainder_w
    padded_images = jnp.pad(
        resized_images,
        ((0, 0), (0, 0), (pad_h0, pad_h1), (pad_w0, pad_w1)),
        constant_values=0 if images.dtype == jnp.uint8 else 0.0,
    )

    if not has_batch_dim:
        padded_images = padded_images[0]
    return padded_images


def crop_image(
    x: jnp.ndarray,
    *,
    shape: Tuple[int, int],
    random: bool = False,
    rng: Optional[chex.PRNGKey] = None,
) -> jnp.ndarray:
    """Center or random crop. x: (C,H,W) or (B,C,H,W)."""
    ch, cw = shape
    x = jnp.asarray(x)

    if x.ndim == 3:
        x = x[None]
        squeeze = True
    elif x.ndim == 4:
        squeeze = False
    else:
        raise ValueError(f"Invalid shape {x.shape}")

    B, C, H, W = x.shape
    max_y, max_x = H - ch, W - cw

    if random:
        if rng is None:
            raise ValueError("rng required for random crop")
        ry, rx = jax.random.split(rng)
        y0 = jax.random.randint(ry, (B,), 0, max_y + 1)
        x0 = jax.random.randint(rx, (B,), 0, max_x + 1)
    else:
        y0 = jnp.full((B,), max_y // 2)
        x0 = jnp.full((B,), max_x // 2)

    def crop(img, y, x):
        return jax.lax.dynamic_slice(img, (0, y, x), (C, ch, cw))

    out = jax.vmap(crop)(x, y0, x0)
    return out[0] if squeeze else out
