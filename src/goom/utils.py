import jax
import jax.numpy as jnp


def randn_like(x, key, dtype=None):
    return jax.random.normal(key, shape=x.shape, dtype=(dtype or x.dtype))


def normalize(x, axis=-1, eps=1e-12):
    norm = jnp.linalg.norm(x, axis=axis, keepdims=True)
    return x / (norm + eps)


def rand_like_normalized(jac_vals, axis, key):
    return normalize(
        randn_like(jac_vals[..., 0, :1, :], key),
        axis=axis,
    )
