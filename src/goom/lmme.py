import jax.numpy as jnp
from jax import jax

from goom.goom import from_goom, to_goom


def log_matmul_exp(log_x1: jax.Array, log_x2: jax.Array, min_c: int = 0) -> jax.Array:
    """
    Broadcastable log(exp(log_x1) @ exp(log_x2)), implemented by calling
    JAX's existing implementation of matmul over scaled float tensors.
    Inputs:
        log_x1: log-tensor of shape [..., d1, d2].
        log_x2: log-tensor of shape [..., d2, d3].
        min_c: (optional) float, minimum log-scaling constant. Default: 0.
    Outputs:
        log_y: log-tensor of shape [..., d1, d3].
    """
    c1 = jnp.maximum(jnp.max(log_x1.real, axis=-1, keepdims=True), min_c)
    c2 = jnp.maximum(jnp.max(log_x2.real, axis=-2, keepdims=True), min_c)

    x1 = from_goom(log_x1 - c1)
    x2 = from_goom(log_x2 - c2)

    scaled_y = jnp.matmul(x1, x2)

    log_y = to_goom(scaled_y) + c1 + c2
    return log_y


def alternate_log_matmul_exp(log_x1: jax.Array, log_x2: jax.Array) -> jax.Array:
    """
    Broadcastable log(exp(log_x1) @ exp(log_x2)), implemented by composition
    of vmapped log-sum-exp-of-sum operations. Much slower, but more precise.
    Inputs:
        log_x1: log-tensor of shape [..., d1, d2].
        log_x2: log-tensor of shape [..., d2, d3].
    Outputs:
        log_y: log-tensor of shape [..., d1, d3].
    """
    # Get log-scaling constants:
    c1 = jnp.max(log_x1.real, axis=-1, keepdims=True)
    c2 = jnp.max(log_x2.real, axis=-2, keepdims=True)

    # Get log-scaled operands:
    log_s1 = log_x1 - c1
    log_s2 = log_x2 - c2

    # Broadcast preceding dims and flatten them into a single dim:
    d1, d2, d3 = (*log_s1.shape[-2:], log_s2.shape[-1])
    broadcast_szs = jnp.broadcast_shapes(log_s1.shape[:-2], log_s2.shape[:-2])

    log_s1_flat = jnp.broadcast_to(log_s1, broadcast_szs + (d1, d2)).reshape(-1, d1, d2)
    log_s2_flat = jnp.broadcast_to(log_s2, broadcast_szs + (d2, d3)).reshape(-1, d2, d3)

    # Define vmapped sum-exp-of-outer-sum operations:
    _vve = lambda row_vec, col_vec: from_goom(row_vec + col_vec).sum()
    _mve = jax.vmap(_vve, (0, None))
    _mme = jax.vmap(_mve, (None, 1), 1)
    _multi_mme = jax.vmap(_mme)

    # Compute, reshape, and return result:
    scaled_y = _multi_mme(log_s1_flat, log_s2_flat)
    log_y = to_goom(scaled_y) + c1 + c2
    return log_y.reshape(*broadcast_szs, d1, d3)
