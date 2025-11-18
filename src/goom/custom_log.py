from typing import Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
import math

from goom.config import config


@jax.custom_vjp
def goom_log(x: jax.Array) -> jax.Array:
    """Logarithm of ``x`` while providing a custom gradient."""
    return jnp.log(x)


def _goom_log_fwd(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Forward pass returns the primal value and residuals."""
    log_inp = jnp.log(x)
    snn = jnp.finfo(x.real.dtype).smallest_normal
    finite_floor = math.log(snn) * 2  # exps to zero in float_dtype
    keep_finite_idx = (log_inp < finite_floor) & config.keep_logs_finite
    out = jnp.where(keep_finite_idx, finite_floor, log_inp)
    residuals = x
    return out, residuals


def _goom_log_bwd(residuals: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    """Backward pass that perturbs the residuals with a signed epsilon."""
    x = residuals
    eps = jnp.finfo(g.real.dtype).eps
    grad_x = g / (x + eps)
    return (grad_x,)


goom_log.defvjp(_goom_log_fwd, _goom_log_bwd)
