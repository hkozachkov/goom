"""Custom absolute value primitive with a defined derivative at zero."""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import jit


@jax.custom_vjp
@jax.jit
def goom_abs(x: jax.Array) -> jax.Array:
    """Absolute value with a custom derivative that equals 1 at zero."""
    return jnp.abs(x)


def _goom_abs_fwd(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Forward pass that keeps the original input as residuals."""
    y = jnp.abs(x)
    residuals = x
    return y, residuals


def _goom_abs_bwd(residuals: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    """Backward pass that defines grad(|x|) = sign(x), but 1 when x == 0."""
    x = residuals
    grad_x = g * jnp.where(x == 0, jnp.ones_like(x), jnp.sign(x))
    return (grad_x,)


goom_abs.defvjp(_goom_abs_fwd, _goom_abs_bwd)


