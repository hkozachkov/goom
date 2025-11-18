from typing import Tuple

import jax
import jax.numpy as jnp


@jax.custom_vjp
def goom_exp(x: jax.Array) -> jax.Array:
    """Exponentiate ``x`` while providing a custom gradient."""
    return jnp.exp(x)


def _goom_exp_fwd(x: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Forward pass returns the primal value and residuals."""
    y = jnp.exp(x)
    residuals = x
    return y, residuals


def _goom_exp_bwd(residuals: jax.Array, g: jax.Array) -> Tuple[jax.Array]:
    """Backward pass that perturbs the residuals with a signed epsilon."""
    x = residuals
    eps = jnp.finfo(g.real.dtype).eps
    signed_eps = jnp.where(x.real < 0, -eps, eps)
    grad_x = g * (x + signed_eps)
    return (grad_x,)


goom_exp.defvjp(_goom_exp_fwd, _goom_exp_bwd)