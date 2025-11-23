import jax
import jax.numpy as jnp
from jax import lax

import goom.goom as goom
import goom.lmme as lmme
import goom.operations as oprs


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


def jax_estimate_lle_parallel(jac_vals, key, dt=1.0):
    
    """
    Estimate the largest Lyapunov exponent from Jacobians, in parallel
    
    jacobians: array of shape (T, D, D)
               J[t] is the Jacobian at time step t
    key:       jax.random.PRNGKey
    dt:        time step between Jacobians (default 1.0)
    
    """
    # get log_jac_vals
    T = jac_vals.shape[-3]
    log_jac_vals = goom.to_goom(jac_vals)

    # initialize random unit vector u[0]
    key, u0_key = jax.random.split(key)
    u0 = rand_like_normalized(jac_vals, axis=-1, key=u0_key)

    # apply Jacobians from last to first: M[T] = J[T] @ ... @ J[0] in goom space
    log_jac_product = lax.associative_scan(
        lmme.log_matmul_exp, jnp.flip(log_jac_vals, axis=0), axis=0
    )[-1]

    # M[T] @ u[0] in goom space
    log_end_state = lmme.log_matmul_exp(u0, log_jac_product)

    return oprs.log_sum_exp(log_end_state * 2, axis=-1).real / (2 * T * dt)


def jax_estimate_lle_sequential(jacobians, key, dt=1.0, eps=1e-12):
    """
    jacobians: array of shape (T, D, D)
               J[t] is the Jacobian at time step t
    key:       jax.random.PRNGKey
    dt:        time step between Jacobians (default 1.0)
    """
    T, D, _ = jacobians.shape

    # random initial unit vector in R^D
    v0 = jax.random.normal(key, shape=(D,))
    v0 = normalize(v0, axis=0, eps=eps)

    def step(v, J):
        # propagate tangent vector
        w = J @ v
        norm = jnp.linalg.norm(w) + eps
        v_next = w / norm
        log_norm = jnp.log(norm)
        return v_next, log_norm

    # run sequentially over time with lax.scan (JAX-friendly loop)
    vT, log_norms = lax.scan(step, v0, jacobians)

    # average growth rate -> largest Lyapunov exponent
    # divide by dt if your Jacobians correspond to time step dt
    lambda_max = jnp.mean(log_norms) / dt
    return lambda_max
