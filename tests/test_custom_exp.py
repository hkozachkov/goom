import jax
import jax.numpy as jnp

from goom.custom_exp import goom_exp


def test_stable_exp_matches_jnp_exp():
    values = jnp.linspace(-5.0, 5.0, 11)
    assert jnp.allclose(goom_exp(values), jnp.exp(values))


def test_custom_gradient_matches_spec():
    grad_fn = jax.grad(goom_exp)
    for x in (-3.0, -0.1, 0.0, 0.5, 5.0):
        x_arr = jnp.array(x, dtype=jnp.float32)
        grad = grad_fn(x_arr)
        eps = jnp.finfo(x_arr.dtype).eps
        expected = x_arr + (eps if x >= 0 else -eps)
        assert jnp.allclose(grad, expected)
