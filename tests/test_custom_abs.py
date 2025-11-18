import jax
import jax.numpy as jnp

from goom.custom_abs import goom_abs


def test_goom_abs_matches_jnp_abs():
    values = jnp.linspace(-5.0, 5.0, 21, dtype=jnp.float32)
    assert jnp.allclose(goom_abs(values), jnp.abs(values))


def test_goom_abs_gradient_matches_spec():
    grad_fn = jax.grad(goom_abs)
    for x in (-2.0, -0.5, 0.0, 0.5, 3.0):
        x_arr = jnp.array(x, dtype=jnp.float32)
        grad = grad_fn(x_arr)
        expected = 1.0 if x == 0 else jnp.sign(x_arr)
        assert jnp.allclose(grad, expected)