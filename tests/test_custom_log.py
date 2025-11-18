import jax
import jax.numpy as jnp

from goom.custom_log import goom_log


def test_goom_log_matches_jnp_log_for_positive_values():
    values = jnp.linspace(0.1, 10.0, 25, dtype=jnp.float32)
    assert jnp.allclose(goom_log(values), jnp.log(values))


def test_goom_log_gradient_matches_spec():
    grad_fn = jax.grad(goom_log)
    for x in (0.25, 1.0, 5.0, 50.0):
        x_arr = jnp.array(x, dtype=jnp.float32)
        grad = grad_fn(x_arr)
        eps = jnp.finfo(x_arr.dtype).eps
        expected = 1.0 / (x_arr + eps)
        assert jnp.allclose(grad, expected)
