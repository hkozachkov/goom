import jax
import jax.numpy as jnp

from goom.goom import to_goom, from_goom, generate_random_gooms


def test_generate_random_gooms_no_nans():
    """Test that generate_random_gooms doesn't produce NaNs in real or imag parts."""
    key = jax.random.PRNGKey(0)
    shape = (20, 10)
    gooms = generate_random_gooms(key, shape)
    assert not jnp.isnan(gooms.real).any(), "NaNs found in real part of gooms"
    assert not jnp.isnan(gooms.imag).any(), "NaNs found in imaginary part of gooms"


def test_goom_inversion():
    """Test that from_goom(to_goom(x)) is close to x."""
    key = jax.random.PRNGKey(123)
    shape = (100,)

    # Generate random floats
    minval = jnp.finfo(jnp.float32).min
    maxval = jnp.finfo(jnp.float32).max
    original_floats = jax.random.uniform(
        key, shape, dtype=jnp.float32, minval=minval, maxval=maxval
    )

    # Apply to_goom and then from_goom
    gooms = to_goom(original_floats)
    reconstructed_floats = from_goom(gooms)

    # Check if the reconstructed floats are close to the original ones
    assert jnp.allclose(original_floats, reconstructed_floats, rtol=1e-5, atol=1e-5)
