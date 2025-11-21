"""Tests for the matrix exponential sum implementation."""

# import numpy as np
import jax
import jax.numpy as jnp

from goom.lmme import log_matmul_exp, alternate_log_matmul_exp
from goom.goom import to_goom
from test_gooms import generate_random_gooms


# def test_lmme_correctness():
#     """Test correctness with complex numbers."""
#     # Small test case for verification
#     np.random.seed(123)
#     n, d, m = 4, 3, 5

#     # Create complex test matrices
#     A = jnp.array(np.random.randn(n, d) + 1j * np.random.randn(n, d)) * 0.1
#     B = jnp.array(np.random.randn(d, m) + 1j * np.random.randn(d, m)) * 0.1

#     # Compute using our implementation
#     C = alternate_log_matmul_exp(A, B)

#     # Manual computation for verification (element by element)
#     C_manual = jnp.zeros((n, m), dtype=jnp.complex64)
#     for i in range(n):
#         for j in range(m):
#             sum_val = 0.0 + 0.0j
#             for k in range(d):
#                 sum_val += goom_exp(A[i, k] + B[k, j])
#             C_manual = C_manual.at[i, j].set(sum_val)

#     # Assert correctness
#     assert jnp.allclose(C, C_manual, rtol=1e-5), (
#         f"Max absolute difference: {jnp.max(jnp.abs(C - C_manual))}"
#     )

#     # Verify a sample element
#     i, j = 0, 0
#     manual_element = sum(goom_exp(A[i, k] + B[k, j]) for k in range(d))
#     assert jnp.allclose(C[i, j], manual_element, rtol=1e-5)


def test_lmme_equivalence():
    """Test that log_matmul_exp and alternate_log_matmul_exp give similar results."""
    key = jax.random.PRNGKey(42)
    n, d, m = 8, 7, 6
    key, subkey1 = jax.random.split(key)
    log_x1 = generate_random_gooms(subkey1, (n, d))
    key, subkey2 = jax.random.split(key)
    log_x2 = generate_random_gooms(subkey2, (d, m))
    result_fast = log_matmul_exp(log_x1, log_x2)
    assert not jnp.isnan(
        result_fast.real
    ).any(), "NaNs found in real part of result_fast"
    assert not jnp.isnan(
        result_fast.imag
    ).any(), "NaNs found in imaginary part of result_fast"

    result_precise = alternate_log_matmul_exp(log_x1, log_x2)
    assert not jnp.isnan(
        result_precise.real
    ).any(), "NaNs found in real part of result_precise"
    assert not jnp.isnan(
        result_precise.imag
    ).any(), "NaNs found in imaginary part of result_precise"

    assert jnp.allclose(result_fast, result_precise, rtol=1e-5, atol=1e-5)
