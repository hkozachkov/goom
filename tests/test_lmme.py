"""Tests for the matrix exponential sum implementation."""

import numpy as np
import jax.numpy as jnp

from goom.lmme import matrix_exp_sum
from goom.custom_exp import goom_exp


def test_complex_correctness():
    """Test correctness with complex numbers."""
    # Small test case for verification
    np.random.seed(123)
    n, d, m = 4, 3, 5
    
    # Create complex test matrices
    A = jnp.array(np.random.randn(n, d) + 1j * np.random.randn(n, d)) * 0.1
    B = jnp.array(np.random.randn(d, m) + 1j * np.random.randn(d, m)) * 0.1
    
    # Compute using our implementation
    C = matrix_exp_sum(A, B)
    
    # Manual computation for verification (element by element)
    C_manual = jnp.zeros((n, m), dtype=jnp.complex64)
    for i in range(n):
        for j in range(m):
            sum_val = 0.0 + 0.0j
            for k in range(d):
                sum_val += goom_exp(A[i, k] + B[k, j])
            C_manual = C_manual.at[i, j].set(sum_val)
    
    # Assert correctness
    assert jnp.allclose(C, C_manual, rtol=1e-5), (
        f"Max absolute difference: {jnp.max(jnp.abs(C - C_manual))}"
    )
    
    # Verify a sample element
    i, j = 0, 0
    manual_element = sum(goom_exp(A[i, k] + B[k, j]) for k in range(d))
    assert jnp.allclose(C[i, j], manual_element, rtol=1e-5)

