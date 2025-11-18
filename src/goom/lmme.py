"""
Production-Ready GPU-Optimized Matrix Exponential Sum in JAX

This module provides an efficient implementation of the operation:
C[i,j] = sum_k(exp(A[i,k] + B[k,j]))

Optimized for NVIDIA GPUs with automatic tuning and memory management.
Default dtype is complex64 for complex-valued computations.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap, devices
from functools import partial
import numpy as np
from typing import Optional, Tuple
from goom.custom_exp import goom_exp


class MatrixExpSum:
    """
    GPU-optimized matrix exponential sum operation with automatic tuning.
    """
    
    def __init__(self, dtype=jnp.complex64):
        """
        Initialize the operator.
        
        Args:
            dtype: Data type for computation (complex64, complex128, float32, float64)
                   Default is complex64 for complex-valued computations.
        """
        self.dtype = dtype
        self.gpu_info = self._get_gpu_info()
        
    def _get_gpu_info(self):
        """Get GPU information for optimization decisions."""
        try:
            gpu_devices = devices('gpu')
            if gpu_devices:
                # Rough heuristics for common GPUs
                return {
                    'available': True,
                    'count': len(gpu_devices),
                    'suggested_tile_size': 64,  # Good default for most modern GPUs
                    'max_threads_per_block': 1024,
                    'shared_memory_kb': 48  # Conservative estimate
                }
        except:
            pass
        
        return {
            'available': False,
            'count': 0,
            'suggested_tile_size': 32,
            'max_threads_per_block': 512,
            'shared_memory_kb': 16
        }
    
    def _auto_tune_parameters(self, n: int, d: int, m: int) -> Tuple[int, int, int]:
        """
        Automatically determine optimal tile and chunk sizes.
        
        Args:
            n, d, m: Matrix dimensions
            
        Returns:
            Tuple of (tile_n, tile_m, chunk_d)
        """
        # Memory constraint (in elements)
        # Complex64 uses 8 bytes per element (4 bytes real + 4 bytes imaginary)
        # Complex128 uses 16 bytes per element
        if self.dtype in [jnp.complex64, jnp.complex128]:
            # Reduce max elements for complex types due to doubled memory usage
            max_elements_per_tile = 512 * 1024  # More conservative for complex
        else:
            max_elements_per_tile = 1024 * 1024  # For float32/float64
        
        # Estimate optimal tile sizes
        base_tile = self.gpu_info['suggested_tile_size']
        
        # Adjust based on matrix dimensions
        tile_n = min(base_tile, n)
        tile_m = min(base_tile, m)
        
        # Determine chunk size for reduction dimension
        # We want: tile_n * tile_m * chunk_d <= max_elements_per_tile
        max_chunk_d = max_elements_per_tile // (tile_n * tile_m)
        chunk_d = min(max_chunk_d, d, 256)  # Cap at 256 for stability
        
        # Adjust for very large reduction dimensions
        if d > 1024:
            chunk_d = min(chunk_d, 128)
        
        return tile_n, tile_m, chunk_d
    
    @partial(jit, static_argnums=(0, 4, 5, 6))
    def compute_tiled(self, A, B, C, tile_n, tile_m, chunk_d):
        """
        JIT-compiled tiled computation kernel.
        
        This is separated to allow JIT compilation with static tile sizes.
        """
        n, d = A.shape
        _, m = B.shape
        
        # Process tiles
        for i in range(0, n, tile_n):
            i_end = min(i + tile_n, n)
            
            for j in range(0, m, tile_m):
                j_end = min(j + tile_m, m)
                
                # Initialize tile accumulator
                tile_acc = jnp.zeros((i_end - i, j_end - j), dtype=self.dtype)
                
                # Chunk the reduction
                for k in range(0, d, chunk_d):
                    k_end = min(k + chunk_d, d)
                    
                    # Extract blocks
                    A_block = A[i:i_end, k:k_end]
                    B_block = B[k:k_end, j:j_end]
                    
                    # Compute contribution
                    A_exp = A_block[:, jnp.newaxis, :]
                    B_exp = B_block.T[jnp.newaxis, :, :]
                    
                    # Compute exp and sum
                    exp_sum = jnp.sum(goom_exp(A_exp + B_exp), axis=-1)
                    tile_acc = tile_acc + exp_sum
                
                # Write tile to output
                C = C.at[i:i_end, j:j_end].set(tile_acc)
        
        return C
    
    @partial(jit, static_argnums=(0,))
    def compute_vmap(self, A, B):
        """
        vmap-based implementation for moderate sizes.
        """
        def compute_row(a_row, B):
            exp_terms = goom_exp(a_row[:, jnp.newaxis] + B)
            return jnp.sum(exp_terms, axis=0)
        
        return vmap(compute_row, in_axes=(0, None))(A, B)
    
    def __call__(self, A: jnp.ndarray, B: jnp.ndarray, 
                 method: str = 'auto') -> jnp.ndarray:
        """
        Compute C[i,j] = sum_k(exp(A[i,k] + B[k,j])).
        
        Args:
            A: Matrix of shape (n, d), can be complex or real
            B: Matrix of shape (d, m), can be complex or real
            method: 'auto', 'tiled', or 'vmap'
            
        Returns:
            Matrix C of shape (n, m)
        """
        n, d = A.shape
        d_check, m = B.shape
        
        if d != d_check:
            raise ValueError(f"Inner dimensions must match: {d} != {d_check}")
        
        # Convert to specified dtype
        A = A.astype(self.dtype)
        B = B.astype(self.dtype)
        
        # Choose method
        if method == 'auto':
            # Use vmap for smaller matrices, tiled for larger
            total_elements = n * m * d
            method = 'vmap' if total_elements < 10_000_000 else 'tiled'
        
        if method == 'vmap':
            return self.compute_vmap(A, B)
        elif method == 'tiled':
            # Auto-tune parameters
            tile_n, tile_m, chunk_d = self._auto_tune_parameters(n, d, m)
            
            # Initialize output
            C = jnp.zeros((n, m), dtype=self.dtype)
            
            # Compute
            return self.compute_tiled(A, B, C, tile_n, tile_m, chunk_d)
        else:
            raise ValueError(f"Unknown method: {method}")


# ============================================================================
# Convenience Functions
# ============================================================================

# Global operator instances
_operators = {}

def matrix_exp_sum(A: jnp.ndarray, B: jnp.ndarray, 
                   dtype: Optional[str] = None) -> jnp.ndarray:
    """
    Compute C[i,j] = sum_k(exp(A[i,k] + B[k,j])) with automatic optimization.
    
    Args:
        A: Matrix of shape (n, d), can be complex or real
        B: Matrix of shape (d, m), can be complex or real
        dtype: 'complex64' (default), 'complex128', 'float32', or 'float64'
        
    Returns:
        Matrix C of shape (n, m)
    """
    global _operators
    
    # Default to complex64
    if dtype is None:
        dtype = 'complex64'
    
    # Map string dtype to JAX dtype
    dtype_map = {
        'complex64': jnp.complex64,
        'complex128': jnp.complex128,
        'float32': jnp.float32,
        'float64': jnp.float64,
    }
    
    if dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype}. "
                        f"Choose from {list(dtype_map.keys())}")
    
    jax_dtype = dtype_map[dtype]
    
    # Get or create operator
    if dtype not in _operators:
        _operators[dtype] = MatrixExpSum(jax_dtype)
    
    return _operators[dtype](A, B)


# ============================================================================
# Example Usage and Testing
# ============================================================================

def example_usage():
    """Demonstrate usage of the optimized operator with complex numbers."""
    print("GPU-Optimized Matrix Exponential Sum (Complex-Valued)")
    print("="*60)
    
    # Create test matrices with complex values
    np.random.seed(42)
    n, d, m = 256, 128, 256
    
    # Create complex matrices
    A_real = np.random.randn(n, d) * 0.01
    A_imag = np.random.randn(n, d) * 0.01
    A = A_real + 1j * A_imag
    
    B_real = np.random.randn(d, m) * 0.01
    B_imag = np.random.randn(d, m) * 0.01
    B = B_real + 1j * B_imag
    
    # Convert to JAX arrays
    A_jax = jnp.array(A)
    B_jax = jnp.array(B)
    
    print(f"\nInput shapes: A={A_jax.shape}, B={B_jax.shape}")
    print(f"Input dtypes: A={A_jax.dtype}, B={B_jax.dtype}")
    
    # Method 1: Simple function call (defaults to complex64)
    print("\n1. Using convenience function (default complex64):")
    C = matrix_exp_sum(A_jax, B_jax)
    print(f"   Output shape: {C.shape}")
    print(f"   Output dtype: {C.dtype}")
    print(f"   Sample values (real part): {C[0, :3].real}")
    print(f"   Sample values (imag part): {C[0, :3].imag}")
    
    # Method 2: Using the class directly for more control
    print("\n2. Using MatrixExpSum class:")
    operator = MatrixExpSum(dtype=jnp.complex64)
    C_tiled = operator(A_jax, B_jax, method='tiled')
    C_vmap = operator(A_jax, B_jax, method='vmap')
    print(f"   Tiled vs vmap match: {jnp.allclose(C_tiled, C_vmap, rtol=1e-5)}")
    
    # Method 3: Using higher precision complex128
    print("\n3. Using complex128 precision:")
    C_c128 = matrix_exp_sum(A_jax, B_jax, dtype='complex128')
    print(f"   Output dtype: {C_c128.dtype}")
    rel_error = jnp.mean(jnp.abs(C - C_c128) / jnp.abs(C_c128))
    print(f"   Relative error vs complex64: {rel_error:.6f}")
    
    # Method 4: Real-valued matrices with complex output
    print("\n4. Real inputs with complex64 output:")
    A_real_jax = jnp.array(A_real)
    B_real_jax = jnp.array(B_real)
    C_from_real = matrix_exp_sum(A_real_jax, B_real_jax)  # Still uses complex64
    print(f"   Input dtypes: A={A_real_jax.dtype}, B={B_real_jax.dtype}")
    print(f"   Output dtype: {C_from_real.dtype}")
    print(f"   Imaginary part is zero: {jnp.allclose(C_from_real.imag, 0)}")
    
    # Performance comparison
    print("\n5. Performance comparison:")
    import time
    
    methods = [
        ("Auto (complex64)", lambda: matrix_exp_sum(A_jax, B_jax)),
        ("Auto (complex128)", lambda: matrix_exp_sum(A_jax, B_jax, dtype='complex128')),
        ("Auto (float32)", lambda: matrix_exp_sum(A_real_jax, B_real_jax, dtype='float32')),
        ("Tiled (complex64)", lambda: operator(A_jax, B_jax, method='tiled')),
        ("vmap (complex64)", lambda: operator(A_jax, B_jax, method='vmap')),
    ]
    
    for name, func in methods:
        # Warm up
        _ = func().block_until_ready()
        
        # Time
        start = time.time()
        for _ in range(5):
            _ = func().block_until_ready()
        elapsed = (time.time() - start) / 5
        
        print(f"   {name:20s}: {elapsed*1000:.2f} ms")


if __name__ == "__main__":
    example_usage()