import jax
import jax.numpy as jnp

from goom.config import config
from goom.custom_abs import goom_abs
from goom.custom_log import goom_log
from goom.custom_exp import goom_exp

def to_goom(x: jax.Array) -> jax.Array:
    abs_x = goom_abs(x)
    log_abs_x = goom_log(abs_x)
    real = jnp.astype(log_abs_x, config.float_dtype)
    x_is_neg = (x < 0)
    # todo: allow this to return real tensors if config.cast_all_logs_to_complex is False and x is 
    # non-negative
    return jnp.complex64(real + 1j*x_is_neg*jnp.pi)

def from_goom(x: jax.Array) -> jax.Array:
    return goom_exp(x).real

