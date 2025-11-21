import jax
import jax.numpy as jnp
import pytest

from test_gooms import generate_random_gooms
from goom.operations import (
    log_add_exp,
    log_sum_exp,
    log_mean_exp,
    log_cum_sum_exp,
    log_cum_mean_exp,
    log_negate_exp,
    scale,
    scaled_exp,
    log_triu_exp,
    log_tril_exp,
    log_rmsnorm_exp,
    log_unitsquash_exp,
)


# Helper function to check for NaNs
def _check_no_nans(x):
    assert not jnp.isnan(x.real).any()
    assert not jnp.isnan(x.imag).any()


# Fixture for random goom tensors
@pytest.fixture
def random_gooms_2d():
    key = jax.random.PRNGKey(0)
    shape = (10, 5)
    gooms = generate_random_gooms(key, shape)
    _check_no_nans(gooms)
    return gooms


@pytest.fixture
def random_gooms_pair():
    key = jax.random.PRNGKey(1)
    shape = (10, 5)
    key, subkey1, subkey2 = jax.random.split(key, 3)
    gooms1 = generate_random_gooms(subkey1, shape)
    gooms2 = generate_random_gooms(subkey2, shape)
    _check_no_nans(gooms1)
    _check_no_nans(gooms2)
    return gooms1, gooms2


# Tests for each function
def test_log_add_exp(random_gooms_pair):
    gooms1, gooms2 = random_gooms_pair
    result = log_add_exp(gooms1, gooms2)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_sum_exp(random_gooms_2d, axis):
    result = log_sum_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_mean_exp(random_gooms_2d, axis):
    result = log_mean_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_cum_sum_exp(random_gooms_2d, axis):
    result = log_cum_sum_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_cum_mean_exp(random_gooms_2d, axis):
    result = log_cum_mean_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


def test_log_negate_exp(random_gooms_2d):
    result = log_negate_exp(random_gooms_2d)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_scale(random_gooms_2d, axis):
    result = scale(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_scaled_exp(random_gooms_2d, axis):
    result = scaled_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


@pytest.mark.parametrize("offset", [0, 1, -1])
def test_log_triu_exp(random_gooms_2d, offset):
    result = log_triu_exp(random_gooms_2d, diagonal_offset=offset)
    _check_no_nans(result)


@pytest.mark.parametrize("offset", [0, 1, -1])
def test_log_tril_exp(random_gooms_2d, offset):
    result = log_tril_exp(random_gooms_2d, diagonal_offset=offset)
    _check_no_nans(result)


@pytest.mark.parametrize("axis", [0, 1])
def test_log_rmsnorm_exp(random_gooms_2d, axis):
    result = log_rmsnorm_exp(random_gooms_2d, axis=axis)
    _check_no_nans(result)


def test_log_unitsquash_exp(random_gooms_2d):
    result = log_unitsquash_exp(random_gooms_2d)
    _check_no_nans(result)
