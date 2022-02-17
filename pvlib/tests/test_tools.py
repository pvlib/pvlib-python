import pytest

from pvlib import tools
import numpy as np


@pytest.mark.parametrize('keys, input_dict, expected', [
    (['a', 'b'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a', 'b', 'd'], {'a': 1, 'b': 2, 'c': 3}, {'a': 1, 'b': 2}),
    (['a'], {}, {}),
    (['a'], {'b': 2}, {})
])
def test_build_kwargs(keys, input_dict, expected):
    kwargs = tools._build_kwargs(keys, input_dict)
    assert kwargs == expected


def _obj_test_golden_sect(params, loc):
    return params[loc] * (1. - params['c'] * params[loc]**params['n'])


@pytest.mark.parametrize('params, lb, ub, expected, func', [
    ({'c': 1., 'n': 1.}, 0., 1., 0.5, _obj_test_golden_sect),
    ({'c': 1e6, 'n': 6.}, 0., 1., 0.07230200263994839, _obj_test_golden_sect),
    ({'c': 0.2, 'n': 0.3}, 0., 100., 89.14332727531685, _obj_test_golden_sect)
])
def test__golden_sect_DataFrame(params, lb, ub, expected, func):
    v, x = tools._golden_sect_DataFrame(params, lb, ub, func)
    assert np.isclose(x, expected, atol=1e-8)


def test__golden_sect_DataFrame_atol():
    params = {'c': 0.2, 'n': 0.3}
    expected = 89.14332727531685
    v, x = tools._golden_sect_DataFrame(
        params, 0., 100., _obj_test_golden_sect, atol=1e-12)
    assert np.isclose(x, expected, atol=1e-12)


def test__golden_sect_DataFrame_vector():
    params = {'c': np.array([1., 2.]), 'n': np.array([1., 1.])}
    lower = np.array([0., 0.001])
    upper = np.array([1.1, 1.2])
    expected = np.array([0.5, 0.25])
    v, x = tools._golden_sect_DataFrame(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8)


def test__golden_sect_DataFrame_nans():
    # nan in bounds
    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
    lower = np.array([0., 0.001, np.nan])
    upper = np.array([1.1, 1.2, 1.])
    expected = np.array([0.5, 0.25, np.nan])
    v, x = tools._golden_sect_DataFrame(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8, equal_nan=True)
    # nan in function values
    params = {'c': np.array([1., 2., np.nan]), 'n': np.array([1., 1., 1.])}
    lower = np.array([0., 0.001, 0.])
    upper = np.array([1.1, 1.2, 1.])
    expected = np.array([0.5, 0.25, np.nan])
    v, x = tools._golden_sect_DataFrame(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8, equal_nan=True)
    # all nan in bounds
    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
    lower = np.array([np.nan, np.nan, np.nan])
    upper = np.array([1.1, 1.2, 1.])
    expected = np.array([np.nan, np.nan, np.nan])
    v, x = tools._golden_sect_DataFrame(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8, equal_nan=True)
