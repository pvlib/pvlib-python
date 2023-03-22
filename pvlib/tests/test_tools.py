import pytest

from pvlib import tools
import numpy as np
import pandas as pd


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
    # some upper and lower bounds equal
    params = {'c': np.array([1., 2., 1.]), 'n': np.array([1., 1., 1.])}
    lower = np.array([0., 0.001, 1.])
    upper = np.array([1., 1.2, 1.])
    expected = np.array([0.5, 0.25, 1.0])  # x values for maxima
    v, x = tools._golden_sect_DataFrame(params, lower, upper,
                                        _obj_test_golden_sect)
    assert np.allclose(x, expected, atol=1e-8)
    # all upper and lower bounds equal, arrays of length 1
    params = {'c': np.array([1.]), 'n': np.array([1.])}
    lower = np.array([1.])
    upper = np.array([1.])
    expected = np.array([1.])  # x values for maxima
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


def test_degrees_to_index_1():
    """Test that _degrees_to_index raises an error when something other than
    'latitude' or 'longitude' is passed."""
    with pytest.raises(IndexError):  # invalid value for coordinate argument
        tools._degrees_to_index(degrees=22.0, coordinate='width')


def get_match_type_array_like_test_cases():
    return [
        # identity
        (np.array([1]), np.array([1]), lambda a, b: np.array_equal(a, b)),
        (np.array([1]), np.array([1.]), lambda a, b: np.array_equal(a, b)),
        (np.array([1.]), np.array([1]), lambda a, b: np.array_equal(a, b)),
        (np.array([1.]), np.array([1.]), lambda a, b: np.array_equal(a, b)),
        (pd.Series([1]), pd.Series([1]),
         lambda a, b: np.array_equal(a.to_numpy(), b.to_numpy())),
        (pd.Series([1]), pd.Series([1.]),
         lambda a, b: np.array_equal(a.to_numpy(), b.to_numpy())),
        (pd.Series([1.]), pd.Series([1]),
         lambda a, b: np.array_equal(a.to_numpy(), b.to_numpy())),
        (pd.Series([1.]), pd.Series([1.]),
         lambda a, b: np.array_equal(a.to_numpy(), b.to_numpy())),
        # np.ndarray to pd.Series
        (np.array([1]), pd.Series([1]),
         lambda a, b: np.array_equal(a, b.to_numpy())),
        (np.array([1]), pd.Series([1.]),
         lambda a, b: np.array_equal(a, b.to_numpy())),
        (np.array([1.]), pd.Series([1]),
         lambda a, b: np.array_equal(a, b.to_numpy())),
        (np.array([1.]), pd.Series([1.]),
         lambda a, b: np.array_equal(a, b.to_numpy())),
        # pd.Series to np.ndarray
        (pd.Series([1]), np.array([1]),
         lambda a, b: np.array_equal(a.to_numpy(), b)),
        (pd.Series([1]), np.array([1.]),
         lambda a, b: np.array_equal(a.to_numpy(), b)),
        (pd.Series([1.]), np.array([1]),
         lambda a, b: np.array_equal(a.to_numpy(), b)),
        (pd.Series([1.]), np.array([1.]),
         lambda a, b: np.array_equal(a.to_numpy(), b)),
        # x shorter than type_of
        (np.array([1]), np.array([1, 2]),
         lambda a, b: np.array_equal(a, b)),
        (np.array([1]), pd.Series([1, 2]),
         lambda a, b: np.array_equal(a, b.to_numpy())),
        (pd.Series([1]), np.array([1, 2]),
         lambda a, b: np.array_equal(a.to_numpy(), b)),
        (pd.Series([1]), pd.Series([1, 2]),
         lambda a, b: np.array_equal(a.to_numpy(), b.to_numpy())),
        # x longer than type_of
        (np.array([1, 2]), np.array([1]),
         lambda a, b: np.array_equal(a, b)),
        (np.array([1, 2]), pd.Series([1]),
         lambda a, b: np.array_equal(a, b.to_numpy())),
        (pd.Series([1, 2]), np.array([1]),
         lambda a, b: np.array_equal(a.to_numpy(), b)),
        (pd.Series([1, 2]), pd.Series([1]),
         lambda a, b: np.array_equal(a.to_numpy(), b.to_numpy()))
    ]


@pytest.mark.parametrize('x, type_of, content_equal', [
    *get_match_type_array_like_test_cases()
])
def test_match_type_array_like(x, type_of, content_equal):
    x_matched = tools.match_type_array_like(x, type_of)

    assert type(x_matched) is type(type_of)
    assert content_equal(x, x_matched)


@pytest.mark.parametrize('x, type_of, content_equal', [
    (1, 1, lambda a, b: a == b),
    (1, 1., lambda a, b: a == b),
    (1., 1, lambda a, b: a == b),
    (1., 1., lambda a, b: a == b)
])
def test_match_type_numeric_scalar_to_scalar(x, type_of, content_equal):
    x_matched = tools.match_type_numeric(x, type_of)

    assert type(x) is type(x_matched)
    assert content_equal(x, x_matched)


@pytest.mark.parametrize('x, type_of, match_shape, content_equal', [
    # scalar to array with shape (N,)
    (1, np.array([1]), True, lambda a, b: a == b.item()),
    (1, np.array([1, 2]), True,
     lambda a, b: np.array_equal(np.array([a, a], dtype=int), b)),
    (1, np.array([1, 2]), False, lambda a, b: a == b.item()),
    (1, np.array([1., 2.]), False, lambda a, b: np.float64(a) == b.item()),
    (1, pd.Series([1]), True, lambda a, b: a == b.item()),
    (1, pd.Series([1, 2]), True,
     lambda a, b: np.array_equal(np.array([a, a], dtype=int), b)),
    (1, pd.Series([1., 2.]), True,
     lambda a, b: np.array_equal(np.array([a, a], dtype=np.float64), b)),
    (1, pd.Series([1, 2]), False, lambda a, b: a == b.item()),
    (1, pd.Series([1., 2.]), False, lambda a, b: np.float64(a) == b.item()),
    (1., np.array([1]), True, lambda a, b: a == b.item()),
    (1., np.array([1, 2]), True, lambda a, b: np.array_equal([a, a], b)),
    (1., np.array([1, 2]), False, lambda a, b: a == b.item()),
    (1., np.array([1., 2.]), False, lambda a, b: np.float64(a) == b.item()),
    (1., pd.Series([1]), True, lambda a, b: a == b.item()),
    (1., pd.Series([1, 2]), True, lambda a, b: np.array_equal([a, a], b)),
    (1., pd.Series([1., 2.]), True,
     lambda a, b: np.array_equal(np.array([a, a], dtype=np.float64), b)),
    (1., pd.Series([1, 2]), False, lambda a, b: a == b.item()),
    (1., pd.Series([1., 2.]), False, lambda a, b: np.float64(a) == b.item()),
    # scalar to np.ndarray with any shape. this does not work for pd.Series
    # because they only have shape (N,)
    (1, np.array([[1]]), True, lambda a, b: np.array_equal([[a]], b)),
    (1, np.array([[1, 1]]), True, lambda a, b: np.array_equal([[a, a]], b)),
    (1, np.array([[1], [1]]), True,
     lambda a, b: np.array_equal([[a], [a]], b)),
    (1, np.array([[[1], [1]], [[1], [1]]]), True,
     lambda a, b: np.array_equal([[[a], [a]], [[a], [a]]], b)),
    (1, np.array([[1]]), False, lambda a, b: a == b.item()),
    (1, np.array([[1, 1]]), False, lambda a, b: a == b.item()),
    (1, np.array([[1], [1]]), False, lambda a, b: a == b.item()),
    (1, np.array([[[1], [1]], [[1], [1]]]), False, lambda a, b: a == b.item())
])
def test_match_type_numeric_scalar_to_array_like(x, type_of, match_shape,
                                                 content_equal):
    x_matched = tools.match_type_numeric(x, type_of, match_shape=match_shape)

    assert type(x_matched) is type(type_of)
    assert content_equal(x, x_matched)


@pytest.mark.parametrize('x, type_of, content_equal', [
    (np.array([1]), 1, lambda a, b: a.item() == b),
    (np.array([1.]), 1, lambda a, b: a.item() == b),
    (np.array([1]), 1., lambda a, b: a.item() == b),
    (np.array([1.]), 1., lambda a, b: a.item() == b),
    (pd.Series([1]), 1, lambda a, b: a.item() == b),
    (pd.Series([1.]), 1, lambda a, b: a.item() == b),
    (pd.Series([1]), 1., lambda a, b: a.item() == b),
    (pd.Series([1.]), 1., lambda a, b: a.item() == b)
])
def test_match_type_numeric_array_like_to_scalar(x, type_of, content_equal):
    x_matched = tools.match_type_numeric(x, type_of)

    assert type(x.item()) is type(x_matched)
    assert content_equal(x, x_matched)


@pytest.mark.parametrize('x, type_of, content_equal', [
    *get_match_type_array_like_test_cases()
])
def test_match_type_numeric_array_like_to_array_like(x, type_of,
                                                     content_equal):
    x_matched = tools.match_type_numeric(x, type_of)

    assert type(x_matched) is type(type_of)
    assert content_equal(x, x_matched)


@pytest.mark.parametrize('args, expected_type', [
    ((1, np.array([1]), pd.Series([1])), np.isscalar),
    ((np.array([1]), 1, np.array([1]), pd.Series([1])),
     lambda a: isinstance(a, np.ndarray)),
    ((pd.Series([1]), 1, np.array([1]), pd.Series([1])),
     lambda a: isinstance(a, pd.Series))
])
def test_match_type_all_numeric(args, expected_type):
    assert all(map(expected_type, tools.match_type_all_numeric(*args)))


@pytest.mark.parametrize('args, match_shape', [
    ((np.array([1, 2]), 1), True),
    ((pd.Series([1, 2]), 1), True),
    ((np.array([1, 2]), 1), False),
    ((pd.Series([1, 2]), 1), False)
])
def test_match_type_all_numeric_match_size(args, match_shape):
    first, second = tools.match_type_all_numeric(*args,
                                                 match_shape=match_shape)

    assert type(first) is type(second)
    if match_shape:
        assert first.size == second.size
    else:
        assert second.size == 1
