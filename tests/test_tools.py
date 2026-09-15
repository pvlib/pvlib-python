from datetime import datetime
from io import StringIO
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest

from pvlib import location, tools


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
    # invalid value for coordinate argument
    with pytest.raises(ValueError, match="coordinate must be"):
        tools._degrees_to_index(degrees=22.0, coordinate='width')


@pytest.mark.parametrize('args, args_idx', [
    # no pandas.Series or pandas.DataFrame args
    ((1,), None),
    (([1],), None),
    ((np.array(1),), None),
    ((np.array([1]),), None),
    # has pandas.Series or pandas.DataFrame args
    ((pd.DataFrame([1], index=[1]),), 0),
    ((pd.Series([1], index=[1]),), 0),
    ((1, pd.Series([1], index=[1]),), 1),
    ((1, pd.DataFrame([1], index=[1]),), 1),
    # first pandas.Series or pandas.DataFrame is used
    ((1, pd.Series([1], index=[1]), pd.DataFrame([2], index=[2]),), 1),
    ((1, pd.DataFrame([1], index=[1]), pd.Series([2], index=[2]),), 1),
])
def test_get_pandas_index(args, args_idx):
    index = tools.get_pandas_index(*args)

    if args_idx is None:
        assert index is None
    else:
        pd.testing.assert_index_equal(args[args_idx].index, index)


@pytest.mark.parametrize('data_in,expected', [
    (np.array([1, 2, 3, 4, 5]),
     np.array([0.2, 0.4, 0.6, 0.8, 1])),
    (np.array([[0, 1, 2], [0, 3, 6]]),
     np.array([[0, 0.5, 1], [0, 0.5, 1]])),
    (pd.Series([1, 2, 3, 4, 5]),
     pd.Series([0.2, 0.4, 0.6, 0.8, 1])),
    (pd.DataFrame({"a": [0, 1, 2], "b": [0, 2, 8]}),
     pd.DataFrame({"a": [0, 0.5, 1], "b": [0, 0.25, 1]})),
    # test with NaN and all zeroes
    (pd.DataFrame({"a": [0, np.nan, 1], "b": [0, 0, 0]}),
     pd.DataFrame({"a": [0, np.nan, 1], "b": [np.nan]*3})),
    # test with negative values
    (np.array([1, 2, -3, 4, -5]),
     np.array([0.2, 0.4, -0.6, 0.8, -1])),
    (pd.Series([-2, np.nan, 1]),
     pd.Series([-1, np.nan, 0.5])),
])
def test_normalize_max2one(data_in, expected):
    result = tools.normalize_max2one(data_in)
    assert_allclose(result, expected)


def test_datetime_to_djd():
    expected = 27201.47934027778
    dt_aware = datetime(1974, 6, 22, 18, 30, 15, tzinfo=ZoneInfo("Etc/GMT+5"))
    assert tools.datetime_to_djd(dt_aware) == expected
    dt_naive_utc = datetime(1974, 6, 22, 23, 30, 15)
    assert tools.datetime_to_djd(dt_naive_utc) == expected


def test_djd_to_datetime():
    djd = 27201.47934027778
    tz = "Etc/GMT+5"

    expected = datetime(1974, 6, 22, 18, 30, 15, tzinfo=ZoneInfo(tz))
    assert tools.djd_to_datetime(djd, tz) == expected

    expected = datetime(1974, 6, 22, 23, 30, 15, tzinfo=ZoneInfo("UTC"))
    assert tools.djd_to_datetime(djd) == expected


def test__file_context_manager():
    with tempfile.TemporaryDirectory() as td:
        # make a test file
        filename = os.path.join(td, 'test.txt')
        with open(filename, 'w') as fh:
            fh.write('test content')

        # test with filename as string:
        with tools._file_context_manager(filename) as obj:
            assert obj.read() == "test content"

        # test with filename as Path:
        with tools._file_context_manager(Path(filename)) as obj:
            assert obj.read() == "test content"

        # test with file object:
        with open(filename, "r") as f:
            with tools._file_context_manager(f) as obj:
                assert obj.read() == "test content"

    # test with buffer:
    buffer = StringIO("test content")
    with tools._file_context_manager(buffer) as obj:
        assert obj.read() == "test content"
