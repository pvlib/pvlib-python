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


def test_enforce_numpy_arrays():
    """ Check that when pandas objects are included in the inputs, the
    wrapped function receives numpy arrays and the outputs are turned into
    pandas series """

    # Create a test function and decorate it
    @tools.enforce_numpy_arrays
    def fun_function(a, b, c, d, kwarg_1=None, kwarg_2=None):
        assert isinstance(a, np.ndarray)
        assert isinstance(b, np.ndarray)
        assert isinstance(c, np.ndarray)
        assert isinstance(d, str)
        assert isinstance(kwarg_1, np.ndarray)
        assert isinstance(kwarg_2, np.ndarray)
        out_1 = np.array([1, 2])
        out_2 = pd.Series([3, 4])
        out_3 = 5.
        return out_1, out_2, out_3

    # Check with no pandas inputs
    a = b = c = np.array([1, 2])
    d = 'string'
    kwarg_1 = np.array([1, 2])
    kwarg_2 = np.array([1, 2])
    out_1, out_2, out_3 = fun_function(a, b, c, d,
                                       kwarg_1=kwarg_1, kwarg_2=kwarg_2)
    assert isinstance(out_1, np.ndarray)
    assert isinstance(out_2, pd.Series)
    assert isinstance(out_3, float)

    # Check with some pandas inputs in both args and kwargs
    b = pd.Series([1, 2])
    c = pd.DataFrame([1, 2], columns=['e'], index=range(2))
    kwarg_1 = pd.Series([1, 2])
    kwarg_2 = pd.DataFrame([1, 2], columns=['kwarg_2'], index=range(2))
    out_1, out_2, out_3 = fun_function(a, b, c, d,
                                       kwarg_1=kwarg_1, kwarg_2=kwarg_2)
    assert isinstance(out_1, pd.Series)
    assert isinstance(out_2, pd.Series)
    assert isinstance(out_3, float)
