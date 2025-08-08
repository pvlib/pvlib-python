"""
Test the _deprecation module.
"""

import pytest
import pandas as pd

from pvlib import _deprecation
from .conftest import fail_on_pvlib_version

import warnings


@pytest.mark.xfail(strict=True,
                   reason='fail_on_pvlib_version should cause test to fail')
@fail_on_pvlib_version('0.0')
def test_fail_on_pvlib_version():
    pass  # pragma: no cover


@fail_on_pvlib_version('100000.0')
def test_fail_on_pvlib_version_pass():
    pass


@pytest.mark.xfail(strict=True, reason='ensure that the test is called')
@fail_on_pvlib_version('100000.0')
def test_fail_on_pvlib_version_fail_in_test():
    raise Exception


# set up to test using fixtures with function decorated with
# conftest.fail_on_pvlib_version
@pytest.fixture
def some_data():
    return "some data"


def alt_func(*args):
    return args


@pytest.fixture
def deprec_func():
    return _deprecation.deprecated(
        "350.8", alternative="alt_func", name="deprec_func", removal="350.9"
    )(alt_func)


@fail_on_pvlib_version('350.9')
def test_use_fixture_with_decorator(some_data, deprec_func):
    # test that the correct data is returned by the some_data fixture
    assert some_data == "some data"
    with pytest.warns(_deprecation.pvlibDeprecationWarning):
        # test for custom deprecation warning provided by pvlib
        deprec_func(some_data)


@pytest.fixture
def renamed_kwarg_func():
    """Returns a function decorated by renamed_kwarg_warning.
    This function is called 'func' and has a docstring equal to 'docstring'.
    """

    @_deprecation.renamed_kwarg_warning(
        "0.1.0", "old_kwarg", "new_kwarg", "0.2.0"
    )
    def func(new_kwarg):
        """docstring"""
        return new_kwarg

    return func


def test_renamed_kwarg_warning(renamed_kwarg_func):
    # assert decorated function name and docstring are unchanged
    assert renamed_kwarg_func.__name__ == "func"
    assert renamed_kwarg_func.__doc__ == "docstring"

    # assert no warning is raised when using the new kwarg
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert renamed_kwarg_func(new_kwarg=1) == 1  # as keyword argument
        assert renamed_kwarg_func(1) == 1  # as positional argument

    # assert a warning is raised when using the old kwarg
    with pytest.warns(Warning, match="Parameter 'old_kwarg' has been renamed"):
        assert renamed_kwarg_func(old_kwarg=1) == 1

    # assert an error is raised when using both the old and new kwarg
    with pytest.raises(ValueError, match="they refer to the same parameter."):
        renamed_kwarg_func(old_kwarg=1, new_kwarg=2)

    # assert when not providing any of them
    with pytest.raises(
        TypeError, match="missing 1 required positional argument"
    ):
        renamed_kwarg_func()


def test_renamed_key_items_warning():
    """Test the renamed_key_items_warning decorator."""
    # Test on a dictionary
    data_dict = {
        "new_key1": [1, 2, 3],
        "new_key2": [4, 5, 6],
        "another_key": [7, 8, 9],
    }
    data_dict_wrapped = _deprecation.renamed_key_items_warning(
        "0.1.0", {"old_key1": "new_key1"}, "0.2.0"
    )(data_dict)

    # Check that the new key is present in the wrapped object
    assert "new_key1" in data_dict_wrapped
    assert "new_key2" in data_dict_wrapped
    assert "another_key" in data_dict_wrapped
    assert "old_key1" not in data_dict_wrapped
    # Check that the old key still exists in the wrapped object
    assert data_dict_wrapped["new_key1"] == [1, 2, 3]
    assert data_dict_wrapped["new_key2"] == [4, 5, 6]
    assert data_dict_wrapped["another_key"] == [7, 8, 9]
    with pytest.warns(Warning, match="use `new_key1` instead of `old_key1`."):
        assert data_dict_wrapped["old_key1"] == [1, 2, 3]
    # check yet again, to ensure there is no weird persistences
    with pytest.warns(Warning, match="use `new_key1` instead of `old_key1`."):
        assert data_dict_wrapped["old_key1"] == [1, 2, 3]

    # Test on a DataFrame
    data_df = pd.DataFrame(data_dict)
    data_df = _deprecation.renamed_key_items_warning(
        "0.1.0", {"old_key1": "new_key1", "old_key2": "new_key2"}, "0.2.0"
    )(data_df)

    assert "new_key1" in data_df.columns
    assert data_df.new_key1 is not None  # ensure attribute access works
    assert "new_key2" in data_df.columns
    assert "old_key1" not in data_df.columns
    assert "old_key2" not in data_df.columns
    # Check that the old key still exists in the DataFrame
    assert data_df["new_key1"].tolist() == [1, 2, 3]
    with pytest.warns(Warning, match="use `new_key1` instead of `old_key1`."):
        assert data_df["old_key1"].tolist() == [1, 2, 3]
    with pytest.warns(Warning, match="use `new_key1` instead of `old_key1`."):
        assert data_df["old_key1"].tolist() == [1, 2, 3]

    # Test chaining decorators, on a dict, first new_key1, then new_key2
    data_dict_wrapped = _deprecation.renamed_key_items_warning(
        "0.1.0", {"old_key1": "new_key1"}, "0.2.0"
    )(data_dict)
    data_dict_wrapped = _deprecation.renamed_key_items_warning(
        "0.3.0", {"old_key2": "new_key2"}, "0.4.0"
    )(data_dict_wrapped)
    # Check that the new keys are present in the wrapped object
    assert "new_key1" in data_dict_wrapped
    assert "new_key2" in data_dict_wrapped

    with pytest.warns(Warning, match="use `new_key1` instead of `old_key1`."):
        assert data_dict_wrapped["old_key1"] == [1, 2, 3]
    with pytest.warns(Warning, match="use `new_key2` instead of `old_key2`."):
        assert data_dict_wrapped["old_key2"] == [4, 5, 6]
