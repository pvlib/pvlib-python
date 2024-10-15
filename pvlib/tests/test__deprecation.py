"""
Test the _deprecation module.
"""

import pytest

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
