import pytest

from conftest import fail_on_pvlib_version

from pvlib._deprecation import pvlibDeprecationWarning, deprecated

@pytest.mark.xfail(strict=True,
                   reason='fail_on_pvlib_version should cause test to fail')
@fail_on_pvlib_version('0.0')
def test_fail_on_pvlib_version():
    pass


@fail_on_pvlib_version('100000.0')
def test_fail_on_pvlib_version_pass():
    pass


@pytest.mark.xfail(strict=True, reason='ensure that the test is called')
@fail_on_pvlib_version('100000.0')
def test_fail_on_pvlib_version_fail_in_test():
    raise Exception


# set up to test passing arguments to conftest.fail_on_pvlib_version
@pytest.fixture(scope='function')
def some_data():
    return "some data"


def alt_func(*args):
    print(args[0])
    return


deprec_func = deprecated('0.8', alternative='alt_func',
                         name='deprec_func', removal='0.9')(alt_func)


@fail_on_pvlib_version('0.9', some_data)
def test_deprecated_09(some_data):
    with pytest.warns(pvlibDeprecationWarning):
        deprec_func(some_data)
