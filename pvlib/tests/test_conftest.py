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


# set up to test using fixtures with function decorated with
# conftest.fail_on_pvlib_version
@pytest.fixture()
def some_data():
    return "some data"


def alt_func(*args):
    return args


deprec_func = deprecated('350.8', alternative='alt_func',
                         name='deprec_func', removal='350.9')(alt_func)


@fail_on_pvlib_version('350.9')
def test_use_fixture_with_decorator(some_data):
    # test that the correct data is returned by the some_data fixture
    assert some_data == "some data"
    with pytest.warns(pvlibDeprecationWarning):  # test for deprecation warning
        deprec_func(some_data)
