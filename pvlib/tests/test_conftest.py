import pytest

from conftest import fail_on_pvlib_version


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
