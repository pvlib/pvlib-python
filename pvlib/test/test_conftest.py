import pytest

from conftest import fail_on_pvlib_version


@fail_on_pvlib_version('0.0')
def test_fail_on_pvlib_version():
    pass


@fail_on_pvlib_version('100000.0')
def test_fail_on_pvlib_version_pass():
    pass


@fail_on_pvlib_version('100000.0')
def test_fail_on_pvlib_version_fail_in_test():
    raise Exception
