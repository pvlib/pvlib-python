import inspect
import os
import platform

import numpy as np
import pandas as pd
from pkg_resources import parse_version
import pytest

import pvlib

pvlib_base_version = \
    parse_version(parse_version(pvlib.__version__).base_version)


# decorator takes one argument: the base version for which it should fail
# for example @fail_on_pvlib_version('0.7') will cause a test to fail
# on pvlib versions 0.7a, 0.7b, 0.7rc1, etc.
# test function may not take args, kwargs, or fixtures.
def fail_on_pvlib_version(version):
    # second level of decorator takes the function under consideration
    def wrapper(func):
        # third level defers computation until the test is called
        # this allows the specific test to fail at test runtime,
        # rather than at decoration time (when the module is imported)
        def inner():
            # fail if the version is too high
            if pvlib_base_version >= parse_version(version):
                pytest.fail('the tested function is scheduled to be '
                            'removed in %s' % version)
            # otherwise return the function to be executed
            else:
                return func()
        return inner
    return wrapper


# commonly used directories in the tests
test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
data_dir = os.path.join(test_dir, os.pardir, 'data')


has_python2 = parse_version(platform.python_version()) < parse_version('3')

platform_is_windows = platform.system() == 'Windows'
skip_windows = pytest.mark.skipif(platform_is_windows,
                                  reason='does not run on windows')

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

requires_scipy = pytest.mark.skipif(not has_scipy, reason='requires scipy')


try:
    import tables
    has_tables = True
except ImportError:
    has_tables = False

requires_tables = pytest.mark.skipif(not has_tables, reason='requires tables')


try:
    import ephem
    has_ephem = True
except ImportError:
    has_ephem = False

requires_ephem = pytest.mark.skipif(not has_ephem, reason='requires ephem')


def pandas_0_17():
    return parse_version(pd.__version__) >= parse_version('0.17.0')


needs_pandas_0_17 = pytest.mark.skipif(
    not pandas_0_17(), reason='requires pandas 0.17 or greater')


def numpy_1_10():
    return parse_version(np.__version__) >= parse_version('1.10.0')


needs_numpy_1_10 = pytest.mark.skipif(
    not numpy_1_10(), reason='requires numpy 1.10 or greater')


def pandas_0_22():
    return parse_version(pd.__version__) >= parse_version('0.22.0')


needs_pandas_0_22 = pytest.mark.skipif(
    not pandas_0_22(), reason='requires pandas 0.22 or greater')


def has_spa_c():
    try:
        from pvlib.spa_c_files.spa_py import spa_calc
    except ImportError:
        return False
    else:
        return True


requires_spa_c = pytest.mark.skipif(not has_spa_c(), reason="requires spa_c")


def has_numba():
    try:
        import numba
    except ImportError:
        return False
    else:
        vers = numba.__version__.split('.')
        if int(vers[0] + vers[1]) < 17:
            return False
        else:
            return True


requires_numba = pytest.mark.skipif(not has_numba(), reason="requires numba")

try:
    import siphon
    has_siphon = True
except ImportError:
    has_siphon = False

requires_siphon = pytest.mark.skipif(not has_siphon,
                                     reason='requires siphon')

try:
    import netCDF4  # noqa: F401
    has_netCDF4 = True
except ImportError:
    has_netCDF4 = False

requires_netCDF4 = pytest.mark.skipif(not has_netCDF4,
                                      reason='requires netCDF4')

try:
    import pvfactors  # noqa: F401
    has_pvfactors = True
except ImportError:
    has_pvfactors = False

requires_pvfactors = pytest.mark.skipif(not has_pvfactors,
                                        reason='requires pvfactors')
