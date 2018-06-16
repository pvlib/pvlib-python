import sys
import platform

from pkg_resources import parse_version
import pandas as pd
import numpy as np
import pytest


skip_windows = pytest.mark.skipif('win' in sys.platform,
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
        return True
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
