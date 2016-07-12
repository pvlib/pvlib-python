import sys
import platform

import pandas as pd
import numpy as np
import pytest


try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

requires_scipy = pytest.mark.skipif(not has_scipy, reason='requires scipy')


try:
    import ephem
    has_ephem = True
except ImportError:
    has_ephem = False

requires_ephem = pytest.mark.skipif(not has_ephem, reason='requires ephem')


incompatible_pandas_0131 = pytest.mark.skipif(
    pd.__version__ == '0.13.1', reason='requires numpy 1.10 or greater')


def numpy_1_10():
    version = tuple(map(int, np.__version__.split('.')))
    if version[0] <= 1 and version[1] < 10:
        return False
    else:
        return True

needs_numpy_1_10 = pytest.mark.skipif(
    not numpy_1_10(), reason='requires numpy 1.10 or greater')


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
