# the has/skip patterns closely follow the examples set by
# the xray/xarray project

import sys
import platform

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

try:
    import siphon
    has_siphon = True
except ImportError:
    has_siphon = False

def requires_scipy(test):
    return test if has_scipy else unittest.skip('requires scipy')(test)

def requires_siphon(test):
    return test if has_siphon else unittest.skip('requires siphon')(test)

def incompatible_conda_linux_py3(test):
    """
    Test won't work in Python 3.x due to Anaconda issue.
    """
    major = sys.version_info[0]
    minor = sys.version_info[1]
    system = platform.system()

    if major == 3 and system == 'Linux':
        out = unittest.skip('error on Linux Python 3 due to Anaconda')(test)
    else:
        out = test

    return out
