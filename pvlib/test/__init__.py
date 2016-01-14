import sys

try:
    import unittest2 as unittest
except ImportError:
    import unittest

try:
    import scipy
    has_scipy = True
except ImportError:
    has_scipy = False

def requires_scipy(test):
    return test if has_scipy else unittest.skip('requires scipy')(test)

def incompatible_conda_py3(test):
    """
    Test won't work in Python 3.x due to Anaconda issue.
    """
    major = sys.version_info[0]
    minor = sys.version_info[1]

    if major == 3:
        out = unittest.skip('error on Python 3 due to anaconda')(test)
    else:
        out = test

    return out
