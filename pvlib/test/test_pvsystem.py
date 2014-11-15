import logging
pvl_logger = logging.getLogger('pvlib')

import inspect
import os

from nose.tools import assert_equals

from pvlib import tmy
from pvlib import pvsystem

pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(tmy)))

tmy3_testfile = os.path.join(pvlib_abspath, 'data', '703165TY.csv')
tmy2_testfile = os.path.join(pvlib_abspath, 'data', '12839.tm2')

tmy3_data, tmy3_metadata = tmy.readtmy3(tmy3_testfile)
tmy2_data, tmy2_metadata = tmy.readtmy2(tmy2_testfile)

def test_systemdef_tmy3():
    expected = {'TZ': -9.0,
                'albedo': 0.1,
                'altitude': 7.0,
                'latitude': 55.317,
                'longitude': -160.517,
                'name': '"SAND POINT"',
                'parallel_modules': 5,
                'series_modules': 5,
                'surfaz': 0,
                'surftilt': 0}
    assert_equals(expected, pvsystem.systemdef(tmy3_metadata, 0, 0, .1, 5, 5))
    
def test_systemdef_tmy2():
    expected = {'TZ': -5,
                'albedo': 0.1,
                'altitude': 2.0,
                'latitude': 25.8,
                'longitude': -80.26666666666667,
                'name': 'MIAMI',
                'parallel_modules': 5,
                'series_modules': 5,
                'surfaz': 0,
                'surftilt': 0}
    assert_equals(expected, pvsystem.systemdef(tmy2_metadata, 0, 0, .1, 5, 5))