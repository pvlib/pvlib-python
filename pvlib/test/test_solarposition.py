import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import pandas as pd

from nose.tools import raises, assert_almost_equals
from pandas.util.testing import assert_frame_equal

from ..location import Location
from .. import solarposition


# setup times and locations to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='15Min')

tus = Location(32.2, -111, 'US/Arizona', 700) # no DST issues possible
golden_mst = Location(39.742476, -105.1786, 'MST', 1830.14) # no DST issues possible
golden = Location(39.742476, -105.1786, 'America/Denver', 1830.14) # DST issues possible

times_localized = times.tz_localize(tus.tz)

# the physical tests are run at the same time as the NREL SPA test.
# pyephem reproduces the NREL result to 2 decimal places.
# this doesn't mean that one code is better than the other. 


def test_spa_physical():
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30), periods=1, freq='D')
    ephem_data = solarposition.spa(times, golden_mst).ix[0]
    
    assert_almost_equals(50.111622, ephem_data['zenith'], 6)
    assert_almost_equals(194.340241, ephem_data['azimuth'], 6)
    assert_almost_equals(39.888378, ephem_data['elevation'], 6)
    
    
    
def test_spa_physical_dst():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30), periods=1, freq='D')
    ephem_data = solarposition.spa(times, golden).ix[0]
    
    assert_almost_equals(50.111622, ephem_data['zenith'], 6)
    assert_almost_equals(194.340241, ephem_data['azimuth'], 6)
    assert_almost_equals(39.888378, ephem_data['elevation'], 6)



def test_spa_localization():    
    assert_frame_equal(solarposition.spa(times, tus), solarposition.spa(times_localized, tus))



def test_pyephem_physical():
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30), periods=1, freq='D')
    ephem_data = solarposition.pyephem(times, golden_mst, pressure=82000, temperature=11).ix[0]
    
    assert_almost_equals(50.111622, ephem_data['apparent_zenith'], 2)
    assert_almost_equals(194.340241, ephem_data['apparent_azimuth'], 2)
    assert_almost_equals(39.888378, ephem_data['apparent_elevation'], 2)
    


def test_pyephem_physical_dst():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30), periods=1, freq='D')
    ephem_data = solarposition.pyephem(times, golden, pressure=82000, temperature=11).ix[0]
    
    assert_almost_equals(50.111622, ephem_data['apparent_zenith'], 2)
    assert_almost_equals(194.340241, ephem_data['apparent_azimuth'], 2)
    assert_almost_equals(39.888378, ephem_data['apparent_elevation'], 2)



def test_pyephem_localization():  
    assert_frame_equal(solarposition.pyephem(times, tus), solarposition.pyephem(times_localized, tus))


def test_calc_time():
    import pytz
    import math
    # validation from USNO solar position calculator online

    epoch = datetime.datetime(1970,1,1)
    epoch_dt = pytz.utc.localize(epoch)
    
    loc = tus
    loc.pressure = 0
    actual_time = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, 8, 30))
    lb = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, 6))
    ub = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, 10))
    alt = solarposition.calc_time(lb, ub, loc, 'alt', math.radians(24.7))
    az = solarposition.calc_time(lb, ub, loc, 'az', math.radians(116.3))
    actual_timestamp = (actual_time - epoch_dt).total_seconds()
    
    assert_almost_equals((alt.replace(second=0, microsecond=0) - 
                          epoch_dt).total_seconds(), actual_timestamp)
    assert_almost_equals((az.replace(second=0, microsecond=0) - 
                          epoch_dt).total_seconds(), actual_timestamp)
    
        
# add tests for daylight savings time?
