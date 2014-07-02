import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import pandas as pd

from nose.tools import raises

from pvlib.location import Location
import pvlib.clearsky
import pvlib.solarposition
import pvlib.diffuse_sky

# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

ephem_data = pvlib.solarposition.get_solarposition(times, tus, method='pyephem')

irrad_data = pvlib.clearsky.ineichen(times, tus, solarposition_method='pyephem')


# the test functions

def test_isotropic_float():
    pvlib.diffuse_sky.isotropic(40, 100)
    
def test_isotropic_series():
    pvlib.diffuse_sky.isotropic(40, irrad_data['DHI'])

def test_klucher_series_float():
    pvlib.diffuse_sky.klucher(40, 180, 100, 900, 20, 180)
    
def test_klucher_series():
    pvlib.diffuse_sky.klucher(40, 180, irrad_data['DHI'], irrad_data['GHI'],
                              ephem_data['apparent_zenith'],
                              ephem_data['apparent_azimuth']) 
    