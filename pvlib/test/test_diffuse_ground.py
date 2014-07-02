import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import pandas as pd

from nose.tools import raises

from pvlib.location import Location
import pvlib.clearsky
import pvlib.solarposition
import pvlib.diffuse_ground

# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

ephem_data = pvlib.solarposition.get_solarposition(times, tus)

irrad_data = pvlib.clearsky.haurwitz(ephem_data['zenith'])
ghi = irrad_data['GHI']


# the test functions

def test_diffuse_ground_simple_float():
    pvlib.diffuse_ground.get_diffuse_ground(40, 900)

def test_diffuse_ground_simple_series():
    ground_irrad = pvlib.diffuse_ground.get_diffuse_ground(40, ghi)
    assert ground_irrad.name == 'diffuse_ground'
    
def test_diffuse_ground_albedo_0():
    ground_irrad = pvlib.diffuse_ground.get_diffuse_ground(40, ghi, albedo=0)
    assert 0 == ground_irrad.all()

@raises(KeyError)
def test_diffuse_ground_albedo_invalid_surface():
    pvlib.diffuse_ground.get_diffuse_ground(40, ghi, surface_type='invalid')
    
def test_diffuse_ground_albedo_surface():
    pvlib.diffuse_ground.get_diffuse_ground(40, ghi, surface_type='sand')