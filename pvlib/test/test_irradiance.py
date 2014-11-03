import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import pandas as pd

from nose.tools import raises

from ..location import Location
from .. import clearsky
from .. import solarposition
from .. import irradiance
from .. import atmosphere

# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

ephem_data = solarposition.get_solarposition(times, tus, method='pyephem')

irrad_data = clearsky.ineichen(times, tus, solarposition_method='pyephem')

dni_et = irradiance.extraradiation(times.dayofyear)

ghi = irrad_data['GHI']


# the test functions. these are almost all functional tests.
# need to add physical tests.

def test_extraradiation():
    irradiance.extraradiation(300)

def test_grounddiffuse_simple_float():
    irradiance.grounddiffuse(40, 900)

def test_grounddiffuse_simple_series():
    ground_irrad = irradiance.grounddiffuse(40, ghi)
    assert ground_irrad.name == 'diffuse_ground'
    
def test_grounddiffuse_albedo_0():
    ground_irrad = irradiance.grounddiffuse(40, ghi, albedo=0)
    assert 0 == ground_irrad.all()

@raises(KeyError)
def test_grounddiffuse_albedo_invalid_surface():
    irradiance.grounddiffuse(40, ghi, surface_type='invalid')
    
def test_grounddiffuse_albedo_surface():
    irradiance.grounddiffuse(40, ghi, surface_type='sand')
    
def test_isotropic_float():
    irradiance.isotropic(40, 100)
    
def test_isotropic_series():
    irradiance.isotropic(40, irrad_data['DHI'])

def test_klucher_series_float():
    irradiance.klucher(40, 180, 100, 900, 20, 180)
    
def test_klucher_series():
    irradiance.klucher(40, 180, irrad_data['DHI'], irrad_data['GHI'],
                              ephem_data['apparent_zenith'],
                              ephem_data['apparent_azimuth']) 
    
def test_haydavies():
    irradiance.haydavies(40, 180, irrad_data['DHI'], irrad_data['DNI'],
                                dni_et,
                                ephem_data['apparent_zenith'],
                                ephem_data['apparent_azimuth']) 
                                
def test_reindl():
    irradiance.reindl(40, 180, irrad_data['DHI'], irrad_data['DNI'],
                             irrad_data['GHI'], dni_et,
                             ephem_data['apparent_zenith'],
                             ephem_data['apparent_azimuth'])         
                             
def test_king():
    irradiance.king(40, irrad_data['DHI'], irrad_data['GHI'],
                           ephem_data['apparent_zenith'])                              
                             
def test_perez():
    AM = atmosphere.relativeairmass(ephem_data['apparent_zenith'])
    irradiance.perez(40, 180, irrad_data['DHI'], irrad_data['DNI'],
                            dni_et,
                            ephem_data['apparent_zenith'],
                            ephem_data['apparent_azimuth'],
                            AM) 