import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import pandas as pd

from nose.tools import raises, assert_almost_equals

from pvlib.location import Location
from pvlib import clearsky
from pvlib import solarposition
from pvlib import irradiance
from pvlib import atmosphere

# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

ephem_data = solarposition.get_solarposition(times, tus, method='pyephem')

irrad_data = clearsky.ineichen(times, tus, linke_turbidity=3,
                               solarposition_method='pyephem')

dni_et = irradiance.extraradiation(times.dayofyear)

ghi = irrad_data['GHI']


# the test functions. these are almost all functional tests.
# need to add physical tests.

def test_extraradiation():
    assert_almost_equals(1382, irradiance.extraradiation(300), -1)
    
def test_extraradiation_dtindex():
    irradiance.extraradiation(times)
    
def test_extraradiation_doyarray():
    irradiance.extraradiation(times.dayofyear)
    
def test_extraradiation_asce():
    assert_almost_equals(1382, irradiance.extraradiation(300, method='asce'), -1)
    
def test_extraradiation_spencer():
    assert_almost_equals(1382, irradiance.extraradiation(300, method='spencer'), -1)
    
def test_extraradiation_ephem_dtindex():
    irradiance.extraradiation(times, method='pyephem')
    
def test_extraradiation_ephem_scalar():
    assert_almost_equals(1382, irradiance.extraradiation(300, method='pyephem').values[0], -1)
    
def test_extraradiation_ephem_doyarray():
    irradiance.extraradiation(times.dayofyear, method='pyephem')

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