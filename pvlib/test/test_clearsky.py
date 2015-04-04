import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import pandas as pd

from nose.tools import raises

from numpy.testing import assert_almost_equal

from pvlib.location import Location
from pvlib import clearsky
from pvlib import solarposition

# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

ephem_data = solarposition.get_solarposition(times, tus)



# test the ineichen clear sky model implementation in a few ways

def test_ineichen_required():
    # the clearsky function should lookup the linke turbidity on its own
    # will fail without scipy
    clearsky.ineichen(times, tus)
    
def test_ineichen_supply_linke():
    clearsky.ineichen(times, tus, linke_turbidity=3)

def test_ineichen_solpos():
    clearsky.ineichen(times, tus, linke_turbidity=3,
                      solarposition_method='pyephem')

def test_ineichen_airmass():
    clearsky.ineichen(times, tus, linke_turbidity=3,
                      airmass_model='simple')

def test_ineichen_keys():
    clearsky_data = clearsky.ineichen(times, tus, linke_turbidity=3)
    assert 'GHI' in clearsky_data.columns
    assert 'DNI' in clearsky_data.columns
    assert 'DHI' in clearsky_data.columns

# test the haurwitz clear sky implementation
def test_haurwitz():
    clearsky.haurwitz(ephem_data['zenith'])

def test_haurwitz_keys():
    clearsky_data = clearsky.haurwitz(ephem_data['zenith'])
    assert 'GHI' in clearsky_data.columns
    
    
# test DISC
def test_disc_keys():
    clearsky_data = clearsky.ineichen(times, tus, linke_turbidity=3)
    disc_data = clearsky.disc(clearsky_data['GHI'], ephem_data['zenith'], 
                              ephem_data.index)
    assert 'DNI_gen_DISC' in disc_data.columns
    assert 'Kt_gen_DISC' in disc_data.columns
    assert 'AM' in disc_data.columns

def test_disc_value():
    times = pd.DatetimeIndex(['2014-06-24T12-0700','2014-06-24T18-0700'])
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    pressure = 93193.
    disc_data = clearsky.disc(ghi, zenith, times, pressure=pressure)
    assert_almost_equal(disc_data['DNI_gen_DISC'].values,
                        np.array([830.46, 676.09]), 1)

def test_dirint():
    clearsky_data = clearsky.ineichen(times, tus, linke_turbidity=3)
    pressure = 93193.
    dirint_data = clearsky.dirint(clearsky_data['GHI'], ephem_data['zenith'], 
                                  ephem_data.index, pressure=pressure)

def test_dirint_value():
    times = pd.DatetimeIndex(['2014-06-24T12-0700','2014-06-24T18-0700'])
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    pressure = 93193.
    dirint_data = clearsky.dirint(ghi, zenith, times, pressure=pressure)
    assert_almost_equal(dirint_data.values,
                        np.array([928.85, 688.26]), 1)

def test_dirint_tdew():
    times = pd.DatetimeIndex(['2014-06-24T12-0700','2014-06-24T18-0700'])
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    pressure = 93193.
    dirint_data = clearsky.dirint(ghi, zenith, times, pressure=pressure,
                                  temp_dew=10)
    assert_almost_equal(dirint_data.values,
                        np.array([934.06, 640.67]), 1)

def test_dirint_no_delta_kt():
    times = pd.DatetimeIndex(['2014-06-24T12-0700','2014-06-24T18-0700'])
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    pressure = 93193.
    dirint_data = clearsky.dirint(ghi, zenith, times, pressure=pressure,
                                  use_delta_kt_prime=False)
    assert_almost_equal(dirint_data.values,
                        np.array([901.56, 674.87]), 1)

def test_dirint_coeffs():
    coeffs = clearsky._get_dirint_coeffs()
    assert coeffs[0,0,0,0] == 0.385230
    assert coeffs[0,1,2,1] == 0.229970
    assert coeffs[3,2,6,3] == 1.032260