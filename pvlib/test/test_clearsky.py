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
    assert 'ghi' in clearsky_data.columns
    assert 'dni' in clearsky_data.columns
    assert 'dhi' in clearsky_data.columns

# test the haurwitz clear sky implementation
def test_haurwitz():
    clearsky.haurwitz(ephem_data['zenith'])

def test_haurwitz_keys():
    clearsky_data = clearsky.haurwitz(ephem_data['zenith'])
    assert 'ghi' in clearsky_data.columns
