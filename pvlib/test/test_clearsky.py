import logging
pvl_logger = logging.getLogger('pvlib')

import datetime
from collections import namedtuple

import numpy as np
import pandas as pd

from nose.tools import raises

import pvlib.clearsky
import pvlib.solarposition

# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

Location = namedtuple('Location', ['latitude', 'longitude', 'altitude', 'tz'])
tus = Location(32.2, -111, 700, 'US/Arizona')

times_localized = times.tz_localize(tus.tz)

ephem_data = pvlib.solarposition.get_solarposition(times, tus)



# test the ineichen clear sky model implementation in a few ways

def test_ineichen_required():
    # the clearsky function should lookup the linke turbidity on its own
    pvlib.clearsky.ineichen(times, tus)
    
def test_ineichen_supply_linke():
    pvlib.clearsky.ineichen(times, tus, linke_turbidity=3)

def test_ineichen_solpos():
    pvlib.clearsky.ineichen(times, tus, linke_turbidity=3,
                            solarposition_method='pvlib')

def test_ineichen_airmass():
    pvlib.clearsky.ineichen(times, tus, linke_turbidity=3,
                            airmass_model='simple')



# test the haurwitz clear sky implementation
def test_haurwitz():
    pvlib.clearsky.haurwitz(ephem_data['zenith'])

