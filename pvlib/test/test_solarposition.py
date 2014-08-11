import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import pandas as pd

from nose.tools import raises

from ..location import Location
from .. import solarposition


# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

def test_get_solarposition_basic():    
    ephem_data = solarposition.get_solarposition(times, tus)
    
def test_get_solarposition_pvlib():    
    ephem_data = solarposition.get_solarposition(times, tus, method='pvlib')
    ephem_data = solarposition.get_solarposition(times_localized, tus, method='pvlib')

def test_get_solarposition_pyephem():  
    try:
        ephem_data = solarposition.get_solarposition(times, tus, method='pyephem')
        ephem_data = solarposition.get_solarposition(times_localized, tus, method='pyephem')
    except NameError:
        pvl_logger.error('PyEphem not found. could not run test.')
    
@raises(ValueError)
def test_get_solarposition_invalid(): 
    solarposition.get_solarposition(times, tus, method='invalid')
        
        
# add tests for daylight savings time?