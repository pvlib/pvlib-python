import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import pandas as pd

from nose.tools import raises

from pvlib.location import Location
import pvlib.solarposition as solarposition
import pvlib.airmass as airmass


# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24), 
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

ephem_data = solarposition.get_solarposition(times, tus)



# two functions combined will generate unique unit tests for each model
def test_airmasses():
    models = ['simple', 'kasten1966', 'youngirvine1967', 'kastenyoung1989',
              'gueymard1993', 'young1994', 'pickering2002', 'invalid']
    for model in models:
        yield run_airmass, ephem_data['zenith'], model
    
def run_airmass(zenith, model):
    airmass.relativeairmass(zenith, model)
    
    
    
def test_absoluteairmass():
    relative_am = airmass.relativeairmass(ephem_data['zenith'], 'simple')
    airmass.absoluteairmass(relative_am)
    airmass.absoluteairmass(relative_am, pressure=100000)