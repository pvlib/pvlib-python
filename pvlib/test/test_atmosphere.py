import datetime
import itertools

import numpy as np
import pandas as pd

from nose.tools import raises
from nose.tools import assert_almost_equals

from pvlib.location import Location
from pvlib import solarposition
from pvlib import atmosphere


# setup times and location to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24),
                      end=datetime.datetime(2014,6,26), freq='1Min')

tus = Location(32.2, -111, 'US/Arizona', 700)

times_localized = times.tz_localize(tus.tz)

ephem_data = solarposition.get_solarposition(times_localized, tus.latitude,
                                             tus.longitude)


# need to add physical tests instead of just functional tests

def test_pres2alt():
    atmosphere.pres2alt(100000)

def test_alt2press():
    atmosphere.pres2alt(1000)


# two functions combined will generate unique unit tests for each model
def test_airmasses():
    models = ['simple', 'kasten1966', 'youngirvine1967', 'kastenyoung1989',
              'gueymard1993', 'young1994', 'pickering2002']
    for model in models:
        yield run_airmass, model, ephem_data['zenith']


def run_airmass(model, zenith):
    atmosphere.relativeairmass(zenith, model)


@raises(ValueError)
def test_airmass_invalid():
    atmosphere.relativeairmass(ephem_data['zenith'], 'invalid')


def test_absoluteairmass():
    relative_am = atmosphere.relativeairmass(ephem_data['zenith'], 'simple')
    atmosphere.absoluteairmass(relative_am)
    atmosphere.absoluteairmass(relative_am, pressure=100000)


def test_absoluteairmass_numeric():
    atmosphere.absoluteairmass(2)


def test_absoluteairmass_nan():
    np.testing.assert_equal(np.nan, atmosphere.absoluteairmass(np.nan))


def test_gueymard94_pw():
    temp_air = np.array([0, 20, 40])
    relative_humidity = np.array([0, 30, 100])
    temps_humids = np.array(
        list(itertools.product(temp_air, relative_humidity)))
    pws = atmosphere.gueymard94_pw(temps_humids[:, 0], temps_humids[:, 1])

    expected = np.array(
        [  0.1       ,   0.33702061,   1.12340202,   0.1       ,
         1.12040963,   3.73469877,   0.1       ,   3.44859767,  11.49532557])

    np.testing.assert_allclose(pws, expected, atol=0.01)
