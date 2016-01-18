import logging
pvl_logger = logging.getLogger('pvlib')

import numpy as np
import pandas as pd

from nose.tools import raises
from numpy.testing import assert_almost_equal
from pandas.util.testing import assert_frame_equal, assert_series_equal

from pvlib.location import Location
from pvlib import clearsky
from pvlib import solarposition

from . import requires_scipy

# setup times and location to be tested.
tus = Location(32.2, -111, 'US/Arizona', 700)
times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h')
times_localized = times.tz_localize(tus.tz)

ephem_data = solarposition.get_solarposition(times_localized, tus.latitude,
                                             tus.longitude)

@requires_scipy
def test_ineichen_required():
    # the clearsky function should call lookup_linke_turbidity by default
    expected = pd.DataFrame(np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [51.0100314,259.73341927,82.76689082],
                                      [104.99512329,832.22000002,682.43280974],
                                      [121.97931179,901.31646645,1008.00975362],
                                      [112.5726345,867.73434086,824.48415382],
                                      [76.61228483,587.82419004,253.67624301],
                                      [0.,0.,0.],
                                      [0.,0.,0.]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times_localized, tus.latitude, tus.longitude)
    assert_frame_equal(expected, out)
    

def test_ineichen_supply_linke():
    expected = pd.DataFrame(np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [39.81862097,316.23284759,78.48350328],
                                      [95.12705301,876.43232906,703.24156169],
                                      [118.45796469,939.81499487,1042.33401282],
                                      [105.35769022,909.0884868,851.19721202],
                                      [61.83556162,646.45362207,256.55983299],
                                      [0.,0.,0.],
                                      [0.,0.,0.]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times_localized, tus.latitude, tus.longitude,
                            altitude=tus.altitude,
                            linke_turbidity=3)
    assert_frame_equal(expected, out)


def test_ineichen_solpos():
    clearsky.ineichen(times_localized, tus.latitude, tus.longitude,
                      linke_turbidity=3,
                      solarposition_method='ephemeris')


def test_ineichen_airmass():
    expected = pd.DataFrame(np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [53.52826665,250.64463008,84.17386592],
                                      [101.32775752,842.86030421,686.14824255],
                                      [117.7568185,909.70199089,1012.03056908],
                                      [108.61662929,877.27820363,828.35817853],
                                      [75.15682967,601.03375193,256.19976209],
                                      [0.,0.,0.],
                                      [0.,0.,0.]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times_localized, tus.latitude, tus.longitude,
                            linke_turbidity=3,
                            airmass_model='simple')
    assert_frame_equal(expected, out)


def test_lookup_linke_turbidity():
    times = pd.date_range(start='2014-06-24', end='2014-06-25',
                          freq='12h', tz=tus.tz)
    # expect same value on 2014-06-24 0000 and 1200, and
    # diff value on 2014-06-25
    expected = pd.Series(np.array([3.10126582, 3.10126582, 3.11443038]),
                         index=times)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude, tus.longitude)
    assert_series_equal(expected, out)


def test_lookup_linke_turbidity_nointerp():
    times = pd.date_range(start='2014-06-24', end='2014-06-25',
                          freq='12h', tz=tus.tz)
    # expect same value for all days
    expected = pd.Series(np.array([3., 3., 3.]), index=times)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude, tus.longitude,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)


def test_lookup_linke_turbidity_months():
    times = pd.date_range(start='2014-04-01', end='2014-07-01',
                          freq='1M', tz=tus.tz)
    expected = pd.Series(np.array([2.8943038, 2.97316456, 3.18025316]),
                         index=times)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude,
                                          tus.longitude)
    assert_series_equal(expected, out)


def test_lookup_linke_turbidity_nointerp_months():
    times = pd.date_range(start='2014-04-10', end='2014-07-10',
                          freq='1M', tz=tus.tz)
    expected = pd.Series(np.array([2.85, 2.95, 3.]), index=times)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude, tus.longitude,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)
    # changing the dates shouldn't matter if interp=False
    times = pd.date_range(start='2014-04-05', end='2014-07-05',
                          freq='1M', tz=tus.tz)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude, tus.longitude,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)


def test_haurwitz():
    expected = pd.DataFrame(np.array([[0.],
                                      [0.],
                                      [82.85934048],
                                      [699.74514735],
                                      [1016.50198354],
                                      [838.32103769],
                                      [271.90853863],
                                      [0.],
                                      [0.]]),
                             columns=['ghi'], index=times_localized)
    out = clearsky.haurwitz(ephem_data['zenith'])
    assert_frame_equal(expected, out)
