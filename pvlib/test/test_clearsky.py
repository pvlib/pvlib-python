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

# setup times and location to be tested.
tus = Location(32.2, -111, 'US/Arizona', 700)
times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h')
times_localized = times.tz_localize(tus.tz)

ephem_data = solarposition.get_solarposition(times, tus)


def test_ineichen_required():
    # the clearsky function should call lookup_linke_turbidity by default
    # will fail without scipy
    expected = pd.DataFrame(np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [40.53660309,302.47614235,78.1470311],
                                      [98.88372629,865.98938602,699.93403875],
                                      [122.57870881,931.83716051,1038.62116584],
                                      [109.30270612,899.88002304,847.68806472],
                                      [64.25699595,629.91187925,254.53048144],
                                      [0.,0.,0.],
                                      [0.,0.,0.]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times, tus)
    assert_frame_equal(expected, out)
    

def test_ineichen_supply_linke():
    expected = pd.DataFrame(np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [40.18673553,322.0649964,80.23287692],
                                      [95.14405816,876.49507151,703.48596755],
                                      [118.45873721,939.81653473,1042.34531752],
                                      [105.36671577,909.113377,851.3283881],
                                      [61.91607984,647.40869542,257.47471759],
                                      [0.,0.,0.],
                                      [0.,0.,0.]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times, tus, linke_turbidity=3)
    assert_frame_equal(expected, out)


def test_ineichen_solpos():
    clearsky.ineichen(times, tus, linke_turbidity=3,
                      solarposition_method='ephemeris')


def test_ineichen_airmass():
    expected = pd.DataFrame(np.array([[0.,0.,0.],
                                      [0.,0.,0.],
                                      [41.70761136,293.72203458,78.22953786],
                                      [95.20590465,876.1650047,703.31872722],
                                      [118.46089555,939.8078753,1042.33896321],
                                      [105.39577655,908.97804342,851.24640259],
                                      [62.35382269,642.91022293,256.55363539],
                                      [0.,0.,0.],
                                      [0.,0.,0.]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times, tus, linke_turbidity=3,
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
