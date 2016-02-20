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
    expected = pd.DataFrame(
        np.array([[    0.        ,     0.        ,     0.        ],
                  [    0.        ,     0.        ,     0.        ],
                  [   51.47811191,   265.33462162,    84.48262202],
                  [  105.008507  ,   832.29100407,   682.67761951],
                  [  121.97988054,   901.31821834,  1008.02102657],
                  [  112.57957512,   867.76297247,   824.61702926],
                  [   76.69672675,   588.8462898 ,   254.5808329 ],
                  [    0.        ,     0.        ,     0.        ],
                  [    0.        ,     0.        ,     0.        ]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times_localized, tus.latitude, tus.longitude)
    assert_frame_equal(expected, out)


def test_ineichen_supply_linke():
    expected = pd.DataFrame(np.array(
        [[    0.        ,     0.        ,     0.        ],
         [    0.        ,     0.        ,     0.        ],
         [   40.16490879,   321.71856556,    80.12815294],
         [   95.14336873,   876.49252839,   703.47605855],
         [  118.4587024 ,   939.81646535,  1042.34480815],
         [  105.36645492,   909.11265773,   851.32459694],
         [   61.91187639,   647.35889938,   257.42691896],
         [    0.        ,     0.        ,     0.        ],
         [    0.        ,     0.        ,     0.        ]]),
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
    expected = pd.DataFrame(
        np.array([[    0.        ,     0.        ,     0.        ],
                  [    0.        ,     0.        ,     0.        ],
                  [   53.90422388,   257.01655613,    85.87406435],
                  [  101.34055688,   842.92925705,   686.39337307],
                  [  117.7573735 ,   909.70367947,  1012.04184961],
                  [  108.6233401 ,   877.30589626,   828.49118038],
                  [   75.23108133,   602.06895546,   257.10961202],
                  [    0.        ,     0.        ,     0.        ],
                  [    0.        ,     0.        ,     0.        ]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times_localized)
    out = clearsky.ineichen(times_localized, tus.latitude, tus.longitude,
                            linke_turbidity=3,
                            airmass_model='simple')
    assert_frame_equal(expected, out)


@requires_scipy
def test_lookup_linke_turbidity():
    times = pd.date_range(start='2014-06-24', end='2014-06-25',
                          freq='12h', tz=tus.tz)
    # expect same value on 2014-06-24 0000 and 1200, and
    # diff value on 2014-06-25
    expected = pd.Series(np.array([3.10126582, 3.10126582, 3.11443038]),
                         index=times)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude, tus.longitude)
    assert_series_equal(expected, out)


@requires_scipy
def test_lookup_linke_turbidity_nointerp():
    times = pd.date_range(start='2014-06-24', end='2014-06-25',
                          freq='12h', tz=tus.tz)
    # expect same value for all days
    expected = pd.Series(np.array([3., 3., 3.]), index=times)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude, tus.longitude,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)


@requires_scipy
def test_lookup_linke_turbidity_months():
    times = pd.date_range(start='2014-04-01', end='2014-07-01',
                          freq='1M', tz=tus.tz)
    expected = pd.Series(np.array([2.8943038, 2.97316456, 3.18025316]),
                         index=times)
    out = clearsky.lookup_linke_turbidity(times, tus.latitude,
                                          tus.longitude)
    assert_series_equal(expected, out)


@requires_scipy
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
