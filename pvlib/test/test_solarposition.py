import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import numpy.testing as npt
import pandas as pd

from nose.tools import raises, assert_almost_equals
from nose.plugins.skip import SkipTest
from pandas.util.testing import assert_frame_equal, assert_index_equal

from pvlib.location import Location
from pvlib import solarposition

from . import requires_ephem, incompatible_pandas_0131

# setup times and locations to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24),
                      end=datetime.datetime(2014,6,26), freq='15Min')

tus = Location(32.2, -111, 'US/Arizona', 700) # no DST issues possible
# In 2003, DST in US was from April 6 to October 26
golden_mst = Location(39.742476, -105.1786, 'MST', 1830.14) # no DST issues possible
golden = Location(39.742476, -105.1786, 'America/Denver', 1830.14) # DST issues possible

times_localized = times.tz_localize(tus.tz)

tol = 5

expected = pd.DataFrame({'elevation': 39.872046,
                         'apparent_zenith': 50.111622,
                         'azimuth': 194.340241,
                         'apparent_elevation': 39.888378},
                        index=['2003-10-17T12:30:30Z'])

# the physical tests are run at the same time as the NREL SPA test.
# pyephem reproduces the NREL result to 2 decimal places.
# this doesn't mean that one code is better than the other.


def test_spa_c_physical():
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    try:
        ephem_data = solarposition.spa_c(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    except ImportError:
        raise SkipTest
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected, ephem_data[expected.columns])


def test_spa_c_physical_dst():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    try:
        ephem_data = solarposition.spa_c(times, golden.latitude,
                                         golden.longitude,
                                         pressure=82000,
                                         temperature=11)
    except ImportError:
        raise SkipTest
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected, ephem_data[expected.columns])


def test_spa_python_numpy_physical():
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                          golden_mst.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected, ephem_data[expected.columns])


def test_spa_python_numpy_physical_dst():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_python(times, golden.latitude,
                                          golden.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected, ephem_data[expected.columns])


def test_spa_python_numba_physical():
    try:
        import numba
    except ImportError:
        raise SkipTest
    vers = numba.__version__.split('.')
    if int(vers[0] + vers[1]) < 17:
        raise SkipTest

    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                          golden_mst.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numba', numthreads=1)
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected, ephem_data[expected.columns])


def test_spa_python_numba_physical_dst():
    try:
        import numba
    except ImportError:
        raise SkipTest
    vers = numba.__version__.split('.')
    if int(vers[0] + vers[1]) < 17:
        raise SkipTest

    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_python(times, golden.latitude,
                                          golden.longitude, pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numba', numthreads=1)
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected, ephem_data[expected.columns])


@incompatible_pandas_0131
def test_get_sun_rise_set_transit():
    south = Location(-35.0, 0.0, tz='UTC')
    times = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 0),
                              datetime.datetime(2004, 12, 4, 0)]
                             ).tz_localize('UTC')
    sunrise = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 7, 8, 15),
                                datetime.datetime(2004, 12, 4, 4, 38, 57)]
                               ).tz_localize('UTC').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 17, 1, 4),
                               datetime.datetime(2004, 12, 4, 19, 2, 2)]
                              ).tz_localize('UTC').tolist()
    result = solarposition.get_sun_rise_set_transit(times, south.latitude,
                                                    south.longitude,
                                                    delta_t=64.0)
    frame = pd.DataFrame({'sunrise':sunrise, 'sunset':sunset}, index=times)
    result_rounded = pd.DataFrame(index=result.index)
    # need to iterate because to_datetime does not accept 2D data
    # the rounding fails on pandas < 0.17
    for col, data in result.iteritems():
        result_rounded[col] = pd.to_datetime(
            np.floor(data.values.astype(np.int64) / 1e9)*1e9, utc=True)

    del result_rounded['transit']
    assert_frame_equal(frame, result_rounded)


    # tests from USNO
    # Golden
    golden = Location(39.0, -105.0, tz='MST')
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2),
                              datetime.datetime(2015, 8, 2),]
                             ).tz_localize('MST')
    sunrise = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 7, 19, 2),
                                datetime.datetime(2015, 8, 2, 5, 1, 26)
                                ]).tz_localize('MST').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 16, 49, 10),
                               datetime.datetime(2015, 8, 2, 19, 11, 31)
                               ]).tz_localize('MST').tolist()
    result = solarposition.get_sun_rise_set_transit(times, golden.latitude,
                                                    golden.longitude,
                                                    delta_t=64.0)
    frame = pd.DataFrame({'sunrise':sunrise, 'sunset':sunset}, index=times)
    result_rounded = pd.DataFrame(index=result.index)
    # need to iterate because to_datetime does not accept 2D data
    # the rounding fails on pandas < 0.17
    for col, data in result.iteritems():
        result_rounded[col] = (pd.to_datetime(
            np.floor(data.values.astype(np.int64) / 1e9)*1e9, utc=True)
            .tz_convert('MST'))

    del result_rounded['transit']
    assert_frame_equal(frame, result_rounded)


@requires_ephem
def test_pyephem_physical():
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.pyephem(times, golden_mst.latitude,
                                       golden_mst.longitude, pressure=82000,
                                       temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected.round(2),
                       ephem_data[this_expected.columns].round(2))

@requires_ephem
def test_pyephem_physical_dst():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30), periods=1,
                          freq='D', tz=golden.tz)
    ephem_data = solarposition.pyephem(times, golden.latitude,
                                       golden.longitude, pressure=82000,
                                       temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    assert_frame_equal(this_expected.round(2),
                       ephem_data[this_expected.columns].round(2))

@requires_ephem
def test_calc_time():
    import pytz
    import math
    # validation from USNO solar position calculator online

    epoch = datetime.datetime(1970,1,1)
    epoch_dt = pytz.utc.localize(epoch)

    loc = tus
    loc.pressure = 0
    actual_time = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, 8, 30))
    lb = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, tol))
    ub = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, 10))
    alt = solarposition.calc_time(lb, ub, loc.latitude, loc.longitude,
                                  'alt', math.radians(24.7))
    az = solarposition.calc_time(lb, ub, loc.latitude, loc.longitude,
                                 'az', math.radians(116.3))
    actual_timestamp = (actual_time - epoch_dt).total_seconds()

    assert_almost_equals((alt.replace(second=0, microsecond=0) -
                          epoch_dt).total_seconds(), actual_timestamp)
    assert_almost_equals((az.replace(second=0, microsecond=0) -
                          epoch_dt).total_seconds(), actual_timestamp)

@requires_ephem
def test_earthsun_distance():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D')
    distance = solarposition.pyephem_earthsun_distance(times).values[0]
    assert_almost_equals(1, distance, 0)


def test_ephemeris_physical():
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.ephemeris(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])


def test_ephemeris_physical_dst():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.ephemeris(times, golden.latitude,
                                         golden.longitude, pressure=82000,
                                         temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])

@raises(ValueError)
def test_get_solarposition_error():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 pressure=82000,
                                                 temperature=11,
                                                 method='error this')

def test_get_solarposition_pressure():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 pressure=82000,
                                                 temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])

    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 pressure=0.0,
                                                 temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    try:
        assert_frame_equal(this_expected, ephem_data[this_expected.columns])
    except AssertionError:
        pass
    else:
        raise AssertionError

def test_get_solarposition_altitude():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 altitude=golden.altitude,
                                                 temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])

    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 altitude=0.0,
                                                 temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    try:
        assert_frame_equal(this_expected, ephem_data[this_expected.columns])
    except AssertionError:
        pass
    else:
        raise AssertionError

def test_get_solarposition_no_kwargs():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])
