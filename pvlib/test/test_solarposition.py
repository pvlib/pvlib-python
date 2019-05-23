import calendar
import datetime
import warnings

import numpy as np
import pandas as pd

from pandas.util.testing import assert_frame_equal, assert_series_equal
from numpy.testing import assert_allclose
import pytest

from pvlib._deprecation import pvlibDeprecationWarning
from pvlib.location import Location
from pvlib import solarposition, spa

from conftest import (fail_on_pvlib_version, requires_ephem, needs_pandas_0_17,
                      requires_spa_c, requires_numba)


# setup times and locations to be tested.
times = pd.date_range(start=datetime.datetime(2014, 6, 24),
                      end=datetime.datetime(2014, 6, 26), freq='15Min')

tus = Location(32.2, -111, 'US/Arizona', 700)  # no DST issues possible
times_localized = times.tz_localize(tus.tz)

tol = 5


@pytest.fixture()
def golden():
    return Location(39.742476, -105.1786, 'America/Denver', 1830.14)


@pytest.fixture()
def golden_mst():
    return Location(39.742476, -105.1786, 'MST', 1830.14)


@pytest.fixture()
def expected_solpos():
    return _expected_solpos_df()


# hack to make tests work without too much modification while avoiding
# pytest 4.0 inability to call features directly
def _expected_solpos_df():
    return pd.DataFrame({'elevation': 39.872046,
                         'apparent_zenith': 50.111622,
                         'azimuth': 194.340241,
                         'apparent_elevation': 39.888378},
                        index=['2003-10-17T12:30:30Z'])


@pytest.fixture()
def expected_solpos_multi():
    return pd.DataFrame({'elevation': [39.872046, 39.505196],
                         'apparent_zenith': [50.111622, 50.478260],
                         'azimuth': [194.340241, 194.311132],
                         'apparent_elevation': [39.888378, 39.521740]},
                        index=['2003-10-17T12:30:30Z', '2003-10-18T12:30:30Z'])


@pytest.fixture()
def expected_rise_set_spa():
    # for Golden, CO, from NREL SPA website
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2),
                              datetime.datetime(2015, 8, 2),
                              ]).tz_localize('MST')
    sunrise = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 7, 21, 55),
                                datetime.datetime(2015, 8, 2, 5, 0, 27)
                                ]).tz_localize('MST').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 16, 47, 43),
                               datetime.datetime(2015, 8, 2, 19, 13, 58)
                               ]).tz_localize('MST').tolist()
    transit = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 12, 4, 45),
                                datetime.datetime(2015, 8, 2, 12, 6, 58)
                                ]).tz_localize('MST').tolist()
    return pd.DataFrame({'sunrise': sunrise,
                         'sunset': sunset,
                         'transit': transit},
                        index=times)


@pytest.fixture()
def expected_rise_set_ephem():
    # for Golden, CO, from USNO websites
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 1),
                              datetime.datetime(2015, 1, 2),
                              datetime.datetime(2015, 1, 3),
                              datetime.datetime(2015, 8, 2),
                              ]).tz_localize('MST')
    sunrise = pd.DatetimeIndex([datetime.datetime(2015, 1, 1, 7, 22, 0),
                                datetime.datetime(2015, 1, 2, 7, 22, 0),
                                datetime.datetime(2015, 1, 3, 7, 22, 0),
                                datetime.datetime(2015, 8, 2, 5, 0, 0)
                                ]).tz_localize('MST').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(2015, 1, 1, 16, 47, 0),
                               datetime.datetime(2015, 1, 2, 16, 48, 0),
                               datetime.datetime(2015, 1, 3, 16, 49, 0),
                               datetime.datetime(2015, 8, 2, 19, 13, 0)
                               ]).tz_localize('MST').tolist()
    transit = pd.DatetimeIndex([datetime.datetime(2015, 1, 1, 12, 4, 0),
                                datetime.datetime(2015, 1, 2, 12, 5, 0),
                                datetime.datetime(2015, 1, 3, 12, 5, 0),
                                datetime.datetime(2015, 8, 2, 12, 7, 0)
                                ]).tz_localize('MST').tolist()
    return pd.DataFrame({'sunrise': sunrise,
                         'sunset': sunset,
                         'transit': transit},
                        index=times)


@fail_on_pvlib_version('0.7')
def test_deprecated_07():
    tt = pd.DatetimeIndex(['2015-01-01 00:00:00']).tz_localize('MST')
    with pytest.warns(pvlibDeprecationWarning):
        solarposition.get_sun_rise_set_transit(tt,
                                               39.7,
                                               -105.2)


# the physical tests are run at the same time as the NREL SPA test.
# pyephem reproduces the NREL result to 2 decimal places.
# this doesn't mean that one code is better than the other.

@requires_spa_c
def test_spa_c_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_c(times, golden_mst.latitude,
                                     golden_mst.longitude,
                                     pressure=82000,
                                     temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_spa_c
def test_spa_c_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_c(times, golden.latitude,
                                     golden.longitude,
                                     pressure=82000,
                                     temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_spa_python_numpy_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                          golden_mst.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_spa_python_numpy_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_python(times, golden.latitude,
                                          golden.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@needs_pandas_0_17
def test_sun_rise_set_transit_spa(expected_rise_set_spa, golden):
    # solution from NREL SAP web calculator
    south = Location(-35.0, 0.0, tz='UTC')
    times = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 0),
                              datetime.datetime(2004, 12, 4, 0)]
                             ).tz_localize('UTC')
    sunrise = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 7, 8, 15),
                                datetime.datetime(2004, 12, 4, 4, 38, 57)]
                               ).tz_localize('UTC').tolist()
    sunset = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 17, 1, 4),
                               datetime.datetime(2004, 12, 4, 19, 2, 3)]
                              ).tz_localize('UTC').tolist()
    transit = pd.DatetimeIndex([datetime.datetime(1996, 7, 5, 12, 4, 36),
                                datetime.datetime(2004, 12, 4, 11, 50, 22)]
                               ).tz_localize('UTC').tolist()
    frame = pd.DataFrame({'sunrise': sunrise,
                          'sunset': sunset,
                          'transit': transit}, index=times)

    result = solarposition.sun_rise_set_transit_spa(times, south.latitude,
                                                    south.longitude,
                                                    delta_t=65.0)
    result_rounded = pd.DataFrame(index=result.index)
    # need to iterate because to_datetime does not accept 2D data
    # the rounding fails on pandas < 0.17
    for col, data in result.iteritems():
        result_rounded[col] = data.dt.round('1s')

    assert_frame_equal(frame, result_rounded)

    # test for Golden, CO compare to NREL SPA
    result = solarposition.sun_rise_set_transit_spa(
        expected_rise_set_spa.index, golden.latitude, golden.longitude,
        delta_t=65.0)

    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    # need to iterate because to_datetime does not accept 2D data
    for col, data in result.iteritems():
        result_rounded[col] = data.dt.round('s').tz_convert('MST')

    assert_frame_equal(expected_rise_set_spa, result_rounded)


@requires_ephem
def test_sun_rise_set_transit_ephem(expected_rise_set_ephem, golden):
    # test for Golden, CO compare to USNO, using local midnight
    result = solarposition.sun_rise_set_transit_ephem(
        expected_rise_set_ephem.index, golden.latitude, golden.longitude,
        next_or_previous='next', altitude=golden.altitude, pressure=0,
        temperature=11, horizon='-0:34')
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.iteritems():
        result_rounded[col] = data.dt.round('min').tz_convert('MST')
    assert_frame_equal(expected_rise_set_ephem, result_rounded)

    # test next sunrise/sunset with times
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 3, 0, 0),
                              datetime.datetime(2015, 1, 2, 10, 15, 0),
                              datetime.datetime(2015, 1, 2, 15, 3, 0),
                              datetime.datetime(2015, 1, 2, 21, 6, 7)
                              ]).tz_localize('MST')
    expected = pd.DataFrame(index=times,
                            columns=['sunrise', 'sunset'],
                            dtype='datetime64[ns]')
    expected['sunrise'] = pd.Series(index=times, data=[
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunrise'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'sunrise'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'sunrise'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'sunrise']])
    expected['sunset'] = pd.Series(index=times, data=[
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunset'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunset'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunset'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'sunset']])
    expected['transit'] = pd.Series(index=times, data=[
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'transit'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'transit'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'transit'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'transit']])

    result = solarposition.sun_rise_set_transit_ephem(times,
                                                      golden.latitude,
                                                      golden.longitude,
                                                      next_or_previous='next',
                                                      altitude=golden.altitude,
                                                      pressure=0,
                                                      temperature=11,
                                                      horizon='-0:34')
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.iteritems():
        result_rounded[col] = data.dt.round('min').tz_convert('MST')
    assert_frame_equal(expected, result_rounded)

    # test previous sunrise/sunset with times
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 3, 0, 0),
                              datetime.datetime(2015, 1, 2, 10, 15, 0),
                              datetime.datetime(2015, 1, 3, 3, 0, 0),
                              datetime.datetime(2015, 1, 3, 13, 6, 7)
                              ]).tz_localize('MST')
    expected = pd.DataFrame(index=times,
                            columns=['sunrise', 'sunset'],
                            dtype='datetime64[ns]')
    expected['sunrise'] = pd.Series(index=times, data=[
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 1), 'sunrise'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunrise'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunrise'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'sunrise']])
    expected['sunset'] = pd.Series(index=times, data=[
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 1), 'sunset'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 1), 'sunset'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunset'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'sunset']])
    expected['transit'] = pd.Series(index=times, data=[
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 1), 'transit'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 1), 'transit'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 2), 'transit'],
        expected_rise_set_ephem.loc[datetime.datetime(2015, 1, 3), 'transit']])

    result = solarposition.sun_rise_set_transit_ephem(
        times,
        golden.latitude, golden.longitude, next_or_previous='previous',
        altitude=golden.altitude, pressure=0, temperature=11, horizon='-0:34')
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.iteritems():
        result_rounded[col] = data.dt.round('min').tz_convert('MST')
    assert_frame_equal(expected, result_rounded)

    # test with different timezone
    times = times.tz_convert('UTC')
    expected = expected.tz_convert('UTC')  # resuse result from previous
    for col, data in expected.iteritems():
        expected[col] = data.dt.tz_convert('UTC')
    result = solarposition.sun_rise_set_transit_ephem(
        times,
        golden.latitude, golden.longitude, next_or_previous='previous',
        altitude=golden.altitude, pressure=0, temperature=11, horizon='-0:34')
    # round to nearest minute
    result_rounded = pd.DataFrame(index=result.index)
    for col, data in result.iteritems():
        result_rounded[col] = data.dt.round('min').tz_convert(times.tz)
    assert_frame_equal(expected, result_rounded)


@requires_ephem
def test_sun_rise_set_transit_ephem_error(expected_rise_set_ephem, golden):
    with pytest.raises(ValueError):
        solarposition.sun_rise_set_transit_ephem(expected_rise_set_ephem.index,
                                                 golden.latitude,
                                                 golden.longitude,
                                                 next_or_previous='other')
    tz_naive = pd.DatetimeIndex([datetime.datetime(2015, 1, 2, 3, 0, 0)])
    with pytest.raises(ValueError):
        solarposition.sun_rise_set_transit_ephem(tz_naive,
                                                 golden.latitude,
                                                 golden.longitude,
                                                 next_or_previous='next')


@requires_ephem
def test_sun_rise_set_transit_ephem_horizon(golden):
    times = pd.DatetimeIndex([datetime.datetime(2016, 1, 3, 0, 0, 0)
                              ]).tz_localize('MST')
    # center of sun disk
    center = solarposition.sun_rise_set_transit_ephem(
        times,
        latitude=golden.latitude, longitude=golden.longitude)
    edge = solarposition.sun_rise_set_transit_ephem(
        times,
        latitude=golden.latitude, longitude=golden.longitude, horizon='-0:34')
    result_rounded = (edge['sunrise'] - center['sunrise']).dt.round('min')

    sunrise_delta = datetime.datetime(2016, 1, 3, 7, 17, 11) - \
        datetime.datetime(2016, 1, 3, 7, 21, 33)
    expected = pd.Series(index=times,
                         data=sunrise_delta,
                         name='sunrise').dt.round('min')
    assert_series_equal(expected, result_rounded)


@requires_ephem
def test_pyephem_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.pyephem(times, golden_mst.latitude,
                                       golden_mst.longitude, pressure=82000,
                                       temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos.round(2),
                       ephem_data[expected_solpos.columns].round(2))


@requires_ephem
def test_pyephem_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.pyephem(times, golden.latitude,
                                       golden.longitude, pressure=82000,
                                       temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos.round(2),
                       ephem_data[expected_solpos.columns].round(2))


@requires_ephem
def test_calc_time():
    import pytz
    import math
    # validation from USNO solar position calculator online

    epoch = datetime.datetime(1970, 1, 1)
    epoch_dt = pytz.utc.localize(epoch)

    loc = tus
    loc.pressure = 0
    actual_time = pytz.timezone(loc.tz).localize(
        datetime.datetime(2014, 10, 10, 8, 30))
    lb = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, tol))
    ub = pytz.timezone(loc.tz).localize(datetime.datetime(2014, 10, 10, 10))
    alt = solarposition.calc_time(lb, ub, loc.latitude, loc.longitude,
                                  'alt', math.radians(24.7))
    az = solarposition.calc_time(lb, ub, loc.latitude, loc.longitude,
                                 'az', math.radians(116.3))
    actual_timestamp = (actual_time - epoch_dt).total_seconds()

    assert_allclose((alt.replace(second=0, microsecond=0) -
                     epoch_dt).total_seconds(), actual_timestamp)
    assert_allclose((az.replace(second=0, microsecond=0) -
                     epoch_dt).total_seconds(), actual_timestamp)


@requires_ephem
def test_earthsun_distance():
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D')
    distance = solarposition.pyephem_earthsun_distance(times).values[0]
    assert_allclose(1, distance, atol=0.1)


def test_ephemeris_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.ephemeris(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_ephemeris_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.ephemeris(times, golden.latitude,
                                         golden.longitude, pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_ephemeris_physical_no_tz(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 19, 30, 30),
                          periods=1, freq='D')
    ephem_data = solarposition.ephemeris(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_get_solarposition_error(golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    with pytest.raises(ValueError):
        solarposition.get_solarposition(times, golden.latitude,
                                        golden.longitude,
                                        pressure=82000,
                                        temperature=11,
                                        method='error this')


@pytest.mark.parametrize("pressure, expected", [
    (82000, _expected_solpos_df()),
    (90000, pd.DataFrame(
        np.array([[39.88997,   50.11003,  194.34024,   39.87205,   14.64151,
                   50.12795]]),
        columns=['apparent_elevation', 'apparent_zenith', 'azimuth',
                 'elevation', 'equation_of_time', 'zenith'],
        index=['2003-10-17T12:30:30Z']))
    ])
def test_get_solarposition_pressure(pressure, expected, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 pressure=pressure,
                                                 temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])


@pytest.mark.parametrize("altitude, expected", [
    (1830.14, _expected_solpos_df()),
    (2000, pd.DataFrame(
        np.array([[39.88788,   50.11212,  194.34024,   39.87205,   14.64151,
                   50.12795]]),
        columns=['apparent_elevation', 'apparent_zenith', 'azimuth',
                 'elevation', 'equation_of_time', 'zenith'],
        index=['2003-10-17T12:30:30Z']))
    ])
def test_get_solarposition_altitude(altitude, expected, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 altitude=altitude,
                                                 temperature=11)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])


@pytest.mark.parametrize("delta_t, method", [
    (None, 'nrel_numpy'),
    pytest.param(
        None, 'nrel_numba',
        marks=[pytest.mark.xfail(
            reason='spa.calculate_deltat not implemented for numba yet')]),
    (67.0, 'nrel_numba'),
    (67.0, 'nrel_numpy'),
    ])
def test_get_solarposition_deltat(delta_t, method, expected_solpos_multi,
                                  golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=2, freq='D', tz=golden.tz)
    with warnings.catch_warnings():
        # don't warn on method reload or num threads
        warnings.simplefilter("ignore")
        ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                     golden.longitude,
                                                     pressure=82000,
                                                     delta_t=delta_t,
                                                     temperature=11,
                                                     method=method)
    this_expected = expected_solpos_multi
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])


def test_get_solarposition_no_kwargs(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_ephem
def test_get_solarposition_method_pyephem(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 method='pyephem')
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_nrel_earthsun_distance():
    times = pd.DatetimeIndex([datetime.datetime(2015, 1, 2),
                              datetime.datetime(2015, 8, 2)]
                             ).tz_localize('MST')
    result = solarposition.nrel_earthsun_distance(times, delta_t=64.0)
    expected = pd.Series(np.array([0.983289204601, 1.01486146446]),
                         index=times)
    assert_series_equal(expected, result)

    times = datetime.datetime(2015, 1, 2)
    result = solarposition.nrel_earthsun_distance(times, delta_t=64.0)
    expected = pd.Series(np.array([0.983289204601]),
                         index=pd.DatetimeIndex([times, ]))
    assert_series_equal(expected, result)


def test_equation_of_time():
    times = pd.date_range(start="1/1/2015 0:00", end="12/31/2015 23:00",
                          freq="H")
    output = solarposition.spa_python(times, 37.8, -122.25, 100)
    eot = output['equation_of_time']
    eot_rng = eot.max() - eot.min()  # range of values, around 30 minutes
    eot_1 = solarposition.equation_of_time_spencer71(times.dayofyear)
    eot_2 = solarposition.equation_of_time_pvcdrom(times.dayofyear)
    assert np.allclose(eot_1 / eot_rng, eot / eot_rng, atol=0.3)  # spencer
    assert np.allclose(eot_2 / eot_rng, eot / eot_rng, atol=0.4)  # pvcdrom


def test_declination():
    times = pd.date_range(start="1/1/2015 0:00", end="12/31/2015 23:00",
                          freq="H")
    atmos_refract = 0.5667
    delta_t = spa.calculate_deltat(times.year, times.month)
    unixtime = np.array([calendar.timegm(t.timetuple()) for t in times])
    _, _, declination = spa.solar_position(unixtime, 37.8, -122.25, 100,
                                           1013.25, 25, delta_t, atmos_refract,
                                           sst=True)
    declination = np.deg2rad(declination)
    declination_rng = declination.max() - declination.min()
    declination_1 = solarposition.declination_cooper69(times.dayofyear)
    declination_2 = solarposition.declination_spencer71(times.dayofyear)
    a, b = declination_1 / declination_rng, declination / declination_rng
    assert np.allclose(a, b, atol=0.03)  # cooper
    a, b = declination_2 / declination_rng, declination / declination_rng
    assert np.allclose(a, b, atol=0.02)  # spencer


def test_analytical_zenith():
    times = pd.date_range(start="1/1/2015 0:00", end="12/31/2015 23:00",
                          freq="H").tz_localize('Etc/GMT+8')
    lat, lon = 37.8, -122.25
    lat_rad = np.deg2rad(lat)
    output = solarposition.spa_python(times, lat, lon, 100)
    solar_zenith = np.deg2rad(output['zenith'])  # spa
    # spencer
    eot = solarposition.equation_of_time_spencer71(times.dayofyear)
    hour_angle = np.deg2rad(solarposition.hour_angle(times, lon, eot))
    decl = solarposition.declination_spencer71(times.dayofyear)
    zenith_1 = solarposition.solar_zenith_analytical(lat_rad, hour_angle, decl)
    # pvcdrom and cooper
    eot = solarposition.equation_of_time_pvcdrom(times.dayofyear)
    hour_angle = np.deg2rad(solarposition.hour_angle(times, lon, eot))
    decl = solarposition.declination_cooper69(times.dayofyear)
    zenith_2 = solarposition.solar_zenith_analytical(lat_rad, hour_angle, decl)
    assert np.allclose(zenith_1, solar_zenith, atol=0.015)
    assert np.allclose(zenith_2, solar_zenith, atol=0.025)


def test_analytical_azimuth():
    times = pd.date_range(start="1/1/2015 0:00", end="12/31/2015 23:00",
                          freq="H").tz_localize('Etc/GMT+8')
    lat, lon = 37.8, -122.25
    lat_rad = np.deg2rad(lat)
    output = solarposition.spa_python(times, lat, lon, 100)
    solar_azimuth = np.deg2rad(output['azimuth'])  # spa
    solar_zenith = np.deg2rad(output['zenith'])
    # spencer
    eot = solarposition.equation_of_time_spencer71(times.dayofyear)
    hour_angle = np.deg2rad(solarposition.hour_angle(times, lon, eot))
    decl = solarposition.declination_spencer71(times.dayofyear)
    zenith = solarposition.solar_zenith_analytical(lat_rad, hour_angle, decl)
    azimuth_1 = solarposition.solar_azimuth_analytical(lat_rad, hour_angle,
                                                       decl, zenith)
    # pvcdrom and cooper
    eot = solarposition.equation_of_time_pvcdrom(times.dayofyear)
    hour_angle = np.deg2rad(solarposition.hour_angle(times, lon, eot))
    decl = solarposition.declination_cooper69(times.dayofyear)
    zenith = solarposition.solar_zenith_analytical(lat_rad, hour_angle, decl)
    azimuth_2 = solarposition.solar_azimuth_analytical(lat_rad, hour_angle,
                                                       decl, zenith)

    idx = np.where(solar_zenith < np.pi/2)
    assert np.allclose(azimuth_1[idx], solar_azimuth.values[idx], atol=0.01)
    assert np.allclose(azimuth_2[idx], solar_azimuth.values[idx], atol=0.017)

    # test for NaN values at boundary conditions (PR #431)
    test_angles = np.radians(np.array(
                   [[   0., -180.,  -20.],
                    [   0.,    0.,   -5.],
                    [   0.,    0.,    0.],
                    [   0.,    0.,   15.],
                    [   0.,  180.,   20.],
                    [  30.,    0.,  -20.],
                    [  30.,    0.,   -5.],
                    [  30.,    0.,    0.],
                    [  30.,  180.,    5.],
                    [  30.,    0.,   10.],
                    [ -30.,    0.,  -20.],
                    [ -30.,    0.,  -15.],
                    [ -30.,    0.,    0.],
                    [ -30., -180.,    5.],
                    [ -30.,  180.,   10.]]))

    zeniths = solarposition.solar_zenith_analytical(*test_angles.T)
    azimuths = solarposition.solar_azimuth_analytical(*test_angles.T,
                                                      zenith=zeniths)

    assert not np.isnan(azimuths).any()


def test_hour_angle():
    """
    Test conversion from hours to hour angles in degrees given the following
    inputs from NREL SPA calculator at Golden, CO
    date,times,eot,sunrise,sunset
    1/2/2015,7:21:55,-3.935172,-70.699400,70.512721
    1/2/2015,16:47:43,-4.117227,-70.699400,70.512721
    1/2/2015,12:04:45,-4.026295,-70.699400,70.512721
    """
    longitude = -105.1786  # degrees
    times = pd.DatetimeIndex([
        '2015-01-02 07:21:55.2132',
        '2015-01-02 16:47:42.9828',
        '2015-01-02 12:04:44.6340'
    ]).tz_localize('Etc/GMT+7')
    eot = np.array([-3.935172, -4.117227, -4.026295])
    hours = solarposition.hour_angle(times, longitude, eot)
    expected = (-70.682338, 70.72118825000001, 0.000801250)
    # FIXME: there are differences from expected NREL SPA calculator values
    # sunrise: 4 seconds, sunset: 48 seconds, transit: 0.2 seconds
    # but the differences may be due to other SPA input parameters
    assert np.allclose(hours, expected)


def test_sun_rise_set_transit_geometric(expected_rise_set_spa, golden_mst):
    """Test geometric calculations for sunrise, sunset, and transit times"""
    times = expected_rise_set_spa.index
    latitude = golden_mst.latitude
    longitude = golden_mst.longitude
    eot = solarposition.equation_of_time_spencer71(times.dayofyear)  # minutes
    decl = solarposition.declination_spencer71(times.dayofyear)  # radians
    sr, ss, st = solarposition.sun_rise_set_transit_geometric(
        times, latitude=latitude, longitude=longitude, declination=decl,
        equation_of_time=eot)
    # sunrise: 2015-01-02 07:26:39.763224487, 2015-08-02 05:04:35.688533801
    # sunset:  2015-01-02 16:41:29.951096777, 2015-08-02 19:09:46.597355085
    # transit: 2015-01-02 12:04:04.857160632, 2015-08-02 12:07:11.142944443
    test_sunrise = solarposition._times_to_hours_after_local_midnight(sr)
    test_sunset = solarposition._times_to_hours_after_local_midnight(ss)
    test_transit = solarposition._times_to_hours_after_local_midnight(st)
    # convert expected SPA sunrise, sunset, transit to local datetime indices
    expected_sunrise = pd.DatetimeIndex(expected_rise_set_spa.sunrise.values,
                                        tz='UTC').tz_convert(golden_mst.tz)
    expected_sunset = pd.DatetimeIndex(expected_rise_set_spa.sunset.values,
                                       tz='UTC').tz_convert(golden_mst.tz)
    expected_transit = pd.DatetimeIndex(expected_rise_set_spa.transit.values,
                                        tz='UTC').tz_convert(golden_mst.tz)
    # convert expected times to hours since midnight as arrays of floats
    expected_sunrise = solarposition._times_to_hours_after_local_midnight(
        expected_sunrise)
    expected_sunset = solarposition._times_to_hours_after_local_midnight(
        expected_sunset)
    expected_transit = solarposition._times_to_hours_after_local_midnight(
        expected_transit)
    # geometric time has about 4-6 minute error compared to SPA sunset/sunrise
    expected_sunrise_error = np.array(
        [0.07910089555555544, 0.06908014805555496])  # 4.8[min], 4.2[min]
    expected_sunset_error = np.array(
        [-0.1036246955555562, -0.06983406805555603])  # -6.2[min], -4.2[min]
    expected_transit_error = np.array(
        [-0.011150788888889096, 0.0036508177777765383])  # -40[sec], 13.3[sec]
    assert np.allclose(test_sunrise, expected_sunrise,
                       atol=np.abs(expected_sunrise_error).max())
    assert np.allclose(test_sunset, expected_sunset,
                       atol=np.abs(expected_sunset_error).max())
    assert np.allclose(test_transit, expected_transit,
                       atol=np.abs(expected_transit_error).max())


def test_spencer():
    """ test for the calculation based on spencer 1972 """
    latitude = 32.2
    longitude = -111
    to_test = solarposition.spencer(times,
                                    latitude,
                                    longitude)
    np.testing.assert_array_almost_equal(
        to_test.zenith.values,
        np.array(
            [60.22983081, 63.34160549, 66.43402974, 69.50442757,
             72.54998112, 75.56768635, 78.55430954, 81.50634428,
             84.41996557, 87.29098172, 90.11478465, 92.88629624,
             95.5999124, 98.24944658, 100.8280723, 103.32826814,
             105.74177, 108.05953355, 110.2717148, 112.36767828,
             114.33604136, 116.16476744, 117.84132062, 119.35289073,
             120.68669583, 121.83036181, 122.77236599, 123.50252116,
             124.01246252, 124.29608957, 124.34991302, 124.17326279,
             123.76832862, 123.14002805, 122.29571959, 121.24479895,
             119.99822708, 118.56803917, 116.96687825, 115.2075839,
             113.30285269, 111.26497738, 109.10566098, 106.83589588,
             104.46589822, 102.00508468, 99.46207977, 96.84474584,
             94.16022733, 91.4150024, 88.61493961, 85.76535568,
             82.87107126, 79.93646571, 76.96552925, 73.9619112,
             70.92896677, 67.86980202, 64.78731633, 61.68424611,
             58.56321009, 55.42675712, 52.27742195, 49.11779228,
             45.95059256, 42.77879745, 39.60579061, 36.43559544,
             33.27322843, 30.12525958, 27.00073787, 23.9127918,
             20.88152421, 17.93949588, 15.1425391, 12.59129497,
             10.47023949, 9.08837272, 8.80250072, 9.71016099,
             11.53222951, 13.91268049, 16.61145941, 19.49478556,
             22.48953441, 25.55411892, 28.66354138, 31.80189434,
             34.95848314, 38.12572775, 41.29797352, 44.47078752,
             47.64052254, 50.80403616, 53.95850349, 57.10128521,
             60.19189057, 63.30411711, 66.39705174, 69.46801919,
             72.51420324, 75.53260234, 78.51998605, 81.472852,
             84.38737999, 87.25938384, 90.08426172, 92.85694249,
             95.5718297, 98.22274508, 100.80287095, 103.30469507,
             105.71996264, 108.03963849, 110.2538871, 112.35208031,
             114.32284084, 116.15413465, 117.83342459, 119.34789444,
             120.68475046, 121.83160023, 122.77689598, 123.51041877,
             124.02376646, 124.31079722, 124.36797868, 124.19459854,
             123.79280774, 123.16749074, 122.32598014, 121.27765375,
             120.03346257, 118.60543913, 117.0062299, 115.2486827,
             113.34550579, 111.30900578, 109.15090068, 106.88219818,
             104.51312938, 102.0531253, 99.51082375, 96.8940993,
             94.21010748, 91.46533641, 88.6656635, 85.81641338,
             82.92241365, 79.98804982, 77.01731753, 74.01387089,
             70.98106934, 67.92202265, 64.8396335, 61.73664121,
             58.6156671, 55.47926225, 52.3299633, 49.17035952,
             46.00317641, 42.83138906, 39.65838052, 36.48817188,
             33.32577437, 30.17774741, 27.05311932, 23.9649783,
             20.93334666, 17.99061869, 15.19226962, 12.63816039,
             10.51116169, 9.11789559, 8.81495244, 9.70524228,
             11.51457879, 13.88698965, 16.5808083, 19.46100386,
             22.45371889, 25.51695489, 28.62547834, 31.76324145,
             34.91946275, 38.08650657, 41.25868161, 44.43153022,
             47.60138827, 50.76510173, 53.91983782, 57.06295199,
             60.15714089]
        )
    )

    np.testing.assert_array_almost_equal(
        to_test.azimuth.values,
        np.array(
            [280.45072709, 282.11633942, 283.7817931, 285.45643927,
             287.14893238, 288.86746085, 290.61992811, 292.41409693,
             294.2577037, 296.15854802, 298.12456058, 300.16384793,
             302.28471289, 304.49564761, 306.80529163, 309.22234814,
             311.75544988, 314.41296211, 317.20271279, 320.13164164,
             323.20536119, 326.42763345, 329.79977946, 333.32005237,
             336.98302936, 340.77909977, 344.69413863, 348.7094615,
             352.80214024, 356.94571294, 1.11126554, 5.26879407,
             9.38869489, 13.44320688, 17.40764042, 21.26126792,
             24.98782151, 28.57560944, 32.01730949, 35.30953094,
             38.45224149, 41.44814098, 44.3020479, 47.02034252,
             49.61048749, 52.08063548, 54.43932309, 56.69524105,
             58.85707173, 60.93338359, 62.93257078, 64.86283035,
             66.73217087, 68.54844619, 70.31941227, 72.05280693,
             73.75645209, 75.438383, 77.10701127, 78.7713305,
             80.44118034, 82.12759231, 83.84324978, 85.60311369,
             87.42529319, 89.33228209, 91.3527583, 93.52426951,
             95.8973478, 98.54199782, 101.55824322, 105.09379965,
             109.37452354, 114.75768924, 121.82299993, 131.5046235,
             145.14346316, 163.81906282, 186.0060095, 206.77205045,
             222.71272671, 234.05797936, 242.20138813, 248.2792225,
             253.019093, 256.86789112, 260.10425777, 262.90749472,
             265.39707784, 267.65536873, 269.74098333, 271.69690061,
             273.5555329, 275.34199145, 277.07624914, 278.77460915,
             280.40857909, 282.07488513, 283.74081962, 285.41575589,
             287.10836546, 288.8268498, 290.57912232, 292.37295337,
             294.21608515, 296.11632179, 298.08159771, 300.12002294,
             302.239904, 304.44973753, 306.75816911, 309.1739102,
             311.70560495, 314.36163417, 317.1498465, 320.07720857,
             323.14936689, 326.37012512, 329.7408537, 333.25986209,
             336.9217888, 340.71708646, 344.63169085, 348.64697055,
             352.74003696, 356.88444935, 1.05129299, 5.21054047,
             9.33254359, 13.38947986, 17.35658738, 21.21306193,
             24.94256075, 28.53332377, 31.9779703, 35.2730625,
             38.41853254, 41.41705548, 44.27343437, 46.99404174,
             49.58633845, 52.05847988, 54.4190084, 56.6766226,
             58.84001384, 60.91776009, 62.91826507, 64.84973506,
             66.72018726, 68.5374834, 70.30938637, 72.04363982,
             73.74807028, 75.43071613, 77.09999039, 78.7648859,
             80.43523879, 82.12207338, 83.83806078, 85.59814255,
             87.420398, 89.32727554, 91.34738372, 93.51816308,
             95.88997704, 98.53255847, 101.54548029, 105.07568588,
             109.34766226, 114.71618082, 121.75632435, 131.39424631,
             144.96107287, 163.54312836, 185.67160477, 206.4608716,
             222.464149, 233.86642744, 242.05129037, 248.1579107,
             252.91794505, 256.78120715, 260.02821949, 262.83947716,
             265.33522383, 267.59832701, 269.68774445, 271.64669179,
             273.50774886, 275.29614781, 277.03195029, 278.73152558,
             280.35936035]
        )
    )


# put numba tests at end of file to minimize reloading
@requires_numba
def test_spa_python_numba_physical(expected_solpos, golden_mst):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 12, 30, 30),
                          periods=1, freq='D', tz=golden_mst.tz)
    with warnings.catch_warnings():
        # don't warn on method reload or num threads
        # ensure that numpy is the most recently used method so that
        # we can use the warns filter below
        warnings.simplefilter("ignore")
        ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                              golden_mst.longitude,
                                              pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numpy', numthreads=1)
    with pytest.warns(UserWarning):
        ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                              golden_mst.longitude,
                                              pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numba', numthreads=1)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_numba
def test_spa_python_numba_physical_dst(expected_solpos, golden):
    times = pd.date_range(datetime.datetime(2003, 10, 17, 13, 30, 30),
                          periods=1, freq='D', tz=golden.tz)

    with warnings.catch_warnings():
        # don't warn on method reload or num threads
        warnings.simplefilter("ignore")
        ephem_data = solarposition.spa_python(times, golden.latitude,
                                              golden.longitude, pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numba', numthreads=1)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])

    with pytest.warns(UserWarning):
        # test that we get a warning when reloading to use numpy only
        ephem_data = solarposition.spa_python(times, golden.latitude,
                                              golden.longitude,
                                              pressure=82000,
                                              temperature=11, delta_t=67,
                                              atmos_refract=0.5667,
                                              how='numpy', numthreads=1)
