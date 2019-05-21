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


def test_spencer_mc():
    """ test for the calculation based on spencer 1972 """
    latitude = 48.367073
    longitude = 10.868378
    to_test = solarposition.spencer_mc(times,
                             latitude,
                             longitude)
    np.testing.assert_array_almost_equal(
        to_test.zenith.values,
        np.array(
            [107.5944161, 107.08772172, 106.43116829, 105.62900042,
             104.68621177, 103.60841558, 102.40171152, 101.07255494,
             99.62763476, 98.07376382, 96.41778336, 94.66648432,
             92.82654495, 90.90448339, 88.90662551, 86.83908587,
             84.70775942, 82.51832403, 80.27625186, 77.98682746,
             75.65517374, 73.28628434, 70.88506165, 68.45636234,
             66.0050506, 63.53605921, 61.05446218, 58.56556086,
             56.07498558, 53.58881946, 51.11374884, 48.65724635,
             46.22779685, 43.83517458, 41.49078009, 39.20804631,
             37.00291337, 34.89435623, 32.90492124, 31.06117096,
             29.39385375, 27.93751303, 26.72916357, 25.80569641,
             25.19996401, 24.93609994, 25.02533181, 25.46381181,
             26.23337616, 27.30490594, 28.64295418, 30.21015627,
             31.97050468, 33.89127116, 35.94381137, 38.10363121,
             40.35004899, 42.66568743, 45.03593362, 47.44843417,
             49.89265446, 52.35950849, 54.8410528, 57.33023629,
             59.8206975, 62.30659851, 64.78248863, 67.24319292,
             69.68371835, 72.09917489, 74.4847094, 76.83544826,
             79.14644817, 81.41265486, 83.62886755, 85.78970975,
             87.88960763, 89.92277472, 91.88320492, 93.76467562,
             95.5607608, 97.26485655, 98.8702211, 100.37002926,
             101.7574431, 103.02569948, 104.16821258, 105.17869036,
             106.05126175, 106.78060921, 107.36210092, 107.79191584,
             108.06715341, 108.18592116, 108.14739387, 107.9518398,
             107.61767954, 107.11306117, 106.45849746, 105.65822415,
             104.71722821, 103.64111837, 102.43599187, 101.10830346,
             99.66474308, 98.11212592, 96.45729665, 94.7070504,
             92.86807018, 90.9468792, 88.94980851, 86.88297787,
             84.75228725, 82.56341928, 80.32185056, 78.03286967,
             75.70160305, 73.33304741, 70.93210757, 68.50364202,
             66.05251605, 63.58366268, 61.10215525, 58.61329329,
             56.12270393, 53.63646542, 51.16125711, 48.70454205,
             46.27479211, 43.88176429, 41.53683649, 39.25341219,
             37.04739373, 34.93770849, 32.94684449, 31.10129575,
             29.43173548, 27.97263389, 26.7609499, 25.83355775,
             25.22335515, 24.95459651, 25.03869819, 25.47203708,
             26.23666504, 27.30363156, 28.63759291, 30.20122571,
             31.95851686, 33.87670365, 35.92709343, 38.08514007,
             40.33011209, 42.64458753, 45.01391492, 47.42570839,
             49.86940645, 52.33590108, 54.81723096, 57.30633064,
             59.79682723, 62.2828738, 64.75901269, 67.22006367,
             69.66102988, 72.0770186, 74.46317504, 76.81462477,
             79.12642433, 81.39351992, 83.61071161, 85.77262406,
             87.87368475, 89.90810859, 91.86989066, 93.75280924,
             95.55043872, 97.25617485, 98.86327462, 100.36491047,
             101.75424074, 103.024497, 104.16908649, 105.1817084,
             106.05648115, 106.78807492, 107.3718441, 107.80395262,
             108.08148404, 108.20252959, 108.1662479, 107.97289173,
             107.64770976]
        )
    )

    np.testing.assert_array_almost_equal(
        to_test.azimuth.values,
        np.array(
            [9.95354884, 13.51797616, 17.04447274, 20.52491225,
             23.95238719, 27.32131453, 30.62748235, 33.86804529,
             37.04147561, 40.14748222, 43.18691015, 46.1616283,
             49.07441566, 51.92885441, 54.72923301, 57.48046422,
             60.18802201, 62.85789652, 65.49656941, 68.11101133,
             70.70870005, 73.29766135, 75.88653487, 78.48466488,
             81.10221958, 83.75034334, 86.44134434, 89.18892358,
             92.00845226, 94.91730112, 97.93522718, 101.08482003,
             104.39199808, 107.88653257, 111.60254901, 115.57890087,
             119.8592317, 124.49141579, 129.52588659, 135.01216661,
             140.99278628, 147.49394635, 154.51314061, 162.00583842,
             169.87594269, 177.97643578, 186.12466927, 194.13016535,
             201.82539566, 209.08806319, 215.84877252, 222.08574604,
             227.81263221, 233.06524098, 237.89058053, 242.3392101,
             246.46057573, 250.30054442, 253.90036735, 257.29648187,
             260.52076061, 263.60096725, 266.56127763, 269.42279422,
             272.20401931, 274.92127106, 277.58904086, 280.22029634,
             282.82673312, 285.41898225, 288.00677995, 290.59910223,
             293.20426943, 295.83002468, 298.48358643, 301.17167776,
             303.9005345, 306.67589094, 309.50294481, 312.38630353,
             315.32991074, 318.33695647, 321.40977511, 324.54973374,
             327.75711805, 331.0310254, 334.36927221, 337.76832766,
             341.22328647, 344.72788848, 348.27459447, 351.85472406,
             355.45865247, 359.07606104, 2.69623101, 6.30836085,
             9.90413618, 13.46983714, 16.99770579, 20.47959425,
             23.90857277, 27.27903624, 30.58675167, 33.82885422,
             37.00379863, 40.1112785, 43.15212578, 46.12819847,
             49.04226661, 51.89790514, 54.69939668, 57.45164925,
             60.1601328, 62.83083386, 65.47023059, 68.08528982,
             70.68348498, 73.27283649, 75.86197737, 78.46024348,
             81.07779237, 83.72575502, 86.41642271, 89.16347527,
             91.9822575, 94.8901073, 97.90674111, 101.05469855,
             104.35983703, 107.85185424, 111.56478847, 115.53739258,
             119.81319937, 124.43997055, 129.46804248, 134.94688671,
             140.91907623, 147.41101064, 154.42060141, 161.90398226,
             169.76591074, 177.86024433, 186.00498616, 194.00988407,
             201.70716439, 208.97394261, 215.74010434, 221.98320309,
             227.71636618, 232.97505067, 237.80605645, 242.2598406,
             246.38581686, 250.22986166, 253.83325807, 257.23248545,
             260.45946126, 263.54199277, 266.50429617, 269.36751012,
             272.15016893, 274.86861866, 277.53737498, 280.1694266,
             282.77648747, 285.3692047, 287.95732871, 290.54984827,
             293.15509539, 295.78082413, 298.43426346, 301.12214686,
             303.85072078, 306.62573055, 309.45238569, 312.3353063,
             315.27844985, 318.28502138, 321.35737159, 324.49688503,
             327.70386592, 330.97743082, 334.31541557, 337.71430836,
             341.1692218, 344.67391164, 348.22085175, 351.80137124,
             355.40585053, 359.02397146, 2.64501087, 6.2581581,
             9.85597076]
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
