import calendar
import datetime

import numpy as np
import pandas as pd

from pandas.util.testing import (assert_frame_equal, assert_series_equal,
                                 assert_index_equal)
from numpy.testing import assert_allclose
import pytest

from pvlib.location import Location
from pvlib import solarposition, spa

from conftest import (requires_ephem, needs_pandas_0_17,
                      requires_spa_c, requires_numba)


# setup times and locations to be tested.
times = pd.date_range(start=datetime.datetime(2014,6,24),
                      end=datetime.datetime(2014,6,26), freq='15Min')

tus = Location(32.2, -111, 'US/Arizona', 700) # no DST issues possible
# In 2003, DST in US was from April 6 to October 26
golden_mst = Location(39.742476, -105.1786, 'MST', 1830.14) # no DST issues possible
golden = Location(39.742476, -105.1786, 'America/Denver', 1830.14) # DST issues possible

times_localized = times.tz_localize(tus.tz)

tol = 5

@pytest.fixture()
def expected_solpos():
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
                        index=[['2003-10-17T12:30:30Z', '2003-10-18T12:30:30Z']])

# the physical tests are run at the same time as the NREL SPA test.
# pyephem reproduces the NREL result to 2 decimal places.
# this doesn't mean that one code is better than the other.

@requires_spa_c
def test_spa_c_physical(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_c(times, golden_mst.latitude,
                                     golden_mst.longitude,
                                     pressure=82000,
                                     temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_spa_c
def test_spa_c_physical_dst(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_c(times, golden.latitude,
                                     golden.longitude,
                                     pressure=82000,
                                     temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])

def test_spa_python_numpy_physical(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                          golden_mst.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])

def test_spa_python_numpy_physical_dst(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_python(times, golden.latitude,
                                          golden.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numpy')
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_numba
def test_spa_python_numba_physical(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.spa_python(times, golden_mst.latitude,
                                          golden_mst.longitude,
                                          pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numba', numthreads=1)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_numba
def test_spa_python_numba_physical_dst(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.spa_python(times, golden.latitude,
                                          golden.longitude, pressure=82000,
                                          temperature=11, delta_t=67,
                                          atmos_refract=0.5667,
                                          how='numba', numthreads=1)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@needs_pandas_0_17
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
def test_pyephem_physical(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.pyephem(times, golden_mst.latitude,
                                       golden_mst.longitude, pressure=82000,
                                       temperature=11)
    expected_solpos.index = times
    assert_frame_equal(expected_solpos.round(2),
                       ephem_data[expected_solpos.columns].round(2))

@requires_ephem
def test_pyephem_physical_dst(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30), periods=1,
                          freq='D', tz=golden.tz)
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

    epoch = datetime.datetime(1970,1,1)
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
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D')
    distance = solarposition.pyephem_earthsun_distance(times).values[0]
    assert_allclose(1, distance, atol=0.1)


def test_ephemeris_physical(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = solarposition.ephemeris(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_ephemeris_physical_dst(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.ephemeris(times, golden.latitude,
                                         golden.longitude, pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_ephemeris_physical_no_tz(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,19,30,30),
                          periods=1, freq='D')
    ephem_data = solarposition.ephemeris(times, golden_mst.latitude,
                                         golden_mst.longitude,
                                         pressure=82000,
                                         temperature=11)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


def test_get_solarposition_error():
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    with pytest.raises(ValueError):
        ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                     golden.longitude,
                                                     pressure=82000,
                                                     temperature=11,
                                                     method='error this')


@pytest.mark.parametrize(
    "pressure, expected", [
    (82000, expected_solpos()),
    (90000, pd.DataFrame(
        np.array([[  39.88997,   50.11003,  194.34024,   39.87205,   14.64151,
                     50.12795]]),
        columns=['apparent_elevation', 'apparent_zenith', 'azimuth', 'elevation',
                 'equation_of_time', 'zenith'],
        index=expected_solpos().index))
    ])
def test_get_solarposition_pressure(pressure, expected):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
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


@pytest.mark.parametrize(
    "altitude, expected", [
    (golden.altitude, expected_solpos()),
    (2000, pd.DataFrame(
        np.array([[  39.88788,   50.11212,  194.34024,   39.87205,   14.64151,
                     50.12795]]),
        columns=['apparent_elevation', 'apparent_zenith', 'azimuth', 'elevation',
                 'equation_of_time', 'zenith'],
        index=expected_solpos().index))
    ])
def test_get_solarposition_altitude(altitude, expected):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
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


@pytest.mark.parametrize(
    "delta_t, method, expected", [
    (None, 'nrel_numpy', expected_solpos_multi()),
    (67.0, 'nrel_numpy', expected_solpos_multi()),
    pytest.mark.xfail(raises=ValueError, reason = 'spa.calculate_deltat not implemented for numba yet')
    ((None, 'nrel_numba', expected_solpos_multi())),
    (67.0, 'nrel_numba', expected_solpos_multi())
    ])
def test_get_solarposition_deltat(delta_t, method, expected):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=2, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude,
                                                 pressure=82000,
                                                 delta_t=delta_t,
                                                 temperature=11,
                                                 method=method)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 5)
    ephem_data = np.round(ephem_data, 5)
    assert_frame_equal(this_expected, ephem_data[this_expected.columns])


def test_get_solarposition_no_kwargs(expected_solpos):
    times = pd.date_range(datetime.datetime(2003,10,17,13,30,30),
                          periods=1, freq='D', tz=golden.tz)
    ephem_data = solarposition.get_solarposition(times, golden.latitude,
                                                 golden.longitude)
    expected_solpos.index = times
    expected_solpos = np.round(expected_solpos, 2)
    ephem_data = np.round(ephem_data, 2)
    assert_frame_equal(expected_solpos, ephem_data[expected_solpos.columns])


@requires_ephem
def test_get_solarposition_method_pyephem(expected_solpos):
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
                              datetime.datetime(2015, 8, 2),]
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
    times = pd.DatetimeIndex(start="1/1/2015 0:00", end="12/31/2015 23:00",
                             freq="H")
    output = solarposition.spa_python(times, 37.8, -122.25, 100)
    eot = output['equation_of_time']
    eot_rng = eot.max() - eot.min()  # range of values, around 30 minutes
    eot_1 = solarposition.equation_of_time_spencer71(times.dayofyear)
    eot_2 = solarposition.equation_of_time_pvcdrom(times.dayofyear)
    assert np.allclose(eot_1 / eot_rng, eot / eot_rng, atol=0.3)  # spencer
    assert np.allclose(eot_2 / eot_rng, eot / eot_rng, atol=0.4)  # pvcdrom


def test_declination():
    times = pd.DatetimeIndex(start="1/1/2015 0:00", end="12/31/2015 23:00",
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
    times = pd.DatetimeIndex(start="1/1/2015 0:00", end="12/31/2015 23:00",
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
    times = pd.DatetimeIndex(start="1/1/2015 0:00", end="12/31/2015 23:00",
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
    assert np.allclose(azimuth_1[idx], solar_azimuth.as_matrix()[idx],
                       atol=0.01)
    assert np.allclose(azimuth_2[idx], solar_azimuth.as_matrix()[idx],
                       atol=0.017)

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

    zeniths  = solarposition.solar_zenith_analytical(*test_angles.T)
    azimuths = solarposition.solar_azimuth_analytical(*test_angles.T, zenith=zeniths)

    assert not np.isnan(azimuths).any()
