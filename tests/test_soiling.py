"""Test losses"""

import datetime
import numpy as np
import pandas as pd
from .conftest import assert_series_equal
from pvlib.soiling import hsu, kimber, _kimber_cleaning_events
from pvlib.iotools import read_tmy3
from .conftest import TESTS_DATA_DIR, PVLIB_DATA_DIR
import pytest


@pytest.fixture
def expected_output():
    # Sample output (calculated manually)
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')

    expected_no_cleaning = pd.Series(
        data=[0.96998483, 0.94623958, 0.92468139, 0.90465654, 0.88589707,
              0.86826366, 0.85167258, 0.83606715, 0.82140458, 0.80764919,
              0.79476875, 0.78273241, 0.77150951, 0.76106905, 0.75137932,
              0.74240789, 0.73412165, 0.72648695, 0.71946981, 0.7130361,
              0.70715176, 0.70178307, 0.69689677, 0.69246034],
        index=dt)
    return expected_no_cleaning


@pytest.fixture
def expected_output_1():
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    expected_output_1 = pd.Series(
        data=[0.98484972, 0.97277367, 0.96167471, 0.95119603, 1.,
              0.98484972, 0.97277367, 0.96167471, 1., 1.,
              0.98484972, 0.97277367, 0.96167471, 0.95119603, 0.94118234,
              0.93154854, 0.922242, 0.91322759, 0.90448058, 0.89598283,
              0.88772062, 0.87968325, 0.8718622, 0.86425049],
        index=dt)
    return expected_output_1


@pytest.fixture
def expected_output_2():
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    expected_output_2 = pd.Series(
        data=[0.95036261, 0.91178179, 0.87774818, 0.84732079, 1.,
              1., 1., 0.95036261, 1., 1.,
              1., 1., 0.95036261, 0.91178179, 0.87774818,
              0.84732079, 0.8201171, 1., 1., 1.,
              1., 0.95036261, 0.91178179, 0.87774818],
        index=dt)
    return expected_output_2


@pytest.fixture
def expected_output_3():
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    timedelta = [0, 0, 0, 0, 0, 30, 0, 30, 0, 30, 0, -30,
                 -30, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dt_new = dt + pd.to_timedelta(timedelta, 'm')
    expected_output_3 = pd.Series(
        data=[0.96576705, 0.9387675, 0.91437615, 0.89186852, 1.,
              1., 0.98093819, 0.9387675, 1., 1.,
              1., 1., 0.96576705, 0.9387675, 0.90291005,
              0.88122293, 0.86104089, 1., 1., 1.,
              0.96576705, 0.9387675, 0.91437615, 0.89186852],
        index=dt_new)
    return expected_output_3


@pytest.fixture
def rainfall_input():

    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    rainfall = pd.Series(
        data=[0., 0., 0., 0., 1., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0.,
              0., 0.3, 0.3, 0.3, 0.3, 0., 0., 0., 0.], index=dt)
    return rainfall


def test_hsu_no_cleaning(rainfall_input, expected_output):
    """Test Soiling HSU function"""

    rainfall = rainfall_input
    pm2_5 = 1.0
    pm10 = 2.0
    depo_veloc = {'2_5': 1.0e-5, '10': 1.0e-4}
    tilt = 0.
    expected_no_cleaning = expected_output

    result = hsu(rainfall=rainfall, cleaning_threshold=10., surface_tilt=tilt,
                 pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                 rain_accum_period=pd.Timedelta('1h'))
    assert_series_equal(result, expected_no_cleaning)


def test_hsu(rainfall_input, expected_output_2):
    """Test Soiling HSU function with cleanings"""

    rainfall = rainfall_input
    pm2_5 = 1.0
    pm10 = 2.0
    depo_veloc = {'2_5': 1.0e-4, '10': 1.0e-4}
    tilt = 0.

    # three cleaning events at 4:00-6:00, 8:00-11:00, and 17:00-20:00
    result = hsu(rainfall=rainfall, cleaning_threshold=0.5, surface_tilt=tilt,
                 pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                 rain_accum_period=pd.Timedelta('3h'))

    assert_series_equal(result, expected_output_2)


def test_hsu_defaults(rainfall_input, expected_output_1):
    """
    Test Soiling HSU function with default deposition velocity and default rain
    accumulation period.
    """
    result = hsu(rainfall=rainfall_input, cleaning_threshold=0.5,
                 surface_tilt=0.0, pm2_5=1.0e-2, pm10=2.0e-2)
    assert np.allclose(result.values, expected_output_1)


def test_hsu_variable_time_intervals(rainfall_input, expected_output_3):
    """
    Test Soiling HSU function with variable time intervals.
    """
    depo_veloc = {'2_5': 1.0e-4, '10': 1.0e-4}
    rain = pd.DataFrame(data=rainfall_input)
    # define time deltas in minutes
    timedelta = [0, 0, 0, 0, 0, 30, 0, 30, 0, 30, 0, -30,
                 -30, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rain['mins_added'] = pd.to_timedelta(timedelta, 'm')
    rain['new_time'] = rain.index + rain['mins_added']
    rain_var_times = rain.set_index('new_time').iloc[:, 0]
    result = hsu(
        rainfall=rain_var_times, cleaning_threshold=0.5, surface_tilt=50.0,
        pm2_5=1, pm10=2, depo_veloc=depo_veloc,
        rain_accum_period=pd.Timedelta('2h'))
    assert np.allclose(result, expected_output_3)


@pytest.fixture
def greensboro_rain():
    # get TMY3 data with rain
    greensboro, _ = read_tmy3(
        PVLIB_DATA_DIR / "723170TYA.CSV", coerce_year=1990, map_variables=True
    )
    return greensboro["Lprecip depth (mm)"]


@pytest.fixture
def expected_kimber_nowash():
    return pd.read_csv(
        TESTS_DATA_DIR / 'greensboro_kimber_soil_nowash.dat',
        parse_dates=True, index_col='timestamp')


def test_kimber_nowash(greensboro_rain, expected_kimber_nowash):
    """Test Kimber soiling model with no manual washes"""
    # Greensboro typical expected annual rainfall is 8345mm
    assert greensboro_rain.sum() == 8345
    # calculate soiling with no wash dates
    nowash = kimber(greensboro_rain)
    # test no washes
    assert np.allclose(nowash.values, expected_kimber_nowash['soiling'].values)


@pytest.fixture
def expected_kimber_manwash():
    return pd.read_csv(
        TESTS_DATA_DIR / 'greensboro_kimber_soil_manwash.dat',
        parse_dates=True, index_col='timestamp')


def test_kimber_manwash(greensboro_rain, expected_kimber_manwash):
    """Test Kimber soiling model with a manual wash"""
    # a manual wash date
    manwash = [datetime.date(1990, 2, 15), ]
    # calculate soiling with manual wash
    manwash = kimber(greensboro_rain, manual_wash_dates=manwash)
    # test manual wash
    assert np.allclose(
        manwash.values,
        expected_kimber_manwash['soiling'].values)


@pytest.fixture
def expected_kimber_norain():
    # expected soiling reaches maximum
    soiling_loss_rate = 0.0015
    max_loss_rate = 0.3
    norain = np.ones(8760) * soiling_loss_rate/24
    norain[0] = 0.0
    norain = np.cumsum(norain)
    return np.where(norain > max_loss_rate, max_loss_rate, norain)


def test_kimber_norain(greensboro_rain, expected_kimber_norain):
    """Test Kimber soiling model with no rain"""
    # a year with no rain
    norain = pd.Series(0, index=greensboro_rain.index)
    # calculate soiling with no rain
    norain = kimber(norain)
    # test no rain, soiling reaches maximum
    assert np.allclose(norain.values, expected_kimber_norain)


@pytest.fixture
def expected_kimber_initial_soil():
    # expected soiling reaches maximum
    soiling_loss_rate = 0.0015
    max_loss_rate = 0.3
    norain = np.ones(8760) * soiling_loss_rate/24
    norain[0] = 0.1
    norain = np.cumsum(norain)
    return np.where(norain > max_loss_rate, max_loss_rate, norain)


def test_kimber_initial_soil(greensboro_rain, expected_kimber_initial_soil):
    """Test Kimber soiling model with initial soiling"""
    # a year with no rain
    norain = pd.Series(0, index=greensboro_rain.index)
    # calculate soiling with no rain
    norain = kimber(norain, initial_soiling=0.1)
    # test no rain, soiling reaches maximum
    assert np.allclose(norain.values, expected_kimber_initial_soil)


def test_kimber_threshold_inclusive():
    """Rain equal to cleaning_threshold cleans the panels (GH 2796)"""
    times = pd.date_range('2020-01-01', periods=4, freq='D', tz='UTC')
    soiling = kimber(pd.Series([0, 0, 6.0, 0], times), cleaning_threshold=6)
    assert soiling.iloc[2] == 0


@pytest.mark.parametrize('freq, first_clean, last_clean', [
    ('15min', '2020-01-04 14:00', '2020-01-05 14:00'),
    ('1h', '2020-01-04 14:00', '2020-01-05 14:00'),
    ('1D', '2020-01-05', '2020-01-06'),  # daily totals labeled at day end
])
def test_kimber_grace_period_resolution(freq, first_clean, last_clean):
    """Grace period lasts grace_period days from the cleaning event,
    regardless of input resolution (GH 2796)"""
    times = pd.date_range('2020-01-01', '2020-01-09', freq='15min', tz='UTC')
    rain = pd.Series(0.0, times)
    rain['2020-01-04 12:15':'2020-01-04 15:00'] = 10 / 12  # 10 mm storm
    rain = rain.resample(freq, label='right', closed='right').sum()
    soiling = kimber(rain, grace_period=1)
    clean = soiling.index[soiling == 0][1:]  # skip initial_soiling at t=0
    assert clean[0] == pd.Timestamp(first_clean, tz='UTC')
    assert clean[-1] == pd.Timestamp(last_clean, tz='UTC')


@pytest.mark.parametrize('rain_mm, event', [
    # {hour after 2020-01-02 00:00: rain [mm]}, last cleaning event hour
    ({8: 5, 9: 5, 19: 0.2}, 9),  # drizzle after a dry gap
    ({8: 5, 9: 5, 10: 0.1, 11: 0.1, 12: 0.1}, 9),  # drizzle, no gap
    ({8: 5, 9: 5, 18: 5, 19: 5}, 19),  # second storm restarts grace
    ({8: 2, 9: 2, 10: 2, 11: 2}, 10),  # grace starts at threshold
    ({8: 2, 9: 2, 12: 2, 13: 2}, 12),  # same with a dry gap
    ({8: 5, 9: np.nan, 10: 5}, 10),  # missing rainfall counts as zero
    ({h: 0.2 for h in range(30)}, None),  # 6 mm, but not within 24 h
    ({8: 2}, None),  # below threshold
])
def test_kimber_cleaning_events(rain_mm, event):
    """Only cleaning_threshold of new rain within rain_accum_period makes a
    cleaning event (GH 2796)"""
    times = pd.date_range('2020-01-01', periods=24 * 5, freq='h', tz='UTC')
    rain = pd.Series(0.0, times)
    for hour, mm in rain_mm.items():
        rain.iloc[24 + hour] = mm
    soiling = kimber(rain, grace_period=1)
    clean = soiling.index[soiling == 0][1:]  # skip initial_soiling at t=0
    if event is None:
        assert len(clean) == 0
    else:
        expected = times[24 + event] + pd.Timedelta('1D')
        assert clean[-1] == expected


def test_kimber_cleaning_events_daily():
    """With daily input, a day is a cleaning event if its rain meets
    cleaning_threshold, as in Kimber 2006 (GH 2796)"""
    rng = np.random.default_rng(2796)
    times = pd.date_range('2020-01-01', periods=3 * 365, freq='D', tz='UTC')
    rain = rng.random(len(times)) * (rng.random(len(times)) < 0.3) * 15
    events = _kimber_cleaning_events(
        times.tz_convert(None).to_numpy(), rain, 6, np.timedelta64(24, 'h'))
    np.testing.assert_array_equal(events, rain >= 6)


def test_kimber_cleaning_events_float_rounding():
    """Rain summing exactly to cleaning_threshold triggers a cleaning event
    despite floating point error from earlier rain (GH 2796)"""
    rng = np.random.default_rng(2796)
    times = pd.date_range('2020-01-01', periods=24 * 4 * 60, freq='15min')
    # 0.3 mm tips, stored slightly below 0.3: 50 days of random history,
    # 2 dry days, then exactly 20 tips (6.0 mm) in 5 hours
    rain = rng.integers(0, 9, len(times)) * (rng.random(len(times)) < 0.2)
    rain = rain * 0.3
    storm = 24 * 4 * 52
    rain[24 * 4 * 50:] = 0
    rain[storm:storm + 20] = 0.3
    events = _kimber_cleaning_events(
        times.to_numpy(), rain, 6, np.timedelta64(24, 'h'))
    assert events[storm:].any()
